import copy
import math

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"

from dataclasses import asdict
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from phalp.configs.base import CACHE_DIR, FullConfig
from phalp.utils.smpl_utils import SMPL
from phalp.utils.utils_download import cache_url
from torchmetrics import MeanMetric
from PAR.evaluators.ava_map import compute_mAP
from PAR.datamodules.components.dexycb_utils import *
from PAR.models.utils.render_utils import render_hoi
from pytorch3d.io import load_obj
from scipy.spatial.transform import Rotation as R
import pyrender, trimesh
from pytorch3d.transforms import quaternion_to_matrix, so3_relative_angle

from PAR.utils import get_pylogger
from PAR.utils.utils_plot import read_ava_pkl
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
import torch.nn as nn
import transformers
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
import torch.nn.functional as F
import cv2
import re
from sklearn.metrics import average_precision_score
from iopath.common.file_io import g_pathmgr
import pickle, yaml

log = get_pylogger(__name__)

FPS = 7.0
MIN_TX = torch.Tensor([-0.13972618, -0.3158688, 0.49080718]).cuda()
MAX_TX = torch.Tensor([0.4228604, 0.2755164, 1.0603081]).cuda()
HOME_DIR = os.environ['HOME']

class Permute(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.permute(0, 2, 1)

class PAR_LitModule(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False,)
        self.cfg = self.hparams.cfg

        self.dexycb_root = os.path.join(HOME_DIR, 'datasets/dexycb')
        intrinsics_path = os.path.join(self.dexycb_root, 'calibration', 'intrinsics/%s_640x480.yml' % CAMERA_ID)
        with open(intrinsics_path, 'r') as f:
            self.intrinsics = yaml.load(f, Loader=yaml.FullLoader)
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.train_rot_loss_stdev = MeanMetric()
        self.train_tx_loss_stdev = MeanMetric()
        self.val_error_rel_rot = MeanMetric()
        self.val_error_tx = MeanMetric()
        self.ang_loss = MeanMetric()
        self.tx_loss = MeanMetric()
        if self.cfg.use_hand_loss and self.cfg.num_agents > 1:
            self.hand_loss = MeanMetric()
            self.train_hand_loss_stdev = MeanMetric()

        self.relative_norm = self.cfg.relative_norm
        
        vgpt_config = LlamaConfig()
        hsize = self.cfg.transformer.hsize
        isize = self.cfg.transformer.isize
        num_hidden_layers = self.cfg.transformer.depth
        num_attention_heads = self.cfg.transformer.heads
        vgpt_config.intermediate_size=isize
        vgpt_config.hidden_size=hsize
        vgpt_config.max_position_embeddings=4096+2
        vgpt_config.num_attention_heads=num_attention_heads
        vgpt_config.num_hidden_layers=num_hidden_layers
        vgpt_config.num_key_value_heads=num_attention_heads
        vgpt_config.use_cache=False
        if self.cfg.dropout_hidden > 0:
            vgpt_config.hidden_dropout_prob = self.cfg.dropout_hidden
        if self.cfg.dropout_attn > 0:
            vgpt_config.attention_probs_dropout_prob = self.cfg.dropout_attn
        self.vgpt = LlamaForCausalLM(vgpt_config)


        if self.cfg.num_agents > 1 and self.cfg.use_multiagent_pos_emb:
            self.agent_id_embedding = nn.Embedding(self.cfg.num_agents, hsize)
        
        self.hand_size = 3
        # self.obj_size = 3*4
        # cannot have self.cfg.pred_rot_only and self.cfg.pred_tx_only at the same time
        assert not (self.cfg.pred_rot_only and self.cfg.pred_tx_only)
        if self.cfg.pred_rot_only:
            self.obj_size = 4 # quaternion
        elif self.cfg.pred_tx_only:
            self.obj_size = 3 # translation
        else:
            self.obj_size = 4+3 # quaternion + translation

        if self.cfg.use_ln:
            if self.cfg.pred_rot_only:
                self.proj_obj_in = nn.Linear(self.obj_size, hsize)
            else:
                self.proj_obj_in = nn.Sequential(
                    nn.LayerNorm(self.obj_size),
                    nn.Linear(self.obj_size, hsize),
                )
            self.proj_hand_in = nn.Sequential(
                nn.LayerNorm(self.hand_size),
                nn.Linear(self.hand_size, hsize),
            )
        else:
            self.proj_obj_in = nn.Linear(self.obj_size, hsize)
            self.proj_hand_in = nn.Linear(self.hand_size, hsize)
        
        self.proj_hand_out = nn.Linear(hsize, self.hand_size)
        self.proj_obj_out = nn.Linear(hsize, self.obj_size)

        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)
                

    def on_train_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): self.trainer.datamodule.data_train.__getitem__(0)
        pass
    
    def unrelative_pose(self, pose, origin):
        return pose + origin

    def make_valid_quaternion_differentiable(self, quat):
        eps = 1e-8
        norm = torch.sqrt(torch.sum(quat**2, dim=-1, keepdim=True) + eps)  # Adding small epsilon to avoid division by zero
        normalized_q = quat / norm
        
        # Use a smooth approximation to flip the sign if w is negative
        w = normalized_q[..., 3]  # Assuming q is in [x, y, z, w] format
        sign = w / (torch.abs(w) + eps)  # Smooth approximation of sign(w)
        
        # Apply the sign to the entire quaternion
        valid_q = normalized_q * sign.unsqueeze(-1)
        
        return valid_q

        def rotate_vector_batch(quat, vec):
            # Conjugate is now [x, y, z, w] -> [-x, -y, -z, w]
            q_conj = torch.cat([-quat[..., :3], quat[..., 3:]], dim=-1)
            # Append 0 as the w component to the vector
            v_quat = torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)
            # Perform rotation
            rotated = quaternion_multiply_batch(quaternion_multiply_batch(quat, v_quat), q_conj)
            # Return only the vector part (xyz)
            return rotated[..., :3]
        
        x_expanded = x.unsqueeze(1).expand(-1, pred_quat.shape[1], -1, -1)
        pred_quat_expanded = pred_quat.unsqueeze(2).expand(-1, -1, x_expanded.shape[2], -1)
        gt_quat_expanded = gt_quat.unsqueeze(2).expand(-1, -1, x_expanded.shape[2], -1)

        pred_rotated = rotate_vector_batch(pred_quat_expanded, x_expanded)
        gt_rotated = rotate_vector_batch(gt_quat_expanded, x_expanded)

        if self.training and self.cfg.check_stdev:
            return F.mse_loss(pred_rotated, gt_rotated), ((pred_rotated - gt_rotated)**2).std()
        else:
            return F.mse_loss(pred_rotated, gt_rotated), None
    
    def quaternion_loss(self, pred_quat, gt_quat):
        if self.training and self.cfg.check_stdev:
            return (1-torch.abs((pred_quat * gt_quat).sum(dim=-1))).mean(), (1-torch.abs((pred_quat * gt_quat).sum(dim=-1))).std()
        else:
            return (1-torch.abs((pred_quat * gt_quat).sum(dim=-1))).mean(), None

    def eval_rot_error(self, pred_quat, gt_quat):
        pred_quat = pred_quat.squeeze()
        gt_quat = gt_quat.squeeze()
        rot_gt = quaternion_to_matrix(gt_quat)
        rot_pred = quaternion_to_matrix(pred_quat)
        rel_angle = so3_relative_angle(rot_gt, rot_pred).mean()
        return rel_angle
    
    def eval_tx_error(self, pred_tx, gt_tx):
        return F.mse_loss(pred_tx, gt_tx)

    def step(self, batch: Any):
        obj = batch['gt_obj'] # bs, t, 7
        if self.cfg.pred_rot_only:
            obj = obj[:, :, :4]
        elif self.cfg.pred_tx_only:
            obj = obj[:, :, 4:]
        bs, t = obj.shape[:2]
        obj = obj.view(bs, t, -1)
        obj_in = self.proj_obj_in(obj.to(torch.float32))
        if self.cfg.num_agents == 1:
            interleaved_hand_obj = obj_in
        else:
            hands = batch['gt_hand'] # bs, t, 3
            hands_in = self.proj_hand_in(hands.to(torch.float32))
            hands_obj = torch.stack((hands_in, obj_in), dim=-1)
            interleaved_hand_obj = rearrange(hands_obj, "b t h n -> b (t n) h")
        
        if self.cfg.num_agents > 1 and self.cfg.use_multiagent_pos_emb:
            # repeat [0, 1, ..., num_agents-1] num_timesteps times
            agent_ids = torch.arange(self.cfg.num_agents).repeat(hands.shape[1])
            # repeat over number of batches 
            agent_ids = agent_ids.repeat(hands.shape[0], 1)
            agent_ids = agent_ids.to(hands.device)
            agent_ids_emb = self.agent_id_embedding(agent_ids)

            interleaved_hand_obj = interleaved_hand_obj + agent_ids_emb

        b1 = self.vgpt(inputs_embeds=interleaved_hand_obj, use_cache=False, output_hidden_states=True)
        output_hs = b1['hidden_states'][-1]

        if self.cfg.num_agents == 1:
            obj_out = self.proj_obj_out(output_hs)
            obj_out = obj_out[:, :-1]
            obj_shifted = obj[:, 1:]
        else:
            output_hands = output_hs[:, ::2]
            output_obj = output_hs[:, 1::2]
            hands_out = self.proj_hand_out(output_hands)
            obj_out = self.proj_obj_out(output_obj)
            hands_out = hands_out[:, :-1]
            obj_out = obj_out[:, :-1]
            hands_shifted = hands[:, 1:]
            obj_shifted = obj[:, 1:]
        if self.cfg.pred_rot_only:
            obj_out_q = obj_out
            obj_out_q = self.make_valid_quaternion_differentiable(obj_out_q)
            obj_shifted_q = obj_shifted
        elif self.cfg.pred_tx_only:
            obj_out_t = obj_out
            obj_shifted_t = obj_shifted
        else:
            obj_out_q = obj_out[:, :, :4]
            obj_out_q = self.make_valid_quaternion_differentiable(obj_out_q)
            obj_out_t = obj_out[:, :, 4:]
            obj_shifted_q = obj_shifted[:, :, :4]
            obj_shifted_t = obj_shifted[:, :, 4:]

        if self.cfg.pred_rot_only:
            ang_loss, ang_loss_stdev = self.quaternion_loss(obj_out_q, obj_shifted_q)
            if self.training and self.cfg.check_stdev:
                self.train_rot_loss_stdev(ang_loss_stdev.item())
                self.log("train/loss/ang_loss_stdev", ang_loss_stdev.item(), on_step=False, on_epoch=True, prog_bar=True)
            loss = ang_loss
        elif self.cfg.pred_tx_only:
            tx_loss = F.mse_loss(obj_out_t, obj_shifted_t)
            if self.training and self.cfg.check_stdev:
                tx_loss_stdev = ((obj_out_t - obj_shifted_t)**2).std()
                self.train_tx_loss_stdev(tx_loss_stdev.item())
                self.log("train/loss/tx_loss_stdev", tx_loss_stdev.item(), on_step=False, on_epoch=True, prog_bar=True)
            loss = tx_loss
        else:
            ang_loss, ang_loss_stdev = self.quaternion_loss(obj_out_q, obj_shifted_q)
            tx_loss = F.mse_loss(obj_out_t, obj_shifted_t)
            if self.training and self.cfg.check_stdev:
                tx_loss_stdev = ((obj_out_t - obj_shifted_t)**2).std()
                self.train_rot_loss_stdev(ang_loss_stdev.item())
                self.train_tx_loss_stdev(tx_loss_stdev.item())
                self.log("train/loss/ang_loss_stdev", ang_loss_stdev.item(), on_step=False, on_epoch=True, prog_bar=True)
                self.log("train/loss/tx_loss_stdev", tx_loss_stdev.item(), on_step=False, on_epoch=True, prog_bar=True)
            loss = ang_loss + tx_loss
        
        if self.training:
            if self.cfg.pred_rot_only:
                self.ang_loss(ang_loss.item())
                self.log("train/loss/ang_loss", ang_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            elif self.cfg.pred_tx_only:
                self.tx_loss(tx_loss.item())
                self.log("train/loss/tx_loss", tx_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
            else:
                self.ang_loss(ang_loss.item())
                self.log("train/loss/ang_loss", ang_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
                self.tx_loss(tx_loss.item())
                self.log("train/loss/tx_loss", tx_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
        
        if self.cfg.use_hand_loss and self.cfg.num_agents > 1:
            hands_shifted = hands_shifted.float()
            hand_loss = F.mse_loss(hands_out, hands_shifted)
            loss += hand_loss
            if self.training:
                self.hand_loss(hand_loss.item())
                self.log("train/loss/hand_loss", hand_loss.item(), on_step=False, on_epoch=True, prog_bar=True)
                if self.cfg.check_stdev:
                    hand_loss_stdev = ((hands_out - hands_shifted)**2).std()
                    self.train_hand_loss_stdev(hand_loss_stdev.item())
                    self.log("train/loss/hand_loss_stdev", hand_loss_stdev.item(), on_step=False, on_epoch=True, prog_bar=True)
        
        loss_dict = {"loss" : loss}
        return loss_dict

    def print_weights(self):
        i = 0
        for name, param in self.vgpt.named_parameters():
            if param.requires_grad:
                if i > 0:
                    break
                print(f"Layer: {name} | Size: {param.size()} | Values : {param.data[0, :5]} \n")
                i += 1

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.step(batch)
        loss = sum([v for k,v in loss_dict.items()])

        self.train_loss(loss.item())
        
        for key in loss_dict.keys():
            self.log("train/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True)
        
        self.log_iter_stats(batch_idx)
            
        del loss_dict, batch
        
        return {"loss": loss}

    def on_train_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Training epoch " + str(self.current_epoch) + " ended.")
        
    def on_validation_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): 
            self.trainer.datamodule.data_val.__getitem__(0)

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.step(batch)
        loss = sum([v for k,v in loss_dict.items()])
        
        generated = []

        obj = batch['gt_obj'] # bs, t, 7
        num_timesteps = obj.shape[1]
        obs_timesteps = int(num_timesteps*self.cfg.obs_ratio)
        pred_timesteps = num_timesteps - obs_timesteps

        if self.cfg.pred_rot_only:
            gt_obj = obj[:, obs_timesteps:num_timesteps, :4] # bs, t, 4
            obj_in_ = obj[:, :obs_timesteps, :4] # bs, t, 4
        elif self.cfg.pred_tx_only:
            gt_obj = obj[:, obs_timesteps:num_timesteps, 4:] # bs, t, 3
            obj_in_ = obj[:, :obs_timesteps, 4:] # bs, t, 3
        else:
            gt_obj = obj[:, obs_timesteps:num_timesteps, :] # bs, t, 7
            obj_in_ = obj[:, :obs_timesteps, :] # bs, t, 7
        
        if self.cfg.num_agents == 1:    
            with torch.no_grad():
                for i in range(pred_timesteps):
                    obj_in = self.proj_obj_in(obj_in_.to(torch.float32))
                    b1 = self.vgpt(inputs_embeds=obj_in, use_cache=False, output_hidden_states=True)
                    hidden_state = b1['hidden_states'][-1]
                    out = self.proj_obj_out(hidden_state[:, -1, :])
                    out = out.unsqueeze(1) # bs, 1, 7
                    if self.cfg.pred_rot_only:
                        out = self.make_valid_quaternion_differentiable(out)
                    elif self.cfg.pred_tx_only:
                        pass
                    else:
                        out[:, :, :4] = self.make_valid_quaternion_differentiable(out[:, :, :4])
                    obj_in_ = torch.cat([obj_in_, out], dim=1) # bs, t, 7

                    generated.append(out)

                    del b1, hidden_state, out, obj_in
                    torch.cuda.empty_cache()
                
                generated = torch.cat(generated, dim=1) # bs, t, 7
                generated_obj = generated
        else:
            hands = batch['gt_hand'] # bs, t, 3
            hands_in_ = hands[:, :obs_timesteps+1]
            with torch.no_grad():
                for i in range(pred_timesteps):
                    hands_in_ = hands[:, :obs_timesteps+i+1]
                    hands_in = self.proj_hand_in(hands_in_.to(torch.float32))
                    obj_in = self.proj_obj_in(obj_in_.to(torch.float32))
                    hands_obj = torch.stack((hands_in[:,:-1], obj_in), dim=-1)
                    interleaved_hand_obj = rearrange(hands_obj, "b t h n -> b (t n) h")
                    interleaved_hand_obj = torch.cat((interleaved_hand_obj, hands_in[:,-1].unsqueeze(1)),dim=1)

                    # repeat [0, 1, ..., num_agents-1] num_timesteps times
                    if self.cfg.use_multiagent_pos_emb:
                        agent_ids = torch.arange(self.cfg.num_agents).repeat(hands_in_.shape[1])
                        agent_ids = agent_ids[:-1]
                        # repeat over number of batches 
                        agent_ids = agent_ids.repeat(hands_in_.shape[0], 1)
                        agent_ids = agent_ids.to(hands.device)
                        agent_ids_emb = self.agent_id_embedding(agent_ids)

                        interleaved_hand_obj = interleaved_hand_obj + agent_ids_emb
                    
                    b1 = self.vgpt(inputs_embeds=interleaved_hand_obj, use_cache=False, output_hidden_states=True)
                    hidden_state = b1['hidden_states'][-1]
                    obj_out = self.proj_obj_out(hidden_state[:, -2, :])
                    obj_out = obj_out.unsqueeze(1) # bs, 1, 7
                    if self.cfg.pred_rot_only:
                        obj_out = self.make_valid_quaternion_differentiable(obj_out)
                    elif self.cfg.pred_tx_only:
                        pass
                    else:
                        obj_out[:, :, :4] = self.make_valid_quaternion_differentiable(obj_out[:, :, :4])
                    obj_in_ = torch.cat([obj_in_, obj_out], dim=1) # bs, t, 7

                    generated.append(obj_out)

                    del b1, hidden_state, obj_out, hands_in, obj_in
                    torch.cuda.empty_cache()
                
                generated = torch.cat(generated, dim=1) # bs, t, 7
                generated_obj = generated
        
        if self.cfg.pred_rot_only:
            rel_rot_error = self.eval_rot_error(generated_obj, gt_obj)
            self.val_error_rel_rot(rel_rot_error.item())
            self.log("val/error/rel_rot", rel_rot_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        elif self.cfg.pred_tx_only:
            tx_error = self.eval_tx_error(generated_obj, gt_obj)
            self.val_error_tx(tx_error.item())
            self.log("val/error/tx", tx_error.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:
            rel_rot_error = self.eval_rot_error(generated_obj[...,:4], gt_obj[...,:4])
            tx_error = self.eval_tx_error(generated_obj[...,4:], gt_obj[...,4:])
            self.val_error_rel_rot(rel_rot_error.item())
            self.log("val/error/rel_rot", rel_rot_error, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.val_error_tx(tx_error.item())
            self.log("val/error/tx", tx_error.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.log_iter_stats(batch_idx)

        del loss_dict, batch
            
        return {"loss": loss}

    @rank_zero_only
    def on_validation_epoch_end(self):
        pass
                    
    
    def log_iter_stats(self, cur_iter):
        
        def gpu_mem_usage():
            """Computes the GPU memory usage for the current device (MB)."""
            mem_usage_bytes = torch.cuda.max_memory_allocated()
            return mem_usage_bytes / 1024 / 1024
        
        if(cur_iter%self.cfg.log_frequency != 0):
            return 0
        
        mem_usage = gpu_mem_usage()
        try:
            stats = {
                "epoch": "{}/{}".format(self.current_epoch, self.trainer.max_epochs),
                "iter": "{}/{}".format(cur_iter + 1, self.trainer.num_training_batches),
                "train_loss": "%.4f"%(self.train_loss.compute().item()),
                "val_loss": "%.4f"%(self.val_loss.compute().item()),
                "time": "%.4f"%(self.timer.time_elapsed()-self.timer_last_iter),
                "lr": self.trainer.optimizers[0].param_groups[0]['lr'],
                "mem": int(np.ceil(mem_usage)),
            }
            self.timer_last_iter = self.timer.time_elapsed()
        except:
            for cb_ in self.trainer.callbacks:
                if(cb_.__class__.__name__ == "Timer"):
                    self.timer = cb_
            self.timer_last_iter = self.timer.time_elapsed()
            stats = {}
            
        self.train_loss.reset()
        self.val_loss.reset()
        self.ang_loss.reset()
        self.tx_loss.reset()
        self.train_rot_loss_stdev.reset()
        self.train_tx_loss_stdev.reset()
        self.val_error_rel_rot.reset()
        self.val_error_tx.reset()
        if self.cfg.use_hand_loss and self.cfg.num_agents > 1:
            self.hand_loss.reset()
            self.train_hand_loss_stdev.reset()
        
        log.info(stats)
    
    def get_param_groups(self):
        def _get_layer_decay(name):
            layer_id = None
            if name in ("encoder.class_token", "encoder.pose_token", "encoder.mask_token"):
                layer_id = 0
            elif ("_encoder" in name):
                layer_id = 0
            elif ("_head" in name):
                layer_id = self.cfg.transformer.depth + 1
            elif name.startswith("encoder.pos_embedding"):
                layer_id = 0
            elif name.startswith("encoder.transformer1.layers"):
                layer_id = int(name.split("encoder.transformer1.layers.")[1].split(".")[0]) + 1
            else:
                layer_id = self.cfg.transformer.depth + 1
            layer_decay = self.cfg.solver.layer_decay ** (self.cfg.transformer.depth + 1 - layer_id)
            return layer_id, layer_decay

        non_bn_parameters_count = 0
        zero_parameters_count = 0
        no_grad_parameters_count = 0
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, p in self.named_parameters():
            if not p.requires_grad:
                group_name = "no_grad"
                no_grad_parameters_count += 1
                continue
            name = name[len("module."):] if name.startswith("module.") else name
            if ((len(p.shape) == 1 or name.endswith(".bias")) and self.cfg.solver.ZERO_WD_1D_PARAM):
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "zero")
                weight_decay = 0.0
                zero_parameters_count += 1
            else:
                layer_id, layer_decay = _get_layer_decay(name)
                group_name = "layer_%d_%s" % (layer_id, "non_bn")
                weight_decay = self.cfg.solver.weight_decay
                non_bn_parameters_count += 1

            if group_name not in parameter_group_names:
                parameter_group_names[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.solver.lr * layer_decay,
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": weight_decay,
                    "params": [],
                    "lr": self.cfg.solver.lr * layer_decay,
                }
            parameter_group_names[group_name]["params"].append(name)
            parameter_group_vars[group_name]["params"].append(p)

        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
        optim_params = list(parameter_group_vars.values())
        return optim_params
    
    def configure_optimizers(self):
        # linear learning rate scaling for multi-gpu
        if(self.trainer.num_devices * self.trainer.num_nodes>1 and self.cfg.solver.apply_linear_scaling):
            self.lr_scaler = self.trainer.num_devices * self.trainer.num_nodes * self.trainer.accumulate_grad_batches * self.cfg.train_batch_size / 256
        else:
            self.lr_scaler = 1
        log.info("num_devices: {}, num_nodes: {}, accumulate_grad_batches: {}, train_batch: {}".format(self.trainer.num_devices, self.trainer.num_nodes, self.trainer.accumulate_grad_batches, self.cfg.train_batch_size))
        log.info("Linear LR scaling factor: {}".format(self.lr_scaler))
        
        if(self.cfg.solver.layer_decay is not None):
            optim_params = self.get_param_groups()
        else:
            optim_params = [{'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': self.cfg.solver.lr * self.lr_scaler}]
        
        if(self.cfg.solver.name=="AdamW"):
            optimizer = optim.AdamW(params=optim_params, weight_decay=self.cfg.solver.weight_decay, betas=(0.9, 0.95))
        elif(self.cfg.solver.name=="lion"):
            from lion_pytorch import Lion
            optimizer = Lion(params=optim_params, weight_decay=self.cfg.solver.weight_decay, betas=(0.9, 0.99))
        elif(self.cfg.solver.name=="SGD"):
            optimizer = optim.SGD(params=optim_params, momentum=self.cfg.solver.momentum, weight_decay=self.cfg.solver.weight_decay)
        else:
            raise NotImplementedError("Unknown solver : " + self.cfg.solver.name)

        def warm_start_and_cosine_annealing(epoch):
            if epoch < self.cfg.solver.warmup_epochs:
                lr = (epoch+1) / self.cfg.solver.warmup_epochs
            else:
                lr = 0.5 * (1. + math.cos(math.pi * ((epoch+1) - self.cfg.solver.warmup_epochs) / (self.trainer.max_epochs - self.cfg.solver.warmup_epochs )))
            return lr

        if(self.cfg.solver.scheduler == "cosine"):
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[warm_start_and_cosine_annealing for _ in range(len(optim_params))], verbose=False)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.cfg.solver.decay_steps, gamma=self.cfg.solver.decay_gamma, verbose=False)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval" : "epoch",
                'frequency': 1,
            }
        }
 
if __name__ == "__main__":
    pass