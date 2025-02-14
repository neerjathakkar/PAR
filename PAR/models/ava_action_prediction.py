import copy
import math
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torchmetrics import MeanMetric
from PAR.evaluators.ava_map import compute_mAP

from PAR.utils import get_pylogger
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
import torch.nn as nn
import cv2
import re
from iopath.common.file_io import g_pathmgr

log = get_pylogger(__name__)

def clean_label(text):
    # Remove text within parentheses (including the parentheses)
    return re.sub(r'\s*\([^)]*\)', '', text)


def read_label_map(label_map_file: str) -> Tuple:
        """
        Read label map and class ids.
        Args:
            label_map_file (str): Path to a .pbtxt containing class id's
                and class names
        Returns:
            (tuple): A tuple of the following,
                label_map (dict): A dictionary mapping class id to
                    the associated class names.
                class_ids (set): A set of integer unique class id's
        """
        label_map = {}
        class_ids = set()
        name = ""
        class_id = ""
        with g_pathmgr.open(label_map_file, "r") as f:
            for line in f:
                if line.startswith("  name:"):
                    name = line.split('"')[1]
                elif line.startswith("  id:") or line.startswith("  label_id:"):
                    class_id = int(line.strip().split(" ")[-1])
                    label_map[class_id] = name
                    class_ids.add(class_id)
        return label_map, class_ids

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
        
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.val_error = MeanMetric()
        
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

        self.proj_act_in = nn.Linear(80, hsize)
        self.proj_act_out = nn.Linear(hsize, 80)

        # make csv file
        self.ava_csv_path = os.path.join(self.cfg.storage_folder, "tmp_results.csv")
        self.ava_per_class_mAP_path = os.path.join(self.cfg.storage_folder, "per_class_mAP.csv")

        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)
                

    def on_train_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): self.trainer.datamodule.data_train.__getitem__(0)
        pass

    def step(self, batch: Any):
        actions = batch['agent_labels'] # bs, t, num_agents, 80 
        
        actions_ = rearrange(actions, "b t n d -> b (t n) d")
        actions_in = self.proj_act_in(actions_.to(torch.float32))

        if self.cfg.num_agents > 1 and self.cfg.use_multiagent_pos_emb:
            # repeat [0, 1, ..., num_agents-1] num_timesteps times
            agent_ids = torch.arange(self.cfg.num_agents).repeat(actions.shape[1])
            # repeat over number of batches 
            agent_ids = agent_ids.repeat(actions.shape[0], 1)
            agent_ids = agent_ids.to(actions.device)
            agent_ids_emb = self.agent_id_embedding(agent_ids)

            actions_in = actions_in + agent_ids_emb
        
        b1 = self.vgpt(inputs_embeds=actions_in, use_cache=False, output_hidden_states=True)
        outputs = self.proj_act_out(b1['hidden_states'][-1]) 
        
        if self.cfg.loss_on_same_agent:
            # shift by num agents, so that the prediction is on the same agent
            outputs = outputs[:, :-1*self.cfg.num_agents]
            actions_shifted = actions_[:, self.cfg.num_agents:]
        else:
            # normal next token prediction
            outputs = outputs[:, :-1]
            actions_shifted = actions_[:, 1:]
    
        loss = torch.mean((outputs - actions_shifted)**2)

        loss_dict                     = {"loss" : loss}
        
        return loss_dict, None, None

    def training_step(self, batch: Any, batch_idx: int):

        loss_dict, output, smpl_output = self.step(batch)
        loss = sum([v for k,v in loss_dict.items()])

        self.train_loss(loss.item())
        
        for key in loss_dict.keys():
            self.log("train/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True)
        
        self.log_iter_stats(batch_idx)
            
        del loss_dict, output, smpl_output, batch
        
        return {"loss": loss}

    def on_train_epoch_end(self):
        log.info("\n " + self.cfg.storage_folder +  " : Training epoch " + str(self.current_epoch) + " ended.")
        
    def on_validation_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): 
            self.trainer.datamodule.data_val.__getitem__(0)

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict, _, _ = self.step(batch)
        loss = sum([v for k,v in loss_dict.items()]) 
        
        generated = []
        actions = batch['agent_labels'] # bs, t, num_agents, 80 
        num_timesteps = actions.shape[1]
        # detect at what timestep the padding starts for the ego agent (assumes val bs=1)
        for t in range(num_timesteps):
            if torch.sum(actions[:, t, -1, :]) == 0:
                num_timesteps = t
                break
        obs_timesteps = int(num_timesteps*0.5)
        pred_timesteps = num_timesteps - obs_timesteps

        gt_actions = actions[:, obs_timesteps:num_timesteps, :, :] 
        actions_in_ = actions[:, :obs_timesteps, :, :]
        actions_in_ = rearrange(actions_in_, "b t n d -> b (t n) d")
        
        if self.cfg.num_agents > 1:    
            with torch.no_grad():
                gt_actions_agent = gt_actions[:, :, -1, :].unsqueeze(2)
                for i in range(pred_timesteps):
                    actions_prev_agents_ = actions[:, obs_timesteps+i, :-1, :] # bs, num_people-1, 80
                    actions_in_ = torch.cat([actions_in_, actions_prev_agents_], dim=1)
                    actions_in = self.proj_act_in(actions_in_.to(torch.float32))
                    if self.cfg.use_multiagent_pos_emb:
                        # repeat [0, 1, ..., num_agents-1] 
                        agent_ids = torch.arange(self.cfg.num_agents).repeat(actions_in.shape[1])
                        agent_ids = agent_ids[:actions_in.shape[1]]
                        # repeat over number of batches 
                        agent_ids = agent_ids.repeat(actions_in.shape[0], 1)
                        agent_ids = agent_ids.to(actions_in.device)
                        agent_ids_emb = self.agent_id_embedding(agent_ids)

                        actions_in = actions_in + agent_ids_emb 
                    b1 = self.vgpt(inputs_embeds=actions_in, use_cache=False, output_hidden_states=True)
                    hidden_state = b1['hidden_states'][-1]
                    if self.cfg.loss_on_same_agent:
                        # last num_agents tokens will correspond to the ego agent, then A1, A2, A3, ...
                        # we just want the ego agent's predicted action, discard the rest
                        # if input is A1, B1, A2, -> output is A2_pred, B2_pred, A3_pred
                        # we only want B2_pred
                        hidden_state_ego_agent = hidden_state[:, -1*self.cfg.num_agents, :]
                        out = self.proj_act_out(hidden_state_ego_agent)
                    else:
                        out = self.proj_act_out(hidden_state[:, -1, :])
                    out = out.unsqueeze(1)
                    actions_in_= torch.cat([actions_in_, out], dim=1)

                    generated.append(out)


                    del b1, hidden_state, out, actions_in
                    torch.cuda.empty_cache()

                generated = torch.cat(generated, dim=1)
                generated_actions_ = rearrange(generated, "b (t n) d -> b t n d ", n=1)

                error = torch.mean((generated_actions_ - gt_actions_agent)**2)
        else:
            with torch.no_grad():
                for i in range(pred_timesteps*self.cfg.num_agents):
                    actions_in = self.proj_act_in(actions_in_.to(torch.float32))
                    b1 = self.vgpt(inputs_embeds=actions_in, use_cache=False, output_hidden_states=True)
                    hidden_state = b1['hidden_states'][-1]
                    out = self.proj_act_out(hidden_state[:, -1, :])
                    out = out.unsqueeze(1)
                    actions_in_= torch.cat([actions_in_, out], dim=1)
                    generated.append(out)

                    del b1, hidden_state, out, actions_in
                    torch.cuda.empty_cache()

                generated = torch.cat(generated, dim=1)
                generated_actions_ = rearrange(generated, "b (t n) d -> b t n d ", n=self.cfg.num_agents)
                
                gt_actions_agent = gt_actions[:, :, -1, :].unsqueeze(2)
                generated_actions_ = generated_actions_[:, :, -1, :].unsqueeze(2)
                error = torch.mean((generated_actions_ - gt_actions_agent)**2)
                
            self.val_error(error.item())

        # write results to CSV for mAP computation
        gt = gt_actions_agent
       
        num_classes = 80
        gt = gt.cpu().numpy().reshape(-1, num_classes)
        predictions = generated_actions_.cpu().numpy().reshape(-1, num_classes)
        with open(self.ava_csv_path, "a") as f:
            gt_pred = np.concatenate([gt, predictions], axis=1)
            np.savetxt(f, gt_pred, delimiter=",")

        self.log("val/error", error.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

        self.log_iter_stats(batch_idx)

        if self.cfg.do_vis:
            AVA_FPS = 30
            label_map, class_ids = read_label_map("/datasets/ava_2024-01-05_0047/annotations/ava_action_list_v2.2_for_activitynet_2019.pbtxt")
            print(label_map)
            video_name = batch['video_name'][0] 
            frames_path = os.path.join("/datasets/ava_2024-01-05_0047/frames/", video_name)

            first_f = 1
            first_frame = f"{video_name}_{first_f:06d}.jpg"
            frame = cv2.imread(os.path.join(frames_path, first_frame))
            width, height = frame.shape[1], frame.shape[0]

            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(self.cfg.storage_folder + f'/results/results_epoch_{self.current_epoch}_{video_name}_batch_{batch_idx}.mp4', fourcc, 30.0, (width, height)) 

            start_timestep = batch["start_timestep"][0].item()
            bboxes = batch["agent_bboxes"][0]
            for t in range(num_timesteps):
                video_timestep = start_timestep + t

                frames_start = int(video_timestep * AVA_FPS - 15)
                frames_end = int((video_timestep + 1) * AVA_FPS - 16)
                frames = [video_timestep for video_timestep in range(frames_start, frames_end)]

                for f in frames:
                    frame = f"{video_name}_{f:06d}.jpg"
                    frame_path = os.path.join(frames_path, frame)
                    if not os.path.exists(frame_path):
                        print("No frame for frame {}".format(f))
                        print(frame_path)
                        continue
                    frame = cv2.imread(frame_path)
                    frame = cv2.resize(frame, (width, height))

                    # visualize gt actions
                    for ag in range(self.cfg.num_agents):
                        agent_bbox = bboxes[t, ag, :]
                        # if all zeros
                        if not torch.any(agent_bbox):
                            continue
                        agent_actions = actions[0, t, ag, :]
                        # get nonzero indices
                        nonzero_indices = torch.nonzero(agent_actions).squeeze(1)
                        nonzero_indices += 1

                        x1, y1, x2, y2 = agent_bbox
                        x1, y1, x2, y2 = int(x1.item() * width), int(y1.item() * height), int(x2.item() * width), int(y2.item() * height)
                        if ag == self.cfg.num_agents - 1:
                            color = (0, 255, 0)
                        else: 
                            color = (255, 0, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        text_y = y1 + 20
                        for action_idx in nonzero_indices:
                            if action_idx.item() in label_map:
                                action = label_map[action_idx.item()]
                            else:
                                action = "Unknown"
                            action_text = f"{clean_label(str(action))}: 100%"
                            # black text with white outline
                            cv2.putText(frame, action_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 4, cv2.LINE_AA)
                            cv2.putText(frame, action_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
                            text_y += 20 
                        
                        if ag == self.cfg.num_agents - 1 and t >= obs_timesteps:
                            # visualize predictions as well
                            gen_t = t - obs_timesteps
                            agent_actions_pred = generated_actions_[0, gen_t, :]
                            probs_pred, top_indices_pred = agent_actions_pred.topk(3, dim=1)
                            top_idx_pred = top_indices_pred[0].tolist()
                            top_idx_pred = [x + 1 for x in top_idx_pred]

                            for action_idx, prob in zip(top_idx_pred, probs_pred[0].tolist()):
                                if action_idx in label_map:
                                    action = label_map[action_idx]
                                else:
                                    action = "Unknown"
                                action_text = f"{clean_label(str(action))}: {prob*100:.1f}%"
                                cv2.putText(frame, action_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 0), 4, cv2.LINE_AA)
                                cv2.putText(frame, action_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 0, 255), 2, cv2.LINE_AA)
                                text_y += 20 


                    out.write(frame)
            out.release()

        del loss_dict, batch
            
        return {"loss": loss}

    @rank_zero_only
    def on_validation_epoch_end(self):
        mAP, per_class_APs, mAP_person, mAP_non_person = compute_mAP(self.ava_csv_path)

        # if self.current_epoch  % 10 == 0:
        with open(self.ava_per_class_mAP_path, "a") as f:
            # write per class APs as one line 
            f.write(str(self.current_epoch) + ",")
            f.write(",".join([str(ap) for ap in per_class_APs]) + "\n")

        # clear file after epoch
        with open(self.ava_csv_path, "w") as f:
            pass
        self.log("val/mAP", mAP, prog_bar=True)
        self.log("val/mAP_person", mAP_person, prog_bar=True)
        self.log("val/mAP_non_person", mAP_non_person, prog_bar=True)
        log.info("mAP : " + str(mAP))
        log.info("mAP_person : " + str(mAP_person))
        log.info("mAP_non_person : " + str(mAP_non_person))

    
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

        optim_params = list(parameter_group_vars.values())
        return optim_params
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

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