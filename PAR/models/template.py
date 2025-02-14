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

from PAR.utils import get_pylogger
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
import torch.nn as nn

log = get_pylogger(__name__)

class LART_LitModule(LightningModule):
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
        # get input tokens, pass through model, and compute loss
        loss = None
        loss_dict                     = {"loss" : loss}
        
        return loss_dict, None, None

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
        loss_dict, _, _ = self.step(batch)
        loss = sum([v for k,v in loss_dict.items()]) 

        # write for loop to generate tokens
        
        # compute error on generated tokens
        error = None

        self.log("val/error", error.item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
        # update and log metrics
        self.val_loss(loss.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        

        self.log_iter_stats(batch_idx)

         # write visualization code here
        if self.cfg.do_vis:
            
            pass 
            # current epoch: self.current_epoch
            # current batch: batch_idx

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