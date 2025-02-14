import math
import os
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig
from torchmetrics import MeanMetric

from PAR.utils import get_pylogger
from transformers import LlamaForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
from einops import rearrange
import torch.nn as nn

from PAR.models.components.trajdata_tokenizer.tokenizer import normalize_trajectory, get_bins, get_tokens, \
                                                        recon_and_unnormalize, normalize_trajectory_relative, rotate_batched_trajectories
from PAR.models.components.trajdata_tokenizer.tokenizer_acceleration import get_bins_first_order, get_second_order_dict, \
                                                        get_tokens_accel, reconstruct_trajectory_accel, recon_and_unnormalize_accel, \
                                                        delta_to_bin
from PAR.models.components.embedders.sin_cos_enc import SinCosPositionalEncoding

from trajdata.visualization.vis import plot_agent_batch, plot_agent_batch_video_frames
from trajdata import AgentBatch
import matplotlib.pyplot as plt

log = get_pylogger(__name__)

def check_deltas(tensor1, tensor2, epsilon):
    combined_tensor = torch.cat((tensor1, tensor2), dim=1)
    deltas = torch.abs(combined_tensor[:, 1:] - combined_tensor[:, :-1])
    return torch.any(deltas > epsilon).item()

class PAR_LitModule(LightningModule):
    def __init__(
        self,
        cfg: DictConfig
    ):

        super().__init__()
        
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False,)
        self.cfg = self.hparams.cfg
        
        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss   = MeanMetric()
        self.val_ade = MeanMetric()
        self.val_fde = MeanMetric()
        self.ade_recon = MeanMetric()
        self.subset_val_ade = MeanMetric()
        self.subset_val_fde = MeanMetric()
        self.subset_ade_recon = MeanMetric()

        if self.cfg.velocity_tokens:
            self.bins = get_bins()
            self.num_bins = 128
        else:
            self.num_bins_first_order = 128 
            self.num_bins = self.cfg.acc_token_size
            self.first_order_bins = get_bins_first_order(self.num_bins_first_order, -18, 18)
            self.second_order_bins_dict = get_second_order_dict(self.num_bins)
        
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
        vgpt_config.vocab_size=(self.num_bins*self.num_bins)+1

        if self.cfg.dropout_hidden > 0:
            vgpt_config.hidden_dropout_prob = self.cfg.dropout_hidden
        if self.cfg.dropout_attn > 0:
            vgpt_config.attention_probs_dropout_prob = self.cfg.dropout_attn
        self.vgpt = LlamaForCausalLM(vgpt_config)

        if self.cfg.num_agents > 1 and self.cfg.use_multiagent_pos_emb:
            self.agent_id_embedding = nn.Embedding(self.cfg.num_agents, hsize)
      
        if self.cfg.location_pos_embedding:
            if self.cfg.cat_loc_emb:
                emb_size = int(hsize/2)
            else:
                emb_size = hsize
            if self.cfg.encoding_type == "sin_cos":
                self.loc_enc_size = self.cfg.loc_enc_size
                self.location_pos_emb = SinCosPositionalEncoding(emb_size, max_position=self.loc_enc_size+1)

        os.makedirs(self.cfg.storage_folder + "/results/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/videos/", exist_ok=True)
        os.makedirs(self.cfg.storage_folder + "/results_vid_frames/", exist_ok=True)
        log.info("Storage folder : " + self.cfg.storage_folder)
                

    def on_train_start(self):
        torch.cuda.empty_cache()
        if(self.cfg.debug): self.trainer.datamodule.data_train.__getitem__(0)
        pass

    def get_loc_emb_training(self, batch):
        pad_value = 0
        expected_timesteps = 2*(self.cfg.hist_sec + self.cfg.fut_sec) + 1
        # get xy locations for all agents, normalize relative to ego agent, flatten
        x_ego = torch.cat((batch.agent_hist.get_attr('x').unsqueeze(1), batch.agent_fut.get_attr('x').unsqueeze(1)), dim=2)
        x_neigh = torch.cat((batch.neigh_hist.get_attr('x'), batch.neigh_fut.get_attr('x')), dim=2)
        if x_neigh.shape[2] < expected_timesteps:
            # pad with nan
            pad_toks = torch.ones(x_neigh.shape[0], x_neigh.shape[1], expected_timesteps - x_neigh.shape[2]) * float('nan')
            pad_toks = pad_toks.to(x_neigh.device)
            x_neigh = torch.cat([x_neigh, pad_toks], dim=2)
        if batch.neigh_hist.shape[1] == 0 and batch.neigh_fut.shape[1] == 0:
            x_neigh = torch.ones((x_ego.shape[0], self.cfg.num_agents-1, expected_timesteps)).to(x_ego.device) * float('nan')
        elif x_neigh.shape[1] < self.cfg.num_agents - 1:
            # pad with nan
            pad_toks = torch.ones(x_neigh.shape[0], self.cfg.num_agents-1 - x_neigh.shape[1], expected_timesteps) * float('nan')
            pad_toks = pad_toks.to(x_neigh.device)
            x_neigh = torch.cat([x_neigh, pad_toks], dim=1)
        if batch.neigh_hist.shape[1] == 0 and batch.neigh_fut.shape[1] == 0:
            x_neigh = torch.ones((x_ego.shape[0], self.cfg.num_agents-1, expected_timesteps)).to(x_ego.device) * float('nan')
        elif x_neigh.shape[1] < self.cfg.num_agents - 1:
            # pad with nan
            pad_toks = torch.ones(x_neigh.shape[0], self.cfg.num_agents-1 - x_neigh.shape[1], expected_timesteps) * float('nan')
            pad_toks = pad_toks.to(x_neigh.device)
            x_neigh = torch.cat([x_neigh, pad_toks], dim=1)
        if self.cfg.relative_pos_locs:
            if self.cfg.moving_ego_relative:
                x_ref = x_ego[:, :, :]
                x_neigh = x_neigh - x_ref
                x_ego = x_ego - x_ref
            else:
                x_ref = x_ego[:, :, 0].unsqueeze(2)
                x_neigh = x_neigh - x_ref
                x_ego = x_ego - x_ref
        
        
        x_loc = torch.cat((x_neigh, x_ego), dim=1) # [B, N, T] with ego agent last 
        x_loc = rearrange(x_loc, 'b n t -> b (t n)')  # flatten to [B, N*T] 
        x_loc = x_loc[:, :-1*self.cfg.num_agents] # remove last timestep since we care about start locations
        if not self.cfg.velocity_tokens:
            x_loc = x_loc[:, :-1*self.cfg.num_agents] # remove one more timestep since we have second order tokens
            
        y_ego = torch.cat((batch.agent_hist.get_attr('y').unsqueeze(1), batch.agent_fut.get_attr('y').unsqueeze(1)), dim=2)
        y_neigh = torch.cat((batch.neigh_hist.get_attr('y'), batch.neigh_fut.get_attr('y')), dim=2)
        if y_neigh.shape[2] < expected_timesteps:
            # pad with nan
            pad_toks = torch.ones(y_neigh.shape[0], y_neigh.shape[1], expected_timesteps - y_neigh.shape[2]) * float('nan')
            pad_toks = pad_toks.to(y_neigh.device)
            y_neigh = torch.cat([y_neigh, pad_toks], dim=2)
        if batch.neigh_hist.shape[1] == 0 and batch.neigh_fut.shape[1] == 0:
            y_neigh = torch.ones((y_ego.shape[0], self.cfg.num_agents-1, expected_timesteps)).to(y_ego.device) * float('nan') 
        elif y_neigh.shape[1] < self.cfg.num_agents - 1:
            # pad with nan
            pad_toks = torch.ones(y_neigh.shape[0], self.cfg.num_agents-1 - y_neigh.shape[1], expected_timesteps) * float('nan')
            pad_toks = pad_toks.to(y_neigh.device)
            y_neigh = torch.cat([y_neigh, pad_toks], dim=1)

        if batch.neigh_hist.shape[1] == 0 and batch.neigh_fut.shape[1] == 0:
            y_neigh = torch.ones((y_ego.shape[0], self.cfg.num_agents-1, expected_timesteps)).to(y_ego.device) * float('nan') 
        elif y_neigh.shape[1] < self.cfg.num_agents - 1:
            # pad with nan
            pad_toks = torch.ones(y_neigh.shape[0], self.cfg.num_agents-1 - y_neigh.shape[1], expected_timesteps) * float('nan')
            pad_toks = pad_toks.to(y_neigh.device)
            y_neigh = torch.cat([y_neigh, pad_toks], dim=1)

        if self.cfg.relative_pos_locs:
            if self.cfg.moving_ego_relative:
                y_ref = y_ego[:, :, :]
                y_neigh = y_neigh - y_ref
                y_ego = y_ego - y_ref
            else:
                y_ref = y_ego[:, :, 0].unsqueeze(2)
                y_neigh = y_neigh - y_ref
                y_ego = y_ego - y_ref
        y_loc = torch.cat((y_neigh, y_ego), dim=1) # [B, N, T] with ego agent last
        y_loc = rearrange(y_loc, 'b n t -> b (t n)')  # flatten to [B, N*T]
        y_loc = y_loc[:, :-1*self.cfg.num_agents] # remove last timestep since we care about start locations
        if not self.cfg.velocity_tokens:
            y_loc = y_loc[:, :-1*self.cfg.num_agents] # remove one more timestep since we have second order tokens
        
        # normalize to 0, 1
        x_loc = (x_loc+100)/200
        y_loc = (y_loc+100)/200

        # replace nan values with something
        x_loc[torch.isnan(x_loc)] = pad_value
        y_loc[torch.isnan(y_loc)] = pad_value

        # scale to 0-self.loc_enc_size and round to int
        x_loc *= self.loc_enc_size
        x_loc = x_loc.round().int()
        x_loc = torch.clamp(x_loc, 0, self.loc_enc_size)
        x_loc_emb = self.location_pos_emb(x_loc)
        y_loc *= self.loc_enc_size 
        y_loc = y_loc.round().int()
        y_loc = torch.clamp(y_loc, 0, self.loc_enc_size)
        y_loc_emb = self.location_pos_emb(y_loc)

        if self.cfg.cat_loc_emb:
            loc_emb = torch.cat([x_loc_emb, y_loc_emb], dim=2)
        else:
            loc_emb = x_loc_emb + y_loc_emb
        return loc_emb

    # assumes that we are using generate_social_translation
    # will always get past tokens
    # also assumes validation batch size of 0    
    def get_loc_emb_generation(self, batch, input_traj):
        pad_value = 0
        B, T = input_traj.shape
        timesteps_per_agent = (T // self.cfg.num_agents) + 1
        # get length of input traj, figure out how many timesteps per agent
        # get neighbors gt up to current timestep (ok to use this info)
        x_neigh = torch.cat((batch.neigh_hist.get_attr('x'), batch.neigh_fut.get_attr('x')), dim=2)[:, :, :timesteps_per_agent]
        y_neigh = torch.cat((batch.neigh_hist.get_attr('y'), batch.neigh_fut.get_attr('y')), dim=2)[:, :, :timesteps_per_agent]

        if batch.neigh_hist.shape[1] == 0 and batch.neigh_fut.shape[1] == 0:
            x_neigh = torch.ones((x_neigh.shape[0], self.cfg.num_agents-1, timesteps_per_agent)).to(x_neigh.device) * float('nan')
        elif x_neigh.shape[1] < self.cfg.num_agents - 1:
            # pad with nan
            pad_toks = torch.ones(x_neigh.shape[0], self.cfg.num_agents-1 - x_neigh.shape[1], x_neigh.shape[2]) * float('nan')
            pad_toks = pad_toks.to(x_neigh.device)
            x_neigh = torch.cat([x_neigh, pad_toks], dim=1)
        if x_neigh.shape[2] < timesteps_per_agent:
            # pad with nan
            pad_toks = torch.ones(x_neigh.shape[0], x_neigh.shape[1], timesteps_per_agent - x_neigh.shape[2]) * float('nan')
            pad_toks = pad_toks.to(x_neigh.device)
            x_neigh = torch.cat([x_neigh, pad_toks], dim=2)
        if batch.neigh_hist.shape[1] == 0 and batch.neigh_fut.shape[1] == 0:
            # import ipdb; ipdb.set_trace()
            y_neigh = torch.ones((y_neigh.shape[0], self.cfg.num_agents-1, timesteps_per_agent)).to(y_neigh.device) * float('nan') 
        elif y_neigh.shape[1] < self.cfg.num_agents - 1:
            # pad with nan
            pad_toks = torch.ones(y_neigh.shape[0], self.cfg.num_agents-1 - y_neigh.shape[1], y_neigh.shape[2]) * float('nan')
            pad_toks = pad_toks.to(y_neigh.device)
            y_neigh = torch.cat([y_neigh, pad_toks], dim=1)
        if y_neigh.shape[2] < timesteps_per_agent:
            pad_toks = torch.ones(y_neigh.shape[0], y_neigh.shape[1], timesteps_per_agent - y_neigh.shape[2]) * float('nan')
            pad_toks = pad_toks.to(y_neigh.device)
            y_neigh = torch.cat([y_neigh, pad_toks], dim=2)

        # first get the ego agent history and integrate if necessary
        # then get neighbors and subtract 
        if self.cfg.relative_pos_locs and self.cfg.moving_ego_relative:
            x_ego_hist = batch.agent_hist.get_attr('x').unsqueeze(1)
            y_ego_hist = batch.agent_hist.get_attr('y').unsqueeze(1)

            if x_ego_hist.shape[2] < timesteps_per_agent:
                # do integration
                x_ego, y_ego = self.integrate_ego_agent_from_tokens(input_traj, x_ego_hist, y_ego_hist)
            else:
                x_ego = x_ego_hist
                y_ego = y_ego_hist

            x_ref = x_ego[:, :, :]
            x_neigh = x_neigh - x_ref
            x_ego = x_ego - x_ref
            y_ref = y_ego[:, :, :]
            y_neigh = y_neigh - y_ref
            y_ego = y_ego - y_ref
        
        # normalize just the first frame relative to ego agent starting point, then get gt for all neighbors
        # need to integrate just the ego agent
        else:
            # get ego agent history gt location 
            x_ego_hist = batch.agent_hist.get_attr('x').unsqueeze(1)
            y_ego_hist = batch.agent_hist.get_attr('y').unsqueeze(1)

            if self.cfg.relative_pos_locs:
                x_ref = x_ego_hist[:, :, 0].unsqueeze(2)
                y_ref = y_ego_hist[:, :, 0].unsqueeze(2)
                x_neigh =  x_neigh - x_ref
                y_neigh =  y_neigh - y_ref
                x_ego_hist = x_ego_hist - x_ref
                y_ego_hist = y_ego_hist - y_ref
            
            if x_ego_hist.shape[2] < timesteps_per_agent:
                x_ego, y_ego = self.integrate_ego_agent_from_tokens(input_traj, x_ego_hist, y_ego_hist)
            else:
                x_ego = x_ego_hist 
                y_ego = y_ego_hist

        x_loc = torch.cat((x_neigh, x_ego), dim=1) # [B, N, T] with ego agent last 
        x_loc = rearrange(x_loc, 'b n t -> b (t n)')
        x_loc = x_loc[:, :T] 

        y_loc = torch.cat((y_neigh, y_ego), dim=1) # [B, N, T] with ego agent last
        y_loc = rearrange(y_loc, 'b n t -> b (t n)')
        y_loc = y_loc[:, :T]
    
        x_loc = (x_loc+100)/200
        y_loc = (y_loc+100)/200

        # replace nan values with padding
        x_loc[torch.isnan(x_loc)] = pad_value
        y_loc[torch.isnan(y_loc)] = pad_value
        
        # scale to 0-self.loc_enc_size and round to int
        x_loc *= self.loc_enc_size 
        x_loc = x_loc.round().int()
        x_loc = torch.clamp(x_loc, 0, self.loc_enc_size)
        x_loc_emb = self.location_pos_emb(x_loc)
        y_loc *= self.loc_enc_size 
        y_loc = y_loc.round().int()
        y_loc = torch.clamp(y_loc, 0, self.loc_enc_size)
        y_loc_emb = self.location_pos_emb(y_loc)

        if self.cfg.cat_loc_emb:
            loc_emb = torch.cat([x_loc_emb, y_loc_emb], dim=2)
        else:
            loc_emb = x_loc_emb + y_loc_emb
        return loc_emb

    def integrate_ego_agent_from_tokens(self, input_traj, x_ego_hist, y_ego_hist):
        if self.cfg.velocity_tokens:
                    # get ego agent tokens 
            ego_tok = input_traj[:, self.cfg.num_agents-1::self.cfg.num_agents]
            history_timesteps = self.cfg.hist_sec*2 # 2 hz prediction
                    # we want to start integrating after the history timesteps
            ego_tok_to_integrate = ego_tok[0, history_timesteps:] 
            x_ego_integrated = []
            y_ego_integrated = []
            curr_x = x_ego_hist[:, :, -1]
            curr_y = y_ego_hist[:, :, -1]
            for tok in ego_tok_to_integrate:
                tok = tok.item()
                x_idx, y_idx = divmod(tok, self.num_bins)
                if x_idx >= self.num_bins:
                    delta_x = self.num_bins -1
                else: 
                    delta_x = (torch.tensor(self.bins[x_idx]) + torch.tensor(self.bins[x_idx + 1])) / 2
                if y_idx >= self.num_bins:
                    delta_y = self.num_bins -1
                else:
                    delta_y = (torch.tensor(self.bins[y_idx]) + torch.tensor(self.bins[y_idx + 1])) / 2
                curr_x = curr_x + delta_x
                curr_y = curr_y + delta_y
                x_ego_integrated.append(curr_x)
                y_ego_integrated.append(curr_y)
            x_ego_integrated = torch.stack(x_ego_integrated, dim=2)
            y_ego_integrated = torch.stack(y_ego_integrated, dim=2)
            x_ego = torch.cat((x_ego_hist, x_ego_integrated), dim=2)
            y_ego = torch.cat((y_ego_hist, y_ego_integrated), dim=2)

        else:
            reverse_second_order_bins_dict = {v: k for k, v in self.second_order_bins_dict.items()}
                    # get ego agent tokens 
            ego_tok = input_traj[:, self.cfg.num_agents-1::self.cfg.num_agents]
            history_timesteps = self.cfg.hist_sec*2 # 2 hz prediction
                    # we want to start integrating after the history timesteps
            ego_tok_to_integrate = ego_tok[0, history_timesteps:] 
            x_ego_integrated = []
            y_ego_integrated = []
                    # start with last history timestep position
            curr_x = x_ego_hist[:, :, -1]
            curr_y = y_ego_hist[:, :, -1]
                    # and velocity
            velocity_x = x_ego_hist[:, :, -1] - x_ego_hist[:, :, -2]
            velocity_y = y_ego_hist[:, :, -1] - y_ego_hist[:, :, -2]

            velocity_x_token = delta_to_bin(velocity_x, self.first_order_bins)
            velocity_y_token = delta_to_bin(velocity_y, self.first_order_bins)

            for tok in ego_tok_to_integrate:
                tok = tok.item()
                x_idx, y_idx = divmod(tok, self.num_bins)
                
                accel_x = reverse_second_order_bins_dict.get(x_idx, 0)
                accel_y = reverse_second_order_bins_dict.get(y_idx, 0)

                velocity_x_token = velocity_x_token + accel_x
                velocity_y_token = velocity_y_token + accel_y

                if velocity_x_token >= len(self.first_order_bins) - 1:
                    velocity_x_token = len(self.first_order_bins) - 2
                if velocity_y_token >= len(self.first_order_bins) - 1:
                    velocity_y_token = len(self.first_order_bins) - 2

                delta_x = (self.first_order_bins[velocity_x_token] + self.first_order_bins[velocity_x_token + 1]) / 2
                delta_x = torch.tensor(delta_x).to(curr_x.device)
                delta_y = (self.first_order_bins[velocity_y_token] + self.first_order_bins[velocity_y_token + 1]) / 2
                delta_y = torch.tensor(delta_y).to(curr_y.device)
                        
                curr_x = curr_x + delta_x
                curr_y = curr_y + delta_y
                x_ego_integrated.append(curr_x)
                y_ego_integrated.append(curr_y)
            x_ego_integrated = torch.stack(x_ego_integrated, dim=2)
            y_ego_integrated = torch.stack(y_ego_integrated, dim=2)
            x_ego = torch.cat((x_ego_hist, x_ego_integrated), dim=2)
            y_ego = torch.cat((y_ego_hist, y_ego_integrated), dim=2)
        
        return x_ego, y_ego
       
    def get_input_embed(self, batch, return_loc_emb=False, for_generation=False):
        pad_index = self.num_bins * self.num_bins  
        
        batch: AgentBatch
        # ego agent
        x_traj, y_traj, norm_x, norm_y, init_x, init_y, init_h = normalize_trajectory(batch.agent_hist, batch.agent_fut)

        if self.cfg.velocity_tokens:
            tokens = get_tokens(norm_x, norm_y, self.bins)
        else:
            tokens = get_tokens_accel(norm_x, norm_y, self.first_order_bins, self.second_order_bins_dict, self.num_bins)

        if self.cfg.num_agents > 1: 
            all_ag_tok = []
            for ag in range(0, self.cfg.num_agents-1):
                try:
                    if self.cfg.relative_pos_toks: 
                        # normalize each agent to ego agent's frame of reference
                        norm_x, norm_y, _, _, _ = normalize_trajectory_relative(batch.neigh_hist[:, ag], batch.neigh_fut[:, ag], init_x, init_y, init_h)
                    else:
                        # normalize each agent to (0,0) and 0 rad heading
                        _, _, norm_x, norm_y, _, _, _ = normalize_trajectory(batch.neigh_hist[:, ag], batch.neigh_fut[:, ag])

                    if self.cfg.velocity_tokens:
                        agent_tokens = get_tokens(norm_x, norm_y, self.bins)
                    else:
                        agent_tokens = get_tokens_accel(norm_x, norm_y, self.first_order_bins, self.second_order_bins_dict)
                    
                    # this is the case where neighbors leave before the ego agent across all batches and neighbors
                    # we need to pad the tokens 
                    if batch.neigh_fut[:, ag].shape[1] < batch.agent_fut.shape[1]:
                        num_extra_tok = batch.agent_fut.shape[1] - batch.neigh_fut[:, ag].shape[1]
                        pad_toks = (torch.ones(num_extra_tok) * pad_index).unsqueeze(0).to(agent_tokens.device)
                        agent_tokens = torch.cat([agent_tokens, pad_toks], dim=1)
                    if batch.neigh_hist[:, ag].shape[1] < batch.agent_hist.shape[1]:
                        num_extra_tok = batch.agent_hist.shape[1] - batch.neigh_hist[:, ag].shape[1]
                        pad_toks = (torch.ones(num_extra_tok) * pad_index).unsqueeze(0).to(agent_tokens.device)
                        agent_tokens = torch.cat([pad_toks, agent_tokens], dim=1)
                except:
                    agent_tokens = (torch.ones(tokens.shape) * pad_index).to(tokens.device)
                
                all_ag_tok.append(agent_tokens)

            try:
                # ego agent is last for each timestep
                all_ag_tok.append(tokens)
                tokens_stacked = torch.stack(all_ag_tok, dim=2) #[B, T, N]
                # interleave tokens for each agent over timestep 
                # go from [B, T, N] -> [B, T*N]
                tokens = tokens_stacked.view(tokens_stacked.shape[0], -1)
                tokens_stacked = tokens_stacked.long()
            except:
                import ipdb; ipdb.set_trace()
        else:
            tokens_stacked = None
        tokens = tokens.long() 
        t_emb = self.vgpt.get_input_embeddings()(tokens)
        
        return tokens, t_emb, tokens_stacked


    def get_agent_id_emb(self, t_emb):
        agent_ids = torch.arange(self.cfg.num_agents).repeat((t_emb.shape[1]))
        agent_ids = agent_ids[:t_emb.shape[1]]
        # repeat over number of batches 
        agent_ids = agent_ids.repeat(t_emb.shape[0], 1)
        agent_ids = agent_ids.to(t_emb.device)
        agent_ids_emb = self.agent_id_embedding(agent_ids)
        return agent_ids_emb

    def step(self, batch: Any):
        tokens, t_emb, _ = self.get_input_embed(batch)

        if self.cfg.num_agents > 1 and self.cfg.use_multiagent_pos_emb:
            agent_ids_emb = self.get_agent_id_emb(t_emb)
            t_emb = t_emb + agent_ids_emb

        if self.cfg.location_pos_embedding:
            loc_emb = self.get_loc_emb_training(batch)
            t_emb = loc_emb + t_emb
        
        model_out = self.vgpt(inputs_embeds=t_emb, use_cache=False, output_hidden_states=True)
        
        if self.cfg.loss_on_same_agent:    
            # shift by num agents, so that the prediction is on the same agent
            outputs = model_out['logits'].permute(0,2,1)[:, :, :-1*self.cfg.num_agents]
            input_tokens_shifted = tokens[:, self.cfg.num_agents:]
        else:
            # normal next token prediction
            outputs = model_out['logits'].permute(0,2,1)[:, :, :-1]
            input_tokens_shifted = tokens[:, 1:]

        loss = F.cross_entropy(outputs, input_tokens_shifted, ignore_index=169)

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

        tokens, _, tokens_stacked = self.get_input_embed(batch)

        generated = []

        num_timesteps = tokens.shape[1] // self.cfg.num_agents
        
        obs_timesteps = self.cfg.hist_sec*2 # 2 Hz prediction
        pred_timesteps = num_timesteps - obs_timesteps

        input_traj = tokens[:, :obs_timesteps*self.cfg.num_agents]
        
        if self.cfg.num_agents > 1:   
            with torch.no_grad():
                for i in range(pred_timesteps):
                    other_ag_tok = tokens_stacked[:, i + obs_timesteps, :-1]
                    input_traj = torch.cat((input_traj, other_ag_tok), dim=1)
                    t_emb = self.vgpt.get_input_embeddings()(input_traj)
                    if self.cfg.location_pos_embedding:
                        loc_emb = self.get_loc_emb_generation(batch, input_traj)
                        t_emb = t_emb + loc_emb
                    if self.cfg.use_multiagent_pos_emb:
                        agent_ids_emb = self.get_agent_id_emb(t_emb)
                        t_emb = t_emb + agent_ids_emb
                    model_out = self.vgpt(inputs_embeds=t_emb, use_cache=False, output_hidden_states=True)
                    if self.cfg.loss_on_same_agent:
                        # last num_agents tokens will correspond to the ego agent, then A1, A2, A3, ...
                        # we just want the ego agent's predicted action, discard the rest
                        # if input is A1, B1, A2, -> output is A2_pred, B2_pred, A3_pred
                        # we only want B2_pred
                        logits = model_out['logits'].permute(0,2,1)[:, :, -1*self.cfg.num_agents]
                    else:
                        logits = model_out['logits'].permute(0,2,1)[:, :, -1]
                    if self.cfg.multinomial_sampling:
                        probs = torch.softmax(logits, dim=-1)
                        out_tok = torch.multinomial(probs, num_samples=1)
                    else:
                        out_tok = torch.argmax(logits, dim=-1).unsqueeze(1)

                    input_traj = torch.cat([input_traj, out_tok], dim=1)
                    generated.append(out_tok)

                    del t_emb, model_out, logits, out_tok
                    torch.cuda.empty_cache()

                generated_tokens = torch.cat(generated, dim=1)
        else:
            with torch.no_grad():
                for i in range(pred_timesteps*self.cfg.num_agents):
                    t_emb = self.vgpt.get_input_embeddings()(input_traj)
                    if self.cfg.location_pos_embedding:
                        raise NotImplementedError("Location pos embedding not implemented for single-agent")
                    model_out = self.vgpt(inputs_embeds=t_emb, use_cache=False, output_hidden_states=True)
                    logits = model_out['logits'].permute(0,2,1)[:, :, -1]
                    if self.cfg.multinomial_sampling:
                        probs = torch.softmax(logits, dim=-1)
                        out_tok = torch.multinomial(probs, num_samples=1)
                    else:
                        out_tok = torch.argmax(logits, dim=-1).unsqueeze(1)

                    input_traj = torch.cat([input_traj, out_tok], dim=1)
                    generated.append(out_tok)

                    del t_emb, model_out, logits, out_tok
                    torch.cuda.empty_cache()

                generated_tokens = torch.cat(generated, dim=1)
        
        # ego agent only GT
        x_traj, y_traj, norm_x, norm_y, init_x, init_y, init_h = normalize_trajectory(batch.agent_hist, batch.agent_fut)

        if self.cfg.num_agents > 1:
            tokens = tokens_stacked[:, :, -1]
            
        # generated_tokens just corresponds to ego agent in both of these cases
        gen_with_prefix = torch.cat([tokens[:, :obs_timesteps], generated_tokens], dim=1)
        if self.cfg.velocity_tokens:
            gt_x_recon, gt_y_recon = recon_and_unnormalize(tokens, self.bins, len(self.bins)-1, init_x, init_y, init_h)
            gen_x_recon, gen_y_recon = recon_and_unnormalize(gen_with_prefix, self.bins, len(self.bins)-1, init_x, init_y, init_h)
        else:
            initial_velocity_x = norm_x[:, 1] - norm_x[:, 0]
            initial_velocity_y = norm_y[:, 1] - norm_y[:, 0]

            gt_x_recon, gt_y_recon = recon_and_unnormalize_accel(init_x, init_y, init_h, initial_velocity_x, initial_velocity_y, tokens, self.first_order_bins, self.num_bins, self.second_order_bins_dict)
            gen_x_recon, gen_y_recon = recon_and_unnormalize_accel(init_x, init_y, init_h, initial_velocity_x, initial_velocity_y, gen_with_prefix, self.first_order_bins, self.num_bins, self.second_order_bins_dict)

        gt_x_fut = x_traj[:, obs_timesteps:]
        gt_y_fut = y_traj[:, obs_timesteps:]
        gen_x_recon_pred_fut = gen_x_recon[:, obs_timesteps:]
        gen_y_recon_pred_fut = gen_y_recon[:, obs_timesteps:]

        # compute ADE on non-nan vals
        diff = torch.sqrt((gt_x_fut - gen_x_recon_pred_fut)**2 + (gt_y_fut - gen_y_recon_pred_fut)**2)
        mask = ~torch.isnan(diff)
        ade = torch.mean(diff[mask])
        fde = torch.mean(torch.sqrt((gt_x_fut[:,-1] - gen_x_recon_pred_fut[:,-1])**2 + (gt_y_fut[:,-1] - gen_y_recon_pred_fut[:,-1])**2))
        
        recon_ade = torch.mean(torch.sqrt((gt_x_recon - x_traj)**2 + (gt_y_recon - y_traj)**2))

        # if this is an example where the ego car moves, add to subset metrics
        if check_deltas(batch.agent_hist.get_attr("x"), batch.agent_fut.get_attr("x"), 0.1) and check_deltas(batch.agent_hist.get_attr("y"), batch.agent_fut.get_attr("y"), 0.1):
            self.subset_val_ade(ade.item())
            self.subset_val_fde(fde.item())
            self.subset_ade_recon(recon_ade.item())
        
        # update and log metrics
        self.val_loss(loss.item())
        self.val_ade(ade.item())
        self.val_fde(fde.item())
        self.ade_recon(recon_ade.item())
        for key in loss_dict.keys():
            self.log("val/loss/" + key, loss_dict[key].item(), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.log_iter_stats(batch_idx)
        
        if self.cfg.do_vis and batch_idx%5 == 0:
            if self.cfg.num_agents > 1 and not self.cfg.generate_social_translation:
                raise NotImplementedError("Visualization not implemented for multi-agent all agents predictions")

            pred_future = torch.stack((gen_x_recon_pred_fut, gen_y_recon_pred_fut), dim=-1)
            pred_future = pred_future.cpu().numpy()
            vis_dir = self.cfg.storage_folder + "/results/"
            plot_agent_batch(batch, batch_idx=0, data_num=i, prediction=pred_future, save_path=f"{vis_dir}/epoch_{self.current_epoch}_batch_{batch_idx}.png")

            if check_deltas(batch.agent_hist.get_attr("x"), batch.agent_fut.get_attr("x"), 0.1) and check_deltas(batch.agent_hist.get_attr("y"), batch.agent_fut.get_attr("y"), 0.1):
                import cv2
                import numpy as np
                from matplotlib.backends.backend_agg import FigureCanvasAgg

                # video 
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                out = cv2.VideoWriter(f'{vis_dir}/video_batch_{batch_idx}.mp4', fourcc, 30.0, (640, 480)) 

                for f in range(17):
                    fig, ax = plt.subplots(figsize=(6.4, 4.8))  # Create a new figure with the desired size
                    try:
                        plot_agent_batch_video_frames(batch, batch_idx=0, data_num=i, prediction=pred_future, save_path=f"{vis_dir}/frame_{f}_batch_{batch_idx}.png", frame_num=f, ax=ax, show=False)
                    except:
                        continue
                    
                    # Convert matplotlib figure to numpy array
                    canvas = FigureCanvasAgg(fig)
                    canvas.draw()
                    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
                    
                    # Convert RGB to BGR (OpenCV uses BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    
                    # Write the frame
                    out.write(img)
                    
                    plt.close(fig)  # Close the figure to free up memory
                out.release()

        del loss_dict, batch
            
        return {"loss": loss}

    @rank_zero_only
    def on_validation_epoch_end(self):
        final_ade = self.val_ade.compute().item()
        self.log("val/ade", final_ade, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        final_fde = self.val_fde.compute().item()
        self.log("val/fde", final_fde, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        print("Final ADE: ", final_ade)
        print("Final FDE: ", final_fde)
        
        final_ade_recon = self.ade_recon.compute().item()
        self.log("val/ade_recon", final_ade_recon, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        print(self.subset_val_ade)
        final_subset_ade = self.subset_val_ade.compute().item()
        print("Final Subset ADE: ", final_subset_ade)
        self.log("val/subset_ade", final_subset_ade, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        final_subset_fde = self.subset_val_fde.compute().item()
        self.log("val/subset_fde", final_subset_fde, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        final_subset_ade_recon = self.subset_ade_recon.compute().item()
        self.log("val/subset_ade_recon", final_subset_ade_recon, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)


        # reset metric 
        self.val_loss.reset()
        self.val_ade.reset()
        self.val_fde.reset()
        self.subset_val_ade.reset()
        self.subset_val_fde.reset()
        self.subset_ade_recon.reset()
        self.train_loss.reset() 
        self.ade_recon.reset()
    
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

        # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
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