import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

from trajdata import AgentBatch, UnifiedDataset, AgentType
from trajdata.visualization.vis import plot_agent_batch, plot_agent_batch_all, draw_map
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def get_bins(num_bins=128, range_min=-18, range_max=18):
    # Define ranges and bins for coarse and fine bins
    fine_bins = 20
    fine_range_min = -1
    fine_range_max = 1
    epsilon = 1e-5

    # Calculate the number of bins for the coarse regions
    coarse_bins_each_side = (num_bins - fine_bins -2) // 2
    fine_bins_each_side = fine_bins // 2

    # Calculate the bin edges
    coarse_left_bins = np.linspace(range_min, fine_range_min, coarse_bins_each_side + 1, endpoint=False)
    # fine_bins = np.linspace(fine_range_min, fine_range_max, fine_bins + 1)
    coarse_right_bins = np.linspace(fine_range_max, range_max, coarse_bins_each_side + 1)[1:]  # Exclude the first element to avoid overlap
    fine_bins_left = np.linspace(-1, -epsilon, fine_bins_each_side + 1, endpoint=False)[1:]
    fine_bins_right = np.linspace(epsilon, 1, fine_bins_each_side + 1, endpoint=False)[1:]


    # Define zero-centered bins explicitly
    zero_bins = [-epsilon, 0, epsilon][1:]  

    # Combine all bins
    bins = np.concatenate((coarse_left_bins, fine_bins_left, zero_bins, fine_bins_right, coarse_right_bins))
    # print("num bins:", len(bins))
    return bins

def rotate_batched_trajectories(x_coords, y_coords, thetas):
    B, T = x_coords.shape  # Assumes x_coords and y_coords are of shape [B, T]
    
    # Initialize arrays to hold the rotated coordinates
    rotated_x = torch.zeros_like(x_coords)
    rotated_y = torch.zeros_like(y_coords)
    
    # Process each batch
    for i in range(B):
        theta = thetas[i]

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrix = torch.tensor([[cos_theta, -sin_theta],
                                    [sin_theta,  cos_theta]], device=x_coords.device)
        
        # Stack the coordinates for the current batch to form the matrix of points
        points = torch.stack((x_coords[i, :], y_coords[i, :]))
        
        # Rotate points by the current batch's rotation matrix
        rotated_points = torch.mm(rotation_matrix, points)
        
        # Store the rotated coordinates back into their respective arrays
        rotated_x[i, :] = rotated_points[0, :]
        rotated_y[i, :] = rotated_points[1, :]
    
    return rotated_x, rotated_y


def get_bin_center(bin_index):
    """ Compute the center of a bin accurately especially around zero. """
    left_edge = bins[bin_index]
    right_edge = bins[bin_index + 1]
    center = (left_edge + right_edge) / 2
    return center

def delta_to_bin(delta, bins):
    """ Convert a delta to a bin index ensuring zero delta points to zero bin. """
    bin_index = np.digitize(delta.cpu().numpy(), bins) - 1
    return bin_index


def deltas_to_vocab_indices(x_deltas, y_deltas, bins, num_bins):
    """ Convert x and y deltas to a vocabulary index. """
    pad_index = num_bins * num_bins
    B, T = x_deltas.shape
    vocabulary_indices = torch.zeros((B, T), dtype=torch.int, device=x_deltas.device)

    for b in range(B):
        x_bin_indices = torch.tensor([delta_to_bin(delta, bins) if not torch.isnan(delta) else pad_index for delta in x_deltas[b]], device=x_deltas.device)
        y_bin_indices = torch.tensor([delta_to_bin(delta, bins) if not torch.isnan(delta) else pad_index for delta in y_deltas[b]], device=y_deltas.device)
        vocabulary_indices[b] = torch.where(torch.logical_or(x_bin_indices == pad_index, y_bin_indices == pad_index),
                                            pad_index,
                                            x_bin_indices * num_bins + y_bin_indices)


    return vocabulary_indices

# normalizes the trajectory of a single agent to start at (0, 0) and have initial heading at 0
def normalize_trajectory(agent_hist, agent_fut):
    whole_agent_traj_x = torch.cat((agent_hist.get_attr('x'), agent_fut.get_attr('x')), dim=1)
    whole_agent_traj_y = torch.cat((agent_hist.get_attr('y'), agent_fut.get_attr('y')), dim=1)
    whole_agent_traj_heading = torch.cat((agent_hist.get_attr('h'), agent_fut.get_attr('h')), dim=1)
    
    # normalize x and y to start at (0, 0)
    initial_x = whole_agent_traj_x[:, [0]] # for recon 
    initial_y = whole_agent_traj_y[:, [0]]
    whole_agent_traj_x_norm = whole_agent_traj_x - whole_agent_traj_x[:, [0]]
    whole_agent_traj_y_norm = whole_agent_traj_y - whole_agent_traj_y[:, [0]]

    # rotate whole trajectory to have initial heading at 0
    initial_heading = whole_agent_traj_heading[:, 0] # for recon
    normalize_heading = 0 - initial_heading
    rotated_x, rotated_y = rotate_batched_trajectories(whole_agent_traj_x_norm, whole_agent_traj_y_norm, normalize_heading)

    return whole_agent_traj_x, whole_agent_traj_y, rotated_x, rotated_y, initial_x, initial_y, initial_heading

def normalize_trajectory_relative(agent_hist, agent_fut, initial_x, initial_y, initial_heading):
    whole_agent_traj_x = torch.cat((agent_hist.get_attr('x'), agent_fut.get_attr('x')), dim=1)
    whole_agent_traj_y = torch.cat((agent_hist.get_attr('y'), agent_fut.get_attr('y')), dim=1)
    
    # normalize x and y relative to given initial_x and initial_y
    whole_agent_traj_x_norm = whole_agent_traj_x - initial_x
    whole_agent_traj_y_norm = whole_agent_traj_y - initial_y

    # rotate whole trajectory to have initial heading at 0
    normalize_heading = 0 - initial_heading
    rotated_x, rotated_y = rotate_batched_trajectories(whole_agent_traj_x_norm, whole_agent_traj_y_norm, normalize_heading)

    agent_heading = agent_hist.get_attr('h')[:, 0]

    return rotated_x, rotated_y, rotated_x[:, [0]], rotated_y[:, [0]], agent_heading - initial_heading

# normalizes the trajectory of the ego agent and neighbors relative to the ego agent
def normalize_batch_relative_to_ego(batch: AgentBatch):
    results = {}
    ego_agent_x = torch.cat((batch.agent_hist.get_attr('x'), batch.agent_fut.get_attr('x')), dim=1)
    ego_agent_y = torch.cat((batch.agent_hist.get_attr('y'), batch.agent_fut.get_attr('y')), dim=1)
    # import ipdb; ipdb.set_trace()
    results['ego_orig_x'] = ego_agent_x
    results['ego_orig_y'] = ego_agent_y

    initial_x = ego_agent_x[:, [0]] 
    initial_y = ego_agent_y[:, [0]]
    initial_heading = batch.agent_hist.get_attr('h')[:, 0]

    # first, normalize our ego agent 
    ego_agent_x_norm = ego_agent_x - initial_x
    ego_agent_y_norm = ego_agent_y - initial_y

    # rotate whole trajectory to have initial heading at 0
    normalize_heading = 0 - initial_heading
    ego_x, ego_y = rotate_batched_trajectories(ego_agent_x_norm, ego_agent_y_norm, normalize_heading)

    results['ego_norm_x'] = ego_x
    results['ego_norm_y'] = ego_y

    # now, iterate over neighbors and normalize them relative to ego agent
    B, num_neigh, T = batch.neigh_hist.get_attr('x').shape
    for i in range(num_neigh):
        neigh_x = torch.cat((batch.neigh_hist.get_attr('x')[:, i, :], batch.neigh_fut.get_attr('x')[:, i, :]), dim=1)
        neigh_y = torch.cat((batch.neigh_hist.get_attr('y')[:, i, :], batch.neigh_fut.get_attr('y')[:, i, :]), dim=1)

        results[f'neigh_orig_x_{i}'] = neigh_x
        results[f'neigh_orig_y_{i}'] = neigh_y

        neigh_x_norm = neigh_x - initial_x
        neigh_y_norm = neigh_y - initial_y
        neigh_x_rot, neigh_y_rot = rotate_batched_trajectories(neigh_x_norm, neigh_y_norm, normalize_heading)
        print(i)

        results[f'neigh_norm_x_{i}'] = neigh_x_rot
        results[f'neigh_norm_y_{i}'] = neigh_y_rot

    return results

# return a 2 (or 3) x N agent tensor with the initial position of the agents present
# in global coordinates 
# should be in order of neighbor 1, neighbor 2, ..., ego agent
def get_global_initial_locations_batch(batch: AgentBatch, num_neighs, include_heading=False):
    locs = []
    for ag in range(num_neighs):
        neigh_x = batch.neigh_hist[:, ag, 0].get_attr('x')
        neigh_y = batch.neigh_hist[:, ag, 0].get_attr('y')
        neigh_h = batch.neigh_hist[:, ag, 0].get_attr('h')

        if include_heading:
            init_loc = torch.stack((neigh_x, neigh_y, neigh_h), dim=1)
        else:
            init_loc = torch.stack((neigh_x, neigh_y), dim=1)
        locs.append(init_loc) # torch.size([B, 2/3])
    
    ego_x = batch.agent_hist[:, 0].get_attr('x')
    ego_y = batch.agent_hist[:, 0].get_attr('y')
    ego_h = batch.agent_hist[:, 0].get_attr('h')
    if include_heading:
        init_loc = torch.stack((ego_x, ego_y, ego_h), dim=1)
    else:
        init_loc = torch.stack((ego_x, ego_y), dim=1)  # torch.size([B, 2/3])
    locs.append(init_loc)

    return torch.stack(locs, dim=1) # torch.size([B, N, 2/3])
        

# return a 2 (or 3) x N agent tensor with the initial position of the agents
# in relative coordinates to the ego agent
# should be in order of neighbor 1, neighbor 2, ..., ego agent
def get_relative_initial_locations_batch(batch: AgentBatch, num_neighs, include_heading=False):
    _, _, _, _, init_x_ego, init_y_ego, init_h_ego = normalize_trajectory(batch.agent_hist, batch.agent_fut)

    locs = []
    for ag in range(num_neighs):
        _, _, init_x, init_y, init_h = normalize_trajectory_relative(batch.neigh_hist[:, ag], batch.neigh_fut[:, ag], init_x_ego, init_y_ego, init_h_ego)
        init_x = init_x.squeeze(1)
        init_y = init_y.squeeze(1) 
        if include_heading:
            init_loc = torch.stack((init_x, init_y, init_h), dim=1)
        else:
            init_loc = torch.stack((init_x, init_y), dim=1)
        locs.append(init_loc) # torch.size([B, 2/3])
    elems = 3 if include_heading else 2
    ego_loc = torch.zeros((batch.agent_hist.get_attr('x').shape[0], elems), device=init_x.device) # torch.size([B, 2/3])
    locs.append(ego_loc)

    return torch.stack(locs, dim=1) # torch.size([B, N, 2/3])


def get_tokens(norm_x, norm_y, bins):
    deltas_x = torch.diff(norm_x, axis=1)
    deltas_y = torch.diff(norm_y, axis=1)

    num_bins = len(bins) - 1
    bin_indices = deltas_to_vocab_indices(deltas_x, deltas_y, bins, num_bins)
    return bin_indices


def reconstruct_trajectory(start_x, start_y, vocabulary_indices, bins, num_bins):
    """ Reconstruct x and y trajectories from vocabulary indices. """

    pad_index = num_bins * num_bins

    B, T = vocabulary_indices.shape
    x_recon = torch.zeros((B, T + 1), device=start_x.device, dtype=start_x.dtype)
    y_recon = torch.zeros((B, T + 1), device=start_y.device, dtype=start_y.dtype)

    x_recon[:, 0] = start_x.flatten() 
    y_recon[:, 0] = start_y.flatten()

    for b in range(B):
        current_x = start_x[b]
        current_y = start_y[b]
        for t in range(T):
            index = vocabulary_indices[b, t].item()
            if index == pad_index:
                # set nan 
                x_recon[b, t + 1] = float('nan')
                y_recon[b, t + 1] = float('nan')
            else:
                x_idx, y_idx = divmod(index, num_bins)
                delta_x = (torch.tensor(bins[x_idx], device=start_x.device) + torch.tensor(bins[x_idx + 1], device=start_x.device)) / 2
                delta_y = (torch.tensor(bins[y_idx], device=start_y.device) + torch.tensor(bins[y_idx + 1], device=start_y.device)) / 2
                current_x += delta_x
                current_y += delta_y
                x_recon[b, t + 1] = current_x
                y_recon[b, t + 1] = current_y

    return x_recon, y_recon

def recon_and_unnormalize(tokens, bins, num_bins, initial_x, initial_y, initial_heading):
    B, T = tokens.shape
    x_recon, y_recon = reconstruct_trajectory(torch.zeros((B, 1), device=tokens.device), torch.zeros((B, 1), device=tokens.device), tokens, bins, num_bins)
    x_recon, y_recon = rotate_batched_trajectories(x_recon, y_recon, initial_heading)
    # add initial_x and initial_y back to x_recon and y_recon
    x_recon = x_recon + initial_x
    y_recon = y_recon + initial_y
    return x_recon, y_recon

# initial_x and initial_y are the initial position of the ego agent
# first reconstruct trajectory starting at initial_x and initial_y for the neighbor in the ego agents relative frame
# then un-rotate 
# then go from ego agent relative to global
def recon_and_unnormalize_relative(tokens, bins, num_bins, initial_x, initial_y, initial_heading, initial_x_relative, initial_y_relative):
    B, T = tokens.shape
    x_recon, y_recon = reconstruct_trajectory(initial_x.unsqueeze(1), initial_y.unsqueeze(1), tokens, bins, num_bins)
    x_recon, y_recon = rotate_batched_trajectories(x_recon, y_recon, initial_heading)
    # add initial_x and initial_y back to x_recon and y_recon
    x_recon = x_recon + initial_x_relative
    y_recon = y_recon + initial_y_relative
    return x_recon, y_recon
