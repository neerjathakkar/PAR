import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os

from trajdata import AgentBatch, UnifiedDataset, AgentType
from trajdata.visualization.vis import plot_agent_batch, plot_agent_batch_all, draw_map
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from .tokenizer import normalize_trajectory, rotate_batched_trajectories
# from tokenizer import normalize_trajectory, rotate_batched_trajectories

def get_bins(num_bins=128, range_min=-18, range_max=18):
    return np.linspace(range_min, range_max, num_bins + 1)
def get_bins_first_order(num_bins=128, range_min=-18, range_max=18):
    # Define ranges and bins for coarse and fine bins
    fine_bins = 20
    fine_range_min = -2
    fine_range_max = 2
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


def get_bin_center(bin_index, bins):
    """ Compute the center of a bin accurately especially around zero. """
    left_edge = bins[bin_index]
    right_edge = bins[bin_index + 1]
    center = (left_edge + right_edge) / 2
    return center

def delta_to_bin(delta, bins):
    """ Convert a delta to a bin index ensuring zero delta points to zero bin. """
    bin_index = np.digitize(delta.cpu().numpy(), bins) - 1
    return bin_index

def get_second_order_dict(n):
    middle = n//2 
    second_order_bins_dict = {i - middle: i for i in range(n)}
    return second_order_bins_dict


def delta_to_second_order_bin(delta, bins_dict, num_second_bins):
    n = num_second_bins//2
    if delta < n*-1:
        delta = n*-1
    elif delta > n:
        delta = n
    return bins_dict.get(int(delta), -1)  # Return -1 if delta is not found in dict


def deltas_to_vocab_indices(x_deltas, y_deltas, first_order_bins, second_order_bins_dict, second_order_num_bins=13):
    """ Convert x and y deltas to a vocabulary index. """
    pad_index = second_order_num_bins * second_order_num_bins
    B, T = x_deltas.shape
    vocabulary_indices = torch.zeros((B, T-1), dtype=torch.int, device=x_deltas.device)

    for b in range(B):
        x_bin_indices = torch.tensor([delta_to_bin(delta, first_order_bins) for delta in x_deltas[b]], device=x_deltas.device)
        y_bin_indices = torch.tensor([delta_to_bin(delta, first_order_bins) for delta in y_deltas[b]], device=y_deltas.device)
        # print("bin_indices")
        # print(x_bin_indices)
        # print(y_bin_indices)

        x_idx_deltas = torch.diff(x_bin_indices)
        y_idx_deltas = torch.diff(y_bin_indices)

        # print("bin deltas")
        # print(x_idx_deltas)
        # print(y_idx_deltas)

        x_bin_second_order_indices = torch.tensor([delta_to_second_order_bin(delta, second_order_bins_dict, second_order_num_bins) if not torch.isnan(delta) else pad_index for delta in x_idx_deltas], device=x_deltas.device)
        y_bin_second_order_indices = torch.tensor([delta_to_second_order_bin(delta, second_order_bins_dict, second_order_num_bins) if not torch.isnan(delta) else pad_index for delta in y_idx_deltas], device=y_deltas.device)

        # print("second order bin indices")
        # print(x_bin_second_order_indices)
        # print(y_bin_second_order_indices)

        vocabulary_indices[b] = torch.where(torch.logical_or(x_bin_second_order_indices == pad_index, y_bin_second_order_indices == pad_index),
                                            pad_index,
                                            x_bin_second_order_indices * second_order_num_bins + y_bin_second_order_indices)
        # print("vocab")
        # print(vocabulary_indices[b])

    return vocabulary_indices

def get_tokens_accel(norm_x, norm_y, first_order_bins, second_order_bins_dict, second_order_num_bins):
    deltas_x = torch.diff(norm_x, axis=1)
    deltas_y = torch.diff(norm_y, axis=1)

    bin_indices = deltas_to_vocab_indices(deltas_x, deltas_y, first_order_bins, second_order_bins_dict, second_order_num_bins)
    return bin_indices


def reconstruct_trajectory_accel(start_x, start_y, start_vel_x, start_vel_y, vocabulary_indices, first_order_bins, num_second_order_bins, second_order_bins_dict):
    """ Reconstruct x and y trajectories from vocabulary indices, taking into account acceleration. """
    pad_index = num_second_order_bins * num_second_order_bins  # Update this if your pad_index calculation needs to change
    
    # Reverse the second_order_bins_dict for reconstruction
    reverse_second_order_bins_dict = {v: k for k, v in second_order_bins_dict.items()}

    B, T = vocabulary_indices.shape
    # breakpoint()
    x_recon = torch.zeros((B, T + 2), device=start_x.device, dtype=start_x.dtype)
    y_recon = torch.zeros((B, T + 2), device=start_y.device, dtype=start_y.dtype)

    # Setting initial positions and velocities
    x_recon[:, 0] = start_x.flatten()
    y_recon[:, 0] = start_y.flatten()
    current_vel_x = start_vel_x
    current_vel_y = start_vel_y

    recon_x_vel = []
    recon_y_vel = []

    for b in range(B):
        current_x = start_x[b]
        current_y = start_y[b]
        velocity_x = current_vel_x[b]
        velocity_y = current_vel_y[b]
        recon_x_vel.append(velocity_x)
        recon_y_vel.append(velocity_y)

        # Map initial velocity to velocity token index space
        velocity_x_token = delta_to_bin(velocity_x, first_order_bins)
        velocity_y_token = delta_to_bin(velocity_y, first_order_bins)

        # Compute the second position (initial velocity applied)
        current_x += velocity_x
        current_y += velocity_y
        x_recon[b, 1] = current_x
        y_recon[b, 1] = current_y

        x_velocity_tokens = [velocity_x_token]
        y_velocity_tokens = [velocity_y_token]

        for t in range(T):
            index = vocabulary_indices[b, t].item()
            if index == pad_index:
                # Set NaN for padded indices, and stop velocity updates
                x_recon[b, t + 2] = float('nan')
                y_recon[b, t + 2] = float('nan')
                velocity_x = velocity_y = float('nan')  # Stop further updates

            else:
                # Retrieve second-order acceleration indices
                accel_x_idx, accel_y_idx = divmod(index, num_second_order_bins)
                # Get acceleration from second order bins dictionary
                accel_x = reverse_second_order_bins_dict.get(accel_x_idx, 0)  # Centered around zero
                accel_y = reverse_second_order_bins_dict.get(accel_y_idx, 0)  # Centered around zero

                # Update velocity tokens by applying the token-space acceleration (difference in token indices)
                velocity_x_token += accel_x
                velocity_y_token += accel_y

                x_velocity_tokens.append(velocity_x_token)
                y_velocity_tokens.append(velocity_y_token)

                # Map velocity tokens back to actual velocities
                if velocity_x_token >= 128 or velocity_y_token >= 128:
                    velocity_x = velocity_y = float('nan')
                else:
                    velocity_x = (torch.tensor(first_order_bins[velocity_x_token], device=start_x.device) + torch.tensor(first_order_bins[velocity_x_token + 1], device=start_x.device)) / 2
                    velocity_y = (torch.tensor(first_order_bins[velocity_y_token], device=start_y.device) + torch.tensor(first_order_bins[velocity_y_token + 1], device=start_y.device)) / 2

                # Update position based on velocity
                current_x += velocity_x
                current_y += velocity_y

                x_recon[b, t + 2] = current_x
                y_recon[b, t + 2] = current_y
            recon_x_vel.append(velocity_x)
            recon_y_vel.append(velocity_y)
        # print("x velotiy tokens")
        # print(x_velocity_tokens)
        # print("y velocity tokens")
        # print(y_velocity_tokens)

    # return x_recon, y_recon, recon_x_vel, recon_y_vel
    return x_recon, y_recon

def recon_and_unnormalize_accel( initial_x, initial_y, initial_heading, start_vel_x, start_vel_y, tokens, first_order_bins, num_second_order_bins, second_order_bins_dict):
    B, T = tokens.shape
    x_recon, y_recon = reconstruct_trajectory_accel(torch.zeros((B, 1), device=tokens.device), torch.zeros((B, 1), device=tokens.device), 
                                                    start_vel_x, start_vel_y,
                                                    tokens, first_order_bins, num_second_order_bins, second_order_bins_dict)
    x_recon, y_recon = rotate_batched_trajectories(x_recon, y_recon, initial_heading)
    # add initial_x and initial_y back to x_recon and y_recon
    x_recon = x_recon + initial_x
    y_recon = y_recon + initial_y
    return x_recon, y_recon
