#  -*- coding:utf-8 -*-
"""
Author: Vicktor
Date: Jan 2024
"""

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
import os

# -----------------------------------------------------------------------------
# CALCULATE THE NUMBER OF POINTS WITHIN A GIVEN RADIUS
# -----------------------------------------------------------------------------
def count_points_in_circle(center, radius, point_cloud):
    """
    Input:
        center (tuple or list): center of the circle, (x, y, z)
        radius (float): radius of circle
        point_cloud (numpy.ndarray): pointcloud data, [B, N, 3]
    Return:
        num_points_inside(int): number of points in the circle
    """
    distances = np.linalg.norm(point_cloud - np.array(center), axis=1)
    num_points_inside = np.sum(distances <= radius)

    return num_points_inside

# -----------------------------------------------------------------------------
# DOWNSAMPLE DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

# -----------------------------------------------------------------------------
# EVALUATION FOR RADIUS FILTER
# -----------------------------------------------------------------------------
def eval_rafil_fil2(s1, n1, s2, n2, mu, lamda):
    """
    Input:
        s1: target photons before filter
        n1: noise photons before filter
        s2: target photons after filter
        n2: noise photons after filter
        mu, lamda: weight parameters
    Return:
        score: filtering performance
    """
    score = mu * s2 / (n2 + 1) - lamda * (s1 - s2) / s1
    return score

# -----------------------------------------------------------------------------
# ZERO-MEAN NORMALIZE THE DATA
# -----------------------------------------------------------------------------
def z_score_normalize(array):
    """
    Input:
        array (numpy.ndarray): array to be normalized
    Return:
        normalized_array: normalized array
    """
    mean_val = np.mean(array)
    std_val = np.std(array)
    normalized_array = (array - mean_val) / std_val
    return normalized_array

# -----------------------------------------------------------------------------
# READ DATA
# -----------------------------------------------------------------------------
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = torch.load(img_item_path)
        return img

    def __len__(self):
        return len(self.img_path)