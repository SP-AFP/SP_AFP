#  -*- coding:utf-8 -*-
"""
Author: Vicktor
Date: Jan 2024
"""

import os
import argparse
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import nn
from filter_utils import *

def range_generate(r_range, lamda_range, r_interval=0.5, lamda_interval=1):
    """ Generate the seach range of the r and thereshold
        Input:
            r_range: search for the optimal range of radius, [r_min, r_max]
            lamada_range: search for the optimal range of threshold, [lamda_min, lamda_max]
            r_interval: the interval of r_range
            lamda_interval: the interval of lamda_range
        Return:
            par_filter: search range of r* and lamda*, [r,lamda]
    """
    r_filter = np.arange(r_range[0], r_range[1], r_interval)
    lamda_filter = np.arange(lamda_range[0], lamda_range[1], lamda_interval)
    par_filter = torch.zeros(len(r_filter)*len(lamda_filter), 2)
    for i in range(len(r_filter)):
        for j in range(len(lamda_filter)):
            par_filter[i * len(lamda_filter) + j, 0] = r_filter[i]
            par_filter[i * len(lamda_filter) + j, 1] = lamda_filter[j]
    return par_filter

def density_generate(point_cloud, R_seg=64/4):
    """ Generate the density of the point cloud (Input of the Filtering Network)
        Input:
            point_cloud: pointcloud data, [B, N, 3]
            R_seg: Segmentation Ball-radius (We use R_seg=16 to segment the whole 1024*64*64 space)
        Return:
            par_den: the normalised density of the ball space
    """
    centers = [(2 * i * R_seg + R_seg, 2 * j * R_seg + R_seg, 2 * k * R_seg + R_seg)
                 for i in range(32) for j in range(2) for k in range(2)]
    pcd_den = torch.zeros(len(centers))
    for i in range(len(centers)):
        pcd_den[i] = count_points_in_circle(centers[i], R_seg, point_cloud)
    pcd_den[:] = torch.from_numpy(z_score_normalize(np.array(pcd_den[:])))
    return pcd_den

def Asprf(point_cloud, pcd_label, par_filter, M=2):
    """ search for the optimal radius filter parameters
        Input:
            point_cloud: pointcloud data, [B, N, 3]
            pcd_label: the label of each point (signal or noise)
            par_filter: search range of r* and lamda*, [r,lamda]
            M: filtering times
        Return:
            rd_fl_par: the optimal parameters, including r* and lamda*, [r1*,lamda1*,...,rM*,lamdaM*]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    eva_rd_best = 0
    rd_fil_best = torch.zeros(M)
    thre_fil_best = torch.zeros(M)
    rd_fl_par = torch.zeros(M*2)

    for id_test_fil1 in range(len(par_filter)):
        num_points_fil1 = par_filter[id_test_fil1, 1]
        radius_fil1 = par_filter[id_test_fil1, 0]
        ror_pcd_fil1, ind_fil1 = pcd.remove_radius_outlier(num_points_fil1, radius_fil1)
        pcd_filter_fil1 = torch.from_numpy(np.asarray(ror_pcd_fil1.points))
        pcd_filter_fil1_label = pcd_label[ind_fil1]
        for id_test_fil2 in range(len(par_filter)):
            num_points_fil2 = par_filter[id_test_fil2, 1]
            radius_fil2 = par_filter[id_test_fil2, 0]
            ror_pcd_fil2, ind_fil2 = ror_pcd_fil1.remove_radius_outlier(num_points_fil2, radius_fil2)
            pcd_filter_fil2 = torch.from_numpy(np.asarray(ror_pcd_fil2.points))
            pcd_filter_fil2_label = pcd_filter_fil1_label[ind_fil2]
            eva_rd = eval_rafil_fil2(torch.sum(pcd_label == 1), torch.sum(pcd_label == 0),
                            torch.sum(pcd_filter_fil2_label == 1), torch.sum(pcd_filter_fil2_label == 0))
            if eva_rd > eva_rd_best:
                rd_fil_best[0] = radius_fil1
                thre_fil_best[0] = num_points_fil1
                rd_fil_best[1] = radius_fil2
                thre_fil_best[1] = num_points_fil2
                eva_rd_best = eva_rd
    rd_fl_par[0] = rd_fil_best[0]
    rd_fl_par[2] = thre_fil_best[0]
    rd_fl_par[1] = rd_fil_best[1]
    rd_fl_par[3] = thre_fil_best[1]
    return rd_fl_par

