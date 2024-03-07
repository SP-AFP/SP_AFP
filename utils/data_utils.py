#  -*- coding:utf-8 -*-
"""
Author: Vicktor
Date: Jan 2024
"""

import random
import torch
import numpy as np
import os
from scipy.stats import poisson
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# SIMULATE THE PROCESSING OF MEASURING TOF
# -----------------------------------------------------------------------------
def measure_tof(background_set, reflectivity, flag_signal, bin_count, R_real, t_delay, pixel=64):  # 测量得到tof的函数，需要的输入：深度、SBR、背景、反射率、有信号的标识、时间窗长度、光源的脉冲函数（如果深度相同、可以输入延迟时间）
    """
    Input:
        background_set: the number of background photons
        reflectivity: reflectance map
        flag_signal: a category map of each pixel (signal or noise)
        bin_count: the length of time window
        R_real: the impulse response function S of the system
        t_delay: acceptance delay time due to depth
        pixel: the size of the 2D image
    Return:
        tof: the 3D matrix H and the category map of each point (signal or noise)
    """

    T = np.arange(bin_count)
    background_efficient = background_set / bin_count
    signal_efficient_id = np.where(R_real > 1e-2)
    signal_efficient = R_real[signal_efficient_id]
    signal_efficient_length = len(signal_efficient)
    tof = torch.zeros(2, bin_count, pixel, pixel)
    for i in range(pixel):
        for j in range(pixel):
            ts = np.zeros([bin_count, 2])
            if flag_signal[i, j] == 1:

                prob_signal = np.random.rand(signal_efficient_length)
                for jj in range(signal_efficient_length):
                    signal_poisson = poisson.pmf(T, reflectivity[i, j] * signal_efficient[jj] + background_efficient)
                    t_1 = np.where(prob_signal[jj] < signal_poisson)
                    if len(t_1[0]) > 1:
                        ts[signal_efficient_id[0][jj] + t_delay][0] = max(t_1[0])
                        ts[signal_efficient_id[0][jj] + t_delay][1] = 1

                background_poisson = poisson.pmf(T, background_efficient)
                background_length = bin_count - signal_efficient_length
                prob_background = np.random.rand(background_length)
                for tt in range(background_length):
                    t_2 = np.where(prob_background[tt] < background_poisson)
                    if tt < signal_efficient_id[0][0] + t_delay :
                        if len(t_2[0]) > 1:
                            ts[tt][0] = max(t_2[0])
                            ts[tt][1] = 0
                    else:
                        if len(t_2[0]) > 1:
                            ts[tt + signal_efficient_length][0] = max(t_2[0])
                            ts[tt + signal_efficient_length][1] = 0
                tof[0, :, i, j] = torch.tensor(ts[:, 0]).reshape(bin_count)
                tof[1, :, i, j] = torch.tensor(ts[:, 1]).reshape(bin_count)
            else:
                background_poisson = poisson.pmf(T, background_efficient)
                prob_background = np.random.rand(bin_count)
                for tt in range(bin_count):
                    t_2 = np.where(prob_background[tt] < background_poisson)
                    if len(t_2[0]) > 1:
                        ts[tt][0] = max(t_2[0])
                        ts[tt][1] = 0
                tof[0, :, i, j] = torch.tensor(ts[:, 0]).reshape(bin_count)
                tof[1, :, i, j] = torch.tensor(ts[:, 1]).reshape(bin_count)
    return tof

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