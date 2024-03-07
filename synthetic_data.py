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
from data_utils import *

def generate_synthetic(tri_length, squ_length, cir_length,
                       SBR, depth, background, save_path,
                       reflectivity=1, resolution=40e-12, bin_count=1024, pixel=64):
    """
    Input:
        tri_length: the number of triangular
        squ_length: the number of square
        cir_length: the number of circle
        SBR: the range of SBR, we set [0.1, 0.9]
        depth: the range of depth, we set [200, 500]
        background: the range of background photons, we set [8, 15]
        save_path: the path where the file is saved
        reflectivity: reflectivity of the pixel that contains target, we set 1
        resolution: receiver resolution
        bin_count: the length of time window
        pixel: the size of the 2D image

    Return:
        generate the synthetic data of different shapes of triangular, square, circle
    """
    data_length = tri_length + squ_length + cir_length
    c = 3e+8

    SBR_set = np.random.uniform(SBR[0], SBR[1], data_length)
    depth_set = np.random.randint(depth[0], depth[1], data_length)
    reflectivity_set = reflectivity
    background_set = np.random.randint(background[0], background[1], data_length)

    T = np.arange(bin_count)
    r = poisson.pmf(T, 8)
    # t_delay = int(np.ceil(2 * depth_set * 1e-2 / c / resolution))  # 延迟的时间长度

    id_overall = 0

    id_tri = 0
    tri_start_x = np.random.randint(16, 48, tri_length)
    tri_start_y = np.random.randint(1, 48, tri_length)
    tri_size = np.random.randint(7, 17, tri_length)

    id_squ = 0
    squ_start_x = np.random.randint(1, 48, squ_length)
    squ_start_y = np.random.randint(1, 48, squ_length)
    squ_width = np.random.randint(7, 17, squ_length)
    squ_height = np.random.randint(7, 17, squ_length)

    id_cir = 0
    cir_start_x = np.random.randint(9, 55, cir_length)
    cir_start_y = np.random.randint(9, 55, cir_length)
    cir_size = np.random.randint(4, 10, cir_length)

    index = [i for i in range(data_length)]
    random.shuffle(index)

    for i in range(data_length):

        background = np.zeros([pixel, pixel])
        background[:, :] = background_set[i]
        wav_k = SBR_set[i] * background_set[i] / sum(r) / (1 - SBR_set[i])
        R_real = wav_k * r

        t_delay = int(np.ceil(2 * depth_set[i] * 1e-2 / c / resolution))
        # angle = np.random.uniform(325, 360)

        if (index[i] >= 0) and (index[i] <= tri_length-1):
            li_tri = []
            label_tri = 0
            reflectivity_tri = np.zeros([pixel, pixel])
            flag_signal = np.zeros([pixel, pixel])
            for ii in range(tri_size[id_tri]):
                tri_x = np.arange(tri_start_x[id_tri] - ii, tri_start_x[id_tri] + ii + 1, dtype=int)
                tri_y = tri_start_y[id_tri] + ii
                reflectivity_tri[tri_x, tri_y] = reflectivity_set
                flag_signal[tri_x, tri_y] = 1

            tof = measure_tof(background_set[i], reflectivity_tri, flag_signal, bin_count, R_real, t_delay)
            # tof = torch.round(torch.from_numpy(rotate(tof[0, :, :, :], angle, reshape=False, mode='nearest')))
            li_tri.append(tof)
            li_tri.append(label_tri)
            torch.save(li_tri, save_path)

            id_tri = id_tri + 1
            id_overall = id_overall + 1

        elif index[i] >= tri_length and index[i] <= tri_length + squ_length - 1:
            li_squ = []
            label_squ = 1
            reflectivity_squ = np.zeros([pixel, pixel])
            flag_signal = np.zeros([pixel, pixel])
            reflectivity_squ[squ_start_x[id_squ]:squ_start_x[id_squ] + squ_width[id_squ],
            squ_start_y[id_squ]:squ_start_y[id_squ] + squ_height[id_squ]] = reflectivity_set
            flag_signal[squ_start_x[id_squ]:squ_start_x[id_squ] + squ_width[id_squ],
            squ_start_y[id_squ]:squ_start_y[id_squ] + squ_height[id_squ]] = 1

            tof = measure_tof(background_set[i], reflectivity_squ, flag_signal, bin_count, R_real, t_delay)
            # tof = torch.round(torch.from_numpy(rotate(tof[0, :, :, :], angle, reshape=False, mode='nearest')))
            li_squ.append(tof)
            li_squ.append(label_squ)

            torch.save(li_squ, save_path)

            id_squ = id_squ + 1
            id_overall = id_overall + 1

        elif index[i] >= tri_length + squ_length and index[i] <= tri_length + squ_length + cir_length - 1:
            li_cir = []
            label_cir = 2
            reflectivity_cir = np.zeros([pixel, pixel])
            flag_signal = np.zeros([pixel, pixel])
            for ii in range(cir_start_x[id_cir] - cir_size[id_cir], cir_start_x[id_cir] + cir_size[id_cir] + 1):
                for jj in range(cir_start_y[id_cir] - cir_size[id_cir], cir_start_y[id_cir] + cir_size[id_cir] + 1):
                    if (ii - cir_start_x[id_cir]) ** 2 + (jj - cir_start_y[id_cir]) ** 2 <= cir_size[id_cir] ** 2:
                        reflectivity_cir[ii, jj] = reflectivity_set
                        flag_signal[ii, jj] = 1

            tof = measure_tof(background_set[i], reflectivity_cir, flag_signal, bin_count, R_real, t_delay)
            # tof = torch.round(torch.from_numpy(rotate(tof[0, :, :, :], angle, reshape=False, mode='nearest')))
            li_cir.append(tof)
            li_cir.append(label_cir)

            torch.save(li_cir, save_path)

            id_cir = id_cir + 1
            id_overall = id_overall + 1

    return
