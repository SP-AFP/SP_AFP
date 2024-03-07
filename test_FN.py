#  -*- coding:utf-8 -*-
"""
Author: Vicktor
Date: Jan 2024
"""

import os
import torch
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import nn
from filter_utils import *
from Filtering_Network import *
import argparse
from ASPRF import *
import importlib
import logging

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--channel', default=2, type=int,  help='channels of test data')
    parser.add_argument('--model', default='Ada_fil2_pointnet', help='model name [default: FN]')
    parser.add_argument('--test_data_path', default='Data\\FN\\test', help='the path of test data')
    return parser.parse_args()

args = parse_args()

def net_test(model, test_data, R_seg=16):
    f_n = model.eval()
    point_cloud, point_cloud_label = test_data[0], test_data[1]
    density = density_generate(point_cloud)
    inputs = torch.zeros(len(density), 2)
    inputs[:, 0] = R_seg
    inputs[:, 1] = density
    if not args.use_cpu:
        inputs = inputs.cuda()
    pred_par = f_n((inputs.reshape(1, -1, 2)).permute(0, 2, 1))
    return pred_par



def main(args, test_dataset):

    '''LOAD DATA'''
    print('Load dataset ...')
    test_dataset = MyData(args.test_data_path, "")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    '''INITIALIZATION'''
    pred_results = []
    asprf_results = []

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = "adaptive_filter_pointnet_fil2.pth"

    '''DATA LOADING'''
    print('Load dataset ...')
    data_path = 'models/'

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    FN = model.get_model(channel=args.channel)
    FN.load_state_dict(torch.load(os.path.join(data_path, experiment_dir)))

    if not args.use_cpu:
        FN = FN.cuda()

    for i in range(len(test_dataset)):
        point_cloud = test_dataset[i][0]
        point_cloud_label = test_dataset[i][1]

        pred_par = net_test(FN, test_dataset[i], R_seg=16)
        pred_par = pred_par.reshape(-1)
        r_pred_1, r_pred_2 = pred_par[0], pred_par[2]
        lamda_pred_1, lamda_pred_2 = pred_par[1], pred_par[3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        ror_pcd_trained_fil1, ind_trained_fil1 = pcd.remove_radius_outlier(lamda_pred_1, r_pred_1)
        pcd_filter_trained_label_fil1 = point_cloud_label[ind_trained_fil1]
        ror_pcd_trained_fil2, ind_trained_fil2 = ror_pcd_trained_fil1.remove_radius_outlier(lamda_pred_2, r_pred_2)
        pcd_filter_trained_label_fil2 = pcd_filter_trained_label_fil1[ind_trained_fil2]
        score_pred = eval_rafil_fil2(torch.sum(point_cloud_label == 1), torch.sum(point_cloud_label == 0),
                                         torch.sum(pcd_filter_trained_label_fil2 == 1),
                                         torch.sum(pcd_filter_trained_label_fil2 == 0))
        pred_results.append(score_pred)


        par_range = range_generate(r_range=[4, 10], lamda_range=[15, 100])
        asprf_par = Asprf(point_cloud, point_cloud_label, par_range)
        asprf_par = asprf_par.reshape(-1)
        r_asprf_1, r_asprf_2 = asprf_par[0], asprf_par[1]
        lamda_asprf_1, lamda_asprf_2 = asprf_par[2], asprf_par[3]

        ror_pcd_asprf_fil1, ind_asprf_fil1 = pcd.remove_radius_outlier(lamda_asprf_1, r_asprf_1)
        pcd_filter_asprf_label_fil1 = point_cloud_label[ind_asprf_fil1]
        ror_pcd_asprf_fil2, ind_asprf_fil2 = ror_pcd_asprf_fil1.remove_radius_outlier(lamda_asprf_2,r_asprf_2)
        pcd_filter_asprf_label_fil2 = pcd_filter_asprf_label_fil1[ind_asprf_fil2]
        score_asprf = eval_rafil_fil2(torch.sum(point_cloud_label == 1), torch.sum(point_cloud_label == 0),
                                        torch.sum(pcd_filter_asprf_label_fil2 == 1),
                                        torch.sum(pcd_filter_asprf_label_fil2 == 0))
        asprf_results.append(score_asprf)

    plt.plot(pred_results, label='FN')
    plt.plot(asprf_results, label='ASPRF')
    plt.title('Performance of FN and ASPRF')
    plt.xlabel('Times')
    plt.ylabel('Score')
    plt.legend()
    plt.show()


