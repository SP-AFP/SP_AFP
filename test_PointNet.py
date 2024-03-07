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
from  PFSP_PointNet import *
import importlib
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--FN_channel', default=2, type=int,  help='channels of test data')
    parser.add_argument('--test_data_path', default='Data\\test', help='the path of test data')
    parser.add_argument('--channel', default=3, type=int, help='channels of point cloud inputs')
    parser.add_argument('--FN_path', default='models\\adaptive_filter_pointnet_fil2.pth', help='the path of trained FN')
    parser.add_argument('--PointNet_path', default='models\\PFSP_PointNet.pth', help='the path of trained PFSP-PointNet')
    return parser.parse_args()

args = parse_args()

def main(args):

    '''LOAD DATA'''
    print('Load dataset ...')
    test_dataset = MyData(args.test_data_path, "")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''INITIALIZATION'''
    ball_label = 2
    cub_label = 1
    pyr_label = 0

    ball_num = 0
    ball_predict_num = 0
    ball_right_num = 0

    cub_num = 0
    cub_predict_num = 0
    cub_right_num = 0

    pyr_num = 0
    pyr_predict_num = 0
    pyr_right_num = 0

    # map50
    ball_true = torch.zeros(len(test_dataset))
    ball_score = torch.zeros(len(test_dataset))
    cub_true = torch.zeros(len(test_dataset))
    cub_score = torch.zeros(len(test_dataset))
    pyr_true = torch.zeros(len(test_dataset))
    pyr_score = torch.zeros(len(test_dataset))

    '''MODEL LOADING'''
    print('Load Filtering-Network ...')
    FN = Ada_fil2_pointnet(args.FN_channel)
    FN.load_state_dict(args.FN_path)

    print('Load PFSP-PointNet ...')
    classifier = PointNet(args.channel)
    classifier.load_state_dict(args.PointNet_path)

    if not args.use_cpu:
        FN = FN.cuda()
        classifier = classifier.cuda()

    for i in range(len(test_dataset)):
        point_cloud = test_dataset[i][0]
        pcd_label = test_dataset[i][2]

        density = density_generate(point_cloud)
        FN_inputs = torch.zeros(len(density), 2)
        FN_inputs[:, 0] = 16
        FN_inputs[:, 1] = density
        pred_par = FN(FN_inputs)
        pred_par = pred_par.reshape(-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        ror_pcd_1, ind_1 = pcd.remove_radius_outlier(pred_par[1], pred_par[0])
        ror_pcd_2, ind_2 = ror_pcd_1.remove_radius_outlier(pred_par[3], pred_par[2])

        point_cloud_fil = torch.from_numpy(np.asarray(ror_pcd_2.points))
        point_cloud_fil = point_cloud_fil.permute(0, 2, 1)

        if not args.use_cpu:
            point_cloud_fil = point_cloud_fil.cuda()
            pcd_label = pcd_label.cuda()

        outputs = classifier(point_cloud_fil)
        outputs = torch.exp(outputs)
        pred_label = outputs.argmax(1)

        if pcd_label == ball_label:
            ball_num = ball_num + 1
            ball_true[i] = 1
        elif pcd_label == cub_label:
            cub_num = cub_num + 1
            cub_true[i] = 1
        elif pcd_label == pyr_label:
            pyr_num = pyr_num + 1
            pyr_true[i] = 1

        if pred_label == ball_label:
            ball_predict_num = ball_predict_num + 1
        elif pred_label == cub_label:
            cub_predict_num = cub_predict_num + 1
        elif pred_label == pyr_label:
            pyr_predict_num = pyr_predict_num + 1

        if pred_label == pcd_label == ball_label:
            ball_right_num = ball_right_num + 1
        elif pred_label == pcd_label == cub_label:
            cub_right_num = cub_right_num + 1
        if pred_label == pcd_label == pyr_label:
            pyr_right_num = pyr_right_num + 1

        ball_score[i] = outputs[0][2]
        cub_score[i] = outputs[0][1]
        pyr_score[i] = outputs[0][0]

        precision_ball, recall_ball, _ = precision_recall_curve(ball_true.detach(), ball_score.detach())
        pr_auc_ball = auc(recall_ball, precision_ball)
        precision_cub, recall_cub, _ = precision_recall_curve(cub_true.detach(), cub_score.detach())
        pr_auc_cub = auc(recall_cub, precision_cub)
        precision_pyr, recall_pyr, _ = precision_recall_curve(pyr_true.detach(), pyr_score.detach())
        pr_auc_pyr = auc(recall_pyr, precision_pyr)

    return







