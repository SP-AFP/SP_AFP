#  -*- coding:utf-8 -*-
"""
Author: Vicktor
Date: Jan 2024
"""

import time
import torch
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torch.nn import Sequential
from torch.utils.data import DataLoader, Dataset
from PFSP_PointNet import *
import argparse
from Filtering_Network import *
from filter_utils import *
from ASPRF import *

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_category', default=3, type=int,  help='training on PointNet')
    parser.add_argument('--save_path', default='model\\pointnet_cls', help='path saved trained model')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--FN_path', default='models\\adaptive_filter_pointnet_fil2.pth', help='the path of trained FN')
    parser.add_argument('--FN_channel', default=2, type=int, help='channels of FN inputs')
    parser.add_argument('--channel', default=3, type=int, help='channels of point cloud inputs')
    parser.add_argument('--train_data_path', default='Data\\train', help='the path of train data')
    parser.add_argument('--test_data_path', default='Data\\test', help='the path of test data')
    return parser.parse_args()

args = parse_args()


def main(args):

    '''LOAD DATA'''
    print('Load dataset ...')
    train_dataset = MyData(args.train_data_path, "")
    test_dataset = MyData(args.test_data_path, "")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    '''LOG'''
    writer = SummaryWriter("logs_pointnet")

    '''FN LOADING'''
    print('Load Filtering-Network ...')
    FN = Ada_fil2_pointnet(args.FN_channel)
    FN.load_state_dict(args.FN_path)

    '''MODEL LOADING'''
    num_class = args.num_category
    classifier = PointNet(args.channel)


    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.decay_rate
    )


    criterion = nn.CrossEntropyLoss()

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    global_epoch = 0
    global_step = 0
    best_acc = 0

    for epoch in range(args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_loss = 0

        classifier.train()
        for data in train_dataloader:
            point_cloud, _, pcd_label = data

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
            loss = criterion(outputs, pcd_label)
            train_loss = train_loss + loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('Train Loss: %f' % train_loss)
        writer.add_scalar("test_loss", train_loss, epoch)

        classifier.eval()
        with torch.no_grad():

            test_loss = 0
            test_acc = 0

            for data in test_dataloader:
                point_cloud, _, pcd_label = data

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
                loss = criterion(outputs, pcd_label)
                test_loss = test_loss + loss
                acc = (outputs.argmax(1) == pcd_label).sum()
                test_acc = test_acc + acc

            test_acc = test_acc / len(test_dataset)

            if (test_acc >= best_acc):
                best_acc = test_acc
                best_epoch = epoch + 1

            print('Test Loss: %f, Class Accuracy: %f' % (test_loss, test_acc))
            print('Best Accuracy: %f' % best_acc)
            writer.add_scalar("test_loss", test_loss, epoch)

            if test_acc >= best_acc:
                print('Save model...')
                savepath = "models\\PFSP_PointNet.pth"
                print('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'test_loss': test_loss,
                    'model_state_dict': FN.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(classifier.state_dict(), savepath)
            global_epoch += 1

    print('End of training...')

