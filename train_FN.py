#  -*- coding:utf-8 -*-
"""
Author: Vicktor
Date: Jan 2024
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse

from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Filtering_Network import *

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training')
    parser.add_argument('--model', default='Ada_fil2_pointnet', help='model name [default: FN]')
    parser.add_argument('--channel', default=2, type=int,  help='channels of training data')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--train_data_path', default='Data\\FN\\train', help='the path of train data')
    parser.add_argument('--test_data_path', default='Data\\FN\\test', help='the path of test data')

    return parser.parse_args()

args = parse_args()

def test(model, loader):
    test_loss = 0
    f_n = model.eval()
    loss_fn = nn.MSELoss()

    for j, (density, par) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            density, par = density.cuda(), par.cuda()
            loss_fn = loss_fn.cuda()

        density = density.permute(0, 2, 1)
        pred = f_n(density)
        loss = loss_fn(pred, par)

        test_loss = test_loss + loss

    return test_loss


def main(args):

    '''LOAD DATA'''
    print('Load dataset ...')
    train_dataset = MyData(args.train_data_path, "")
    test_dataset = MyData(args.test_data_path, "")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''LOG'''
    writer = SummaryWriter("logs_train")

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)

    FN = model.get_model(channel=args.channel)
    criterion = nn.MSELoss()


    if not args.use_cpu:
        FN = FN.cuda()
        criterion = criterion.cuda()

    if args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(
            FN.parameters(),
            lr=args.learning_rate,
            eps=1e-3,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(FN.parameters(), lr=0.01, momentum=0.9)

    global_epoch = 0
    global_step = 0
    best_test_loss = 0

    '''TRANING'''
    print('Start training...')
    for epoch in range(args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        train_loss = 0

        FN.train()
        for batch_id, (inputs, par) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataset), smoothing=0.9):
            optimizer.zero_grad()

            inputs = inputs.permute(0, 2, 1)

            if not args.use_cpu:
                inputs, par = inputs.cuda(), par.cuda()

            pred = FN(inputs)
            loss = criterion(pred, par)

            train_loss = train_loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

        print('Train Loss: %f' % train_loss)
        writer.add_scalar("test_loss", train_loss, epoch)

        with torch.no_grad():
            test_loss = test(FN.eval(), test_dataloader)

            if (test_loss <= best_test_loss) or epoch == 0:
                best_test_loss = test_loss
                best_epoch = epoch + 1

            print('Test Loss: %f' % test_loss)
            print('Best Test Loss: %f' % best_test_loss)
            writer.add_scalar("test_loss", test_loss, epoch)

            if (test_loss <= best_test_loss) and epoch != 0:
                print('Save model...')
                savepath = "models\\adaptive_filter_pointnet_fil2.pth"
                print('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'test_loss': test_loss,
                    'model_state_dict': FN.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    print('End of training...')


