#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import argparse
import importlib
import time

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.join('/home/liang/下载/votenet/utils'))
sys.path.append(os.path.join('/home/liang/下载/votenet/models'))
sys.path.append(os.path.join('/media/liang/文档2/sunrgbd'))
from pc_util import random_sampling, read_ply
from ap_helper import parse_predictions

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='sunrgbd', help='Dataset: sunrgbd or scannet [default: sunrgbd]')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
#parser.add_argument('--num_point', type=int, default=2, help='Point Number [default: 2]')
FLAGS = parser.parse_args()

def preprocess_point_cloud(point_cloud):
    ''' Prepare the numpy point cloud (N,3) for forward pass '''
    point_cloud = point_cloud[:,0:3] # do not use color for now
    floor_height = np.percentile(point_cloud[:,2],0.99)
    height = point_cloud[:,2] - floor_height
    point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7) 相对地面高度
    point_cloud = random_sampling(point_cloud, FLAGS.num_point)
    pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
    return pc

def hello(point):
    '''if a==1:
        print("hello world!")'''
    if point.all():
        from sunrgbd_detection_dataset import DC  # dataset config
        checkpoint_path = os.path.join('/home/ylr/下载/votenet/demo_files/pretrained_votenet_on_sunrgbd.tar')
        #print(point)
        point_cloud = np.array(point)
        pc = preprocess_point_cloud(point_cloud)
        b = np.ones([2, 3], dtype=float)
        c = np.zeros([1, 3], dtype=float)
        a = [[(2,b),(3,c)]]
        print('Loaded point cloud data')
        print(a)
        return a

