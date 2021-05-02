import collections
import json
import multiprocessing
import os
import shutil
import sys
import time
from glob import glob
from multiprocessing.dummy import Pool as ThreadPool
from random import shuffle
from subprocess import Popen
import h5py
import numpy as np
from tqdm import tqdm
import MinkowskiEngine.utils as ME_utils


def gen_line(point_path, edge_gt, index=0, rotate_angle=None, random_rotate=True, patch_size=20, clean_noise='noise', sigma=0.01, clip=0.02, train_val_test='test'):
    train_val_test = train_val_test
    if random_rotate:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../abc_data/patches_{}_noise_sigma{}clip{}_rotate/{}'.format(patch_size, sigma, clip, train_val_test))
    else:
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../abc_data/patches_{}_noise_sigma{}clip{}/{}'.format(patch_size, sigma, clip, train_val_test))

    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    
    if index is None:
        save_root_path = os.path.join(base_dir, point_path.split('/')[-1])
    else:
        save_root_path = os.path.join(base_dir, point_path.split('/')[-1]).replace('.xyz', '_{}.xyz'.format(index))

    point_down_path = save_root_path.replace('.xyz', '.down')

    '''begin: down_sample'''
    pointcloud_down = np.loadtxt(point_down_path)
    '''end: down_sample'''


    '''begin: line samples'''
    edge_points = edge_gt
    if rotate_angle is not None:
        angles = rotate_angle
        Rx = np.array([[1, 0, 0],
                        [0, np.cos(angles[0]), -np.sin(angles[0])],
                        [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                        [0, 1, 0],
                        [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                        [np.sin(angles[2]), np.cos(angles[2]), 0],
                        [0, 0, 1]])
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

        edge_points[:,:3] = np.dot(edge_points[:,:3], rotation_matrix)
        edge_points[:,3:] = np.dot(edge_points[:,3:], rotation_matrix)

    point_num_in_line = 30

    # positive samples: all edges
    positive_edge_index = []
    positive_edge_end_point_index = []
    for edge in edge_points:
        e1 = np.argmin(np.linalg.norm(pointcloud_down-edge[:3], axis=1))
        e2 = np.argmin(np.linalg.norm(pointcloud_down-edge[3:], axis=1))
        # mid_point_index = np.argmin(np.linalg.norm(pointcloud_down - (pointcloud_down[e1] + pointcloud_down[e2]) / 2.0, axis=1))
        inter_point_list_positive = []
        valid_line = True
        for inter_point in range(1, point_num_in_line+1):
            inter_point_dist = np.linalg.norm(pointcloud_down - ((float(inter_point)/(point_num_in_line+1))*pointcloud_down[e1] + (1-float(inter_point)/(point_num_in_line+1))*pointcloud_down[e2]), axis=1)
            if np.min(inter_point_dist) > 0.030:
                valid_line = False
                break
            inter_point_index = np.argmin(inter_point_dist)
            inter_point_list_positive.append(inter_point_index)
        if not valid_line:
            continue
        positive_edge_index.append(e1)
        positive_edge_index.extend(inter_point_list_positive)
        positive_edge_index.append(e2)
        positive_edge_index.append(1)
        positive_edge_end_point_index.append([e1, e2])
        # positive_edge_index.append([e1, e2, mid_point_index, 1])       
    positive_edge_index = np.array(positive_edge_index)
    positive_edge_index = np.reshape(positive_edge_index, (-1, point_num_in_line+3))

    # negative samples: vertices in the same face but no edge
    negative_edge_index = []
    negative_edge_end_point_index = []
    for edge in positive_edge_index:
        e1, e2 = edge[0], edge[-2]
        # e2_edges_0 = positive_edge_index[positive_edge_index[:,0] == e2][:,1]
        e2_edges_0 = positive_edge_index[positive_edge_index[:,0] == e2][:,-2] # diagonal
        e2_edges_1 = positive_edge_index[positive_edge_index[:,0] == e2][:,5] # offset
        e2_edges = np.concatenate((e2_edges_0, e2_edges_1))
        for e2_v in e2_edges:
            if ([e1, e2_v] in negative_edge_end_point_index) or ([e2_v, e1] in negative_edge_end_point_index):
                continue
            if ([e1, e2_v] in positive_edge_end_point_index) or ([e2_v, e1] in positive_edge_end_point_index) or (e1 == e2_v):
                continue
            if np.linalg.norm(e2 - e2_v) <= 0.03:
                continue
            mid_point_dist = np.min(np.linalg.norm(pointcloud_down - (pointcloud_down[e1] + pointcloud_down[e2_v]) / 2.0, axis=1))
            if mid_point_dist >= 0.030:
                continue
            inter_point_list_negative = []
            valid_line = True
            for inter_point in range(1, point_num_in_line+1):
                inter_point_dist = np.min(np.linalg.norm(pointcloud_down - (pointcloud_down[e1] + pointcloud_down[e2_v]) / 2.0, axis=1))
                if inter_point_dist >= 0.030:
                    valid_line = False
                    break
                inter_point_index = np.argmin(np.linalg.norm(pointcloud_down - (float(inter_point)/(point_num_in_line+1)*pointcloud_down[e1] + (1-float(inter_point)/(point_num_in_line+1))*pointcloud_down[e2_v]), axis=1))
                inter_point_list_negative.append(inter_point_index)
            if not valid_line:
                continue
            negative_edge_index.append(e1)
            negative_edge_index.extend(inter_point_list_negative)
            negative_edge_index.append(e2_v)
            negative_edge_index.append(0)
            negative_edge_end_point_index.append([e1, e2_v])
    negative_edge_index = np.array(negative_edge_index)
    negative_edge_index = np.reshape(negative_edge_index, (-1, point_num_in_line+3))


    np.savetxt(save_root_path.replace('.xyz', '.mini_line'), np.concatenate((positive_edge_index, negative_edge_index)))

    '''end: line samples'''

