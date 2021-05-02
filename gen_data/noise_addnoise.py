import collections
import json
import multiprocessing
import os
import shutil
import sys
import time
from glob import glob
from random import shuffle
from subprocess import Popen
import h5py
import numpy as np
import random
from math import ceil
from tqdm import tqdm


def add_noise(clean_point_path, dst_noisy_file, ori_gt_f, dst_noise_gt_f, sigma=0.01, clip=0.02):
    clean_pts = np.loadtxt(clean_point_path)
    row, col = clean_pts.shape
    noise = np.clip(sigma * np.random.randn(row, col), -1*clip, clip)
    noisy_pts = clean_pts + noise
    max_x, min_x = noisy_pts[:,0].max(), noisy_pts[:,0].min()
    max_y, min_y = noisy_pts[:,1].max(), noisy_pts[:,1].min()
    max_z, min_z = noisy_pts[:,2].max(), noisy_pts[:,2].min()

    norm_noisy_pts = noisy_pts[:]
    norm_noisy_pts[:,0] = (norm_noisy_pts[:,0] - min_x) / (max_x - min_x)
    norm_noisy_pts[:,1] = (norm_noisy_pts[:,1] - min_y) / (max_y - min_y)
    norm_noisy_pts[:,2] = (norm_noisy_pts[:,2] - min_z) / (max_z - min_z)

    np.savetxt(dst_noisy_file, norm_noisy_pts, fmt='%.4f')

    with open(ori_gt_f, 'r') as ori_f:
        v = []
        l = []
        for line in ori_f.readlines():
            if line[0] == 'v' and line[1] == ' ':
                line = line.split()
                v0, v1, v2 = float(line[1]), float(line[2]), float(line[3])
                v.append([v0, v1, v2])
            if line[0] == 'l' and line[1] == ' ':
                l.append(line)
    with open(dst_noise_gt_f, 'w') as f:
        v = np.array(v)
        v[:,0] = (v[:,0] - min_x) / (max_x - min_x)
        v[:,1] = (v[:,1] - min_y) / (max_y - min_y)
        v[:,2] = (v[:,2] - min_z) / (max_z - min_z)
        for _v in v:
            f.write('v {:.4f} {:.4f} {:.4f}\n'.format(_v[0], _v[1], _v[2]))
        for _l in l:
            f.write(_l)
        
        f.write("# max_x: {}, min_x: {}\n".format(max_x, min_x))
        f.write("# max_y: {}, min_y: {}\n".format(max_y, min_y))
        f.write("# max_z: {}, min_z: {}\n".format(max_z, min_z))

def add_noise_one_file(clean_point_path):
    dst_noisy_file = os.path.join(dst_noise_dir, clean_point_path.split('/')[-1])
    ori_gt_f = clean_point_path.replace('/xyz/', '/gt/').replace('.xyz', '.obj')
    dst_noise_gt_f = os.path.join(dst_noise_gt_dir, ori_gt_f.split('/')[-1])
    add_noise(clean_point_path, dst_noisy_file, ori_gt_f, dst_noise_gt_f, sigma=sigma, clip=clip)


if __name__ == '__main__':
    sigma = 0.01
    clip = 0.01

    dst_noise_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../abc_data/noise_sigma{}clip{}'.format(sigma, clip))
    if os.path.exists(dst_noise_root):
        shutil.rmtree(dst_noise_root)

    for dir_name in ['train', 'test', 'validation']:
        file_list = glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../abc_data/clean/xyz/{}/*.xyz'.format(dir_name)))
        file_list.sort()
        
        dst_noise_dir = os.path.join(dst_noise_root, 'xyz/{}'.format(dir_name))
        dst_noise_gt_dir = os.path.join(dst_noise_root, 'gt/{}'.format(dir_name))
        os.makedirs(dst_noise_dir)
        os.makedirs(dst_noise_gt_dir)

        pool = multiprocessing.Pool(7)
        pool.map(add_noise_one_file, file_list)
        
