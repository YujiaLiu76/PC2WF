import MinkowskiEngine.utils as ME_utils

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
from scipy import spatial
from noise_gen_line_train import gen_line


class GenPatch:
    def __init__(self, point_path, patch_size=50, sigma=0.01, clip=0.02, train_val_test='test'):
        self.train_val_test = train_val_test
        self.clean_noise = 'noise'
        # train_val_test = 'train' if is_train else 'test'
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../abc_data/patches_{}_noise_sigma{}clip{}/{}'.format(patch_size, sigma, clip, train_val_test))

        assert os.path.exists(base_dir)

        self.save_root_path = os.path.join(base_dir, point_path.split('/')[-1])

        if os.path.exists(self.save_root_path.replace('.xyz', '.mini_line')):
            return
        if self.train_val_test == 'test' and os.path.exists(self.save_root_path.replace('.xyz', '.patch_index')):
            return
        
        self.point_path = point_path
        self.patch_size = patch_size
        self.pointcloud = np.loadtxt(point_path)
        
        # load pc after down sampling
        self.pointcloud_down = self.down_sample()
        self.build_graph()
        
        if self.train_val_test == 'test':
            seed_points = self.farthest_sample(k=10000)
            _, patch_index = self.gen_patch(self.pointcloud_down[seed_points], patch_per_seed=1)
            np.savetxt(self.save_root_path.replace('.xyz', '.patch_index'), patch_index, fmt='%d')
            return
        
        # load gt
        self.vert_gt, self.curve_gt, self.edge_gt, self.curve_line_gt = self.load_gt()
        
        max_vert = 1000

        # generate patch with vert
        if len(self.vert_gt) <= max_vert:
            seed_vert = self.vert_gt
        else:
            seed_vert = self.vert_gt[np.random.choice(list(range(len(self.vert_gt))), size=max_vert, replace=False)]
        patch_vert, patch_vert_index = self.gen_patch(seed_vert, patch_per_seed=2)
        # filter to guarantee patch_vert containing vert
        keep_idx = []
        patch_vertex_gt = []
        for idx, patch in enumerate(patch_vert):
            dist_vert = [np.linalg.norm(self.vert_gt-point, axis=1) for point in patch]
            patch_vertex_gt.append(self.vert_gt[np.argmin(dist_vert[np.argmin(np.min(dist_vert, axis=1))])])
            if np.min(dist_vert) <= 0.01:
                keep_idx.append(idx)
        patch_vert = patch_vert[keep_idx]
        patch_vert_index = patch_vert_index[keep_idx]
        patch_vertex_gt = np.array(patch_vertex_gt)[keep_idx]

        # generate other patch
        # fistly, patch with edge
        seed_edge = []
        for edge in self.edge_gt:
            for _ in range(2):
                point = edge[0:3]+np.random.random()*(edge[3:6]-edge[0:3])
                seed_edge.append(point)
        patch_edge, patch_edge_index = self.gen_patch(seed_edge, patch_per_seed=1)
        # filter to guarantee patch_edge containing no vert and curve
        keep_idx = []
        for idx, patch in enumerate(patch_edge):
            dist_vert = [np.min(np.linalg.norm(self.vert_gt-point, axis=1)) for point in patch]
            if min(dist_vert) >= 0.03:
                keep_idx.append(idx)
        patch_edge = patch_edge[keep_idx]
        patch_edge_index = patch_edge_index[keep_idx]
        # secondly, patch with random seed
        seed_random = self.pointcloud_down[np.random.randint(0, len(self.pointcloud_down), [len(patch_vert), 1])]
        patch_random, patch_random_index = self.gen_patch(seed_random, patch_per_seed=1)
        # filter to guarantee patch_random containing no vert and curve
        keep_idx = []
        for idx, patch in enumerate(patch_random):
            dist_vert = [np.min(np.linalg.norm(self.vert_gt-point, axis=1)) for point in patch] if len(self.vert_gt) > 0 else [1.0]
            if min(dist_vert) >= 0.03:
                keep_idx.append(idx)
        patch_random = patch_random[keep_idx]
        patch_random_index = patch_random_index[keep_idx]

        # save to file
        np.savetxt(self.save_root_path.replace('.xyz', '.vert_index'), patch_vert_index, fmt='%d')
        np.savetxt(self.save_root_path.replace('.xyz', '.vert_gt'), patch_vertex_gt, fmt='%.6f')
        patch_other_index = np.concatenate((patch_edge_index, patch_random_index), axis=0)
        np.savetxt(self.save_root_path.replace('.xyz', '.other_index'), patch_other_index, fmt='%d')

        # generate line samples
        gen_line(self.point_path, edge_gt=self.edge_gt, index=None, rotate_angle=None, random_rotate=False, patch_size=self.patch_size, clean_noise=self.clean_noise, sigma=sigma, clip=clip, train_val_test=train_val_test)

        
    def gen_patch(self, seed_points, patch_per_seed=1):
        if len(seed_points) == 0:
            return np.array([]), np.array([])
        '''generate patch from seed_points'''
        patch_list = []
        patch_index = []
        # first, randomly select new seeds from neighbors of seed_points
        dist, seed_idx = self.nbrs.query(seed_points, k=10)
        for i in range(len(seed_points)):
            select_seed_idx = seed_idx[i][ dist[i]<0.05 ]
            if len(select_seed_idx) == 0:
                continue
            shuffle(select_seed_idx)
            added_patch_per_seed = 0
            for seed in select_seed_idx:
                patch_idx = self.bfs_knn(seed, k=self.patch_size)
                patch_points = self.pointcloud_down[patch_idx]
                if self.train_val_test != 'test':
                    dists = [np.linalg.norm(patch_points-point, axis=1) for point in self.vert_gt]
                    num_vert_in_patch = sum(np.sort(dists)[:, 0]<0.010)
                    if num_vert_in_patch > 1:
                        continue
                if len(patch_idx) != self.patch_size:
                    continue
                patch_list.append(self.pointcloud_down[patch_idx])
                patch_index.append(patch_idx)
                added_patch_per_seed += 1
                if added_patch_per_seed >= patch_per_seed:
                    break

        return np.array(patch_list), np.array(patch_index)

    def build_graph(self):
        '''build graph based on point-to-point distance'''
        self.nbrs = spatial.cKDTree(self.pointcloud_down)
        dists,idxs = self.nbrs.query(self.pointcloud_down, k=15)
        self.graph=[]
        for item, dist in zip(idxs, dists):
            item = item[dist < 0.05] #use 0.03 for chair7 model; otherwise use 0.05
            self.graph.append(set(item))
    
    def bfs_knn(self, seed=0, k=10):
        '''bfs query'''
        q = collections.deque()
        visited = set()
        result = []
        q.append(seed)
        while len(visited) < k and q:
            vertex = q.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                if len(q) < k * 5:
                    q.extend(self.graph[vertex] - visited)
        return np.array(result)

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)
    
    def farthest_sample(self, k):
        farthest_pts_index = np.zeros((k, ), dtype=np.int32)
        farthest_pts = np.zeros((k, 3))
        farthest_pts_index[0] = np.random.randint(len(self.pointcloud_down))
        farthest_pts[0] = self.pointcloud_down[farthest_pts_index[0]]
        distances = self.calc_distances(farthest_pts[0], self.pointcloud_down)
        for i in range(1, k):
            farthest_pts_index[i] = np.argmax(distances)
            farthest_pts[i] = self.pointcloud_down[farthest_pts_index[i]]
            distances = np.minimum(distances, self.calc_distances(farthest_pts[i], self.pointcloud_down))
        return np.array(farthest_pts_index)
        
    def load_gt(self):
        '''load ground truth file'''
        gt_path = self.point_path.replace('/xyz/', '/gt/').replace('.xyz', '.obj')
        all_gt_path = [
            gt_path.replace('train', 'train').replace('validation', 'train').replace('test', 'train'),
            gt_path.replace('train', 'validation').replace('validation', 'validation').replace('test', 'validation'),
            gt_path.replace('train', 'test').replace('validation', 'test').replace('test', 'test')]
        for gt_path in all_gt_path:
            if os.path.exists(gt_path):
                break
        with open(gt_path, 'r') as gt_f:
            vert_gt = []
            edge_gt = []
            item_gt = gt_f.readline()
            while item_gt:
                item_gt = item_gt.strip().split()
                if item_gt[0] == 'v':
                    vert_gt.append(item_gt[1:])
                elif item_gt[0] == 'l':
                    e1, e2 = int(item_gt[1])-1, int(item_gt[2])-1
                    edge = vert_gt[e1].copy()
                    edge.extend(vert_gt[e2])
                    edge_gt.append(edge)
                item_gt = gt_f.readline()
        # choose verteices with lines
        vert_with_line_index = []
        for e in edge_gt:
            if e[0:3] not in vert_with_line_index:
                vert_with_line_index.append(e[0:3])
            if e[3:6] not in vert_with_line_index:
                vert_with_line_index.append(e[3:6])
        
        vert_gt = np.array(vert_with_line_index).astype(np.float32)
        curve_gt = np.array([])
        edge_gt = np.array(edge_gt).astype(np.float32)
        curve_line_gt = np.array([])
        return vert_gt, curve_gt, edge_gt, curve_line_gt

    def down_sample(self):
        '''down sample pointcloud using FCGF '''
        feats = []
        feats.append(np.ones((self.pointcloud.shape[0], 1)))
        feats = np.hstack(feats)
        coords = np.floor(self.pointcloud / 0.0075)
        inds = ME_utils.sparse_quantize(coords, return_index=True)
        coords = coords[inds]
        coords = np.hstack([coords, np.zeros((len(coords), 1))])
        pointcloud_down = self.pointcloud[inds]
        feats = feats[inds]

        point_down_path = self.save_root_path.replace('.xyz', '.down')
        np.savetxt(self.save_root_path.replace('.xyz', '.down'), pointcloud_down, fmt='%0.6f')
        pointcloud_down = np.loadtxt(point_down_path)
        np.savetxt(self.save_root_path.replace('.xyz', '.feats'), feats)
        np.savetxt(self.save_root_path.replace('.xyz', '.coords'), coords)
        return pointcloud_down

def multi_process(f):
    GenPatch(f, patch_size=patch_size, sigma=sigma, clip=clip, train_val_test=dirname)

if __name__ == '__main__':
    sigma = 0.01
    clip = 0.01
    patch_size = 50

    for dirname in ['train', 'test', 'validation']:
        file_list = glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../abc_data/noise_sigma{}clip{}/xyz/{}/*.xyz'.format(sigma, clip, dirname)))
        file_list.sort()

        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../abc_data/patches_{}_noise_sigma{}clip{}/{}'.format(patch_size, sigma, clip, dirname))
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        os.makedirs(base_dir)

        pool = multiprocessing.Pool(7)
        pool.map(multi_process, file_list)
