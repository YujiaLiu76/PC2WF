import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.append('..')
from train_end2end import patchNet, vertexNet, lineNet, PointcloudDataset
from tqdm import tqdm
import random
import shutil
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import MinkowskiEngine.utils as ME_utils
from model.resunet import ResUNetBN2C
import numpy as np
from itertools import combinations
from glob import glob


def predict(test_file_path, backbone_pth, patch_pth, vertex_pth, line_pth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load backbone_net
    backbone_net = ResUNetBN2C(1, 32, normalize_feature=True, conv1_kernel_size=7, D=3)
    backbone_net.load_state_dict(torch.load(backbone_pth))
    backbone_net = backbone_net.to(device)
    backbone_net.eval()
    # load patch_net
    patch_net = patchNet()
    patch_net.load_state_dict(torch.load(patch_pth))
    patch_net = patch_net.to(device)
    patch_net.eval()
    # load vertex_net
    vertex_net = vertexNet()
    vertex_net.load_state_dict(torch.load(vertex_pth))
    vertex_net = vertex_net.to(device)
    vertex_net.eval()
    # load line_net
    line_net = lineNet()
    line_net.load_state_dict(torch.load(line_pth))
    line_net = line_net.to(device)
    line_net.eval()

    '''first, load test_data and feed into backbone_net'''
    pc_down = np.loadtxt(test_file_path, dtype=np.float32) # Ndx3
    feats = np.expand_dims(np.loadtxt(test_file_path.replace('.down', '.feats'), dtype=np.float32), 1) # Ndx1
    coords = np.loadtxt(test_file_path.replace('.down', '.coords'), dtype=np.float32) # Ndx3
    patch_index = np.loadtxt(test_file_path.replace('.down', '.patch_index'), dtype=np.int32) # Ndx3

    
    stensor = ME.SparseTensor(torch.from_numpy(feats).float(), coords=torch.from_numpy(coords)).to(device)
    features = backbone_net(stensor).F

    '''second, feed into patch_net to find patches with vertex'''
    patch_features = []
    patch_coords = []
    pc_down = torch.from_numpy(pc_down)
    patch_index = torch.from_numpy(patch_index).long()
    if len(patch_index.shape) == 1:
        patch_index = patch_index.unsqueeze(-1)
    for index in patch_index:
        patch_features.append(features[index].unsqueeze(0))
        curr_coord = pc_down[index.long()]
        patch_coords.append(curr_coord)
    
    batch_features = torch.cat(patch_features, 0).cuda()
    batch_coords = torch.stack(patch_coords, 0).cuda()
    batch_input_patch = torch.cat([batch_coords, batch_features], 2).transpose(1, 2)
    batch_output_patch = patch_net(batch_input_patch)
    
    # select pacthes with positive vertex
    predicted_patch_index = torch.sigmoid(batch_output_patch.squeeze())
    batch_input_vertex = []
    batch_input_vertex_prob = []
    batch_coords_center_vertex = []
    batch_coords_lwh_vertex = []
    for i, predicted_index in enumerate(predicted_patch_index):
      if predicted_index > 0.85:
        batch_input_vertex.append(batch_input_patch[i])
        batch_input_vertex_prob.append(predicted_index)
    batch_input_vertex = torch.stack(batch_input_vertex, 0)

    '''third, feed into vertex_net to produce new vertex'''
    batch_output_vertex = vertex_net(batch_input_vertex)
    batch_output_vertex_coord = batch_output_vertex
    predicted_vertex_list = batch_output_vertex_coord.detach().cpu().numpy()

    # NMS to select vertex
    nms_threshhold = 0.01
    dropped_vertex_index = []
    for i in range(len(predicted_vertex_list)):
        if i in dropped_vertex_index:
            continue
        dist_all = np.linalg.norm(predicted_vertex_list-predicted_vertex_list[i], axis=1)
        same_region_indexes = (dist_all < nms_threshhold).nonzero()
        for same_region_i in same_region_indexes[0]:
            if same_region_i == i:
                continue
            if batch_input_vertex_prob[same_region_i] <= batch_input_vertex_prob[i]:
                dropped_vertex_index.append(same_region_i)
            else:
                dropped_vertex_index.append(i)
    selected_vertex_index = [i for i in range(len(predicted_vertex_list)) if i not in dropped_vertex_index]
    batch_output_vertex_coord = batch_output_vertex_coord[selected_vertex_index]
    batch_input_vertex_prob = np.array(batch_input_vertex_prob)[selected_vertex_index]

    predicted_vertex_list = batch_output_vertex_coord.detach().cpu().numpy()
    predicted_vertex_probs = np.array(batch_input_vertex_prob)

    predicted_vertex_features = []
    for coord in batch_output_vertex_coord.detach().cpu():
        pred_vertex_index = torch.argmin(torch.norm(pc_down - coord, dim=1))
        predicted_vertex_features.append(features[pred_vertex_index])


    '''forth, feed into line_net to predict lines'''
    point_num_in_line = 30
    input_line_features = predicted_vertex_features
    pc_down = pc_down.to(device)
    batch_input_line = []
    batch_index_line = []
    batch_index_dist = []
    for i1, e1 in enumerate(batch_output_vertex_coord):
        for i2, e2 in enumerate(batch_output_vertex_coord):
            if i1 >= i2:
                continue
            mid_point_dist = torch.min(torch.norm(pc_down - (e1 + e2) / 2.0, dim=1))
            if mid_point_dist >= 0.03:
                continue
            tmp_input_line = [input_line_features[i1]]
            tmp_input_dist = 0
            valid_line = True
            for inter_point in range(1, point_num_in_line+1):
                inter_point_coord = (float(inter_point)/(point_num_in_line+1)*e1 + (1-float(inter_point)/(point_num_in_line+1))*e2)
                inter_point_dist = torch.norm(pc_down - inter_point_coord, dim=1)
                if torch.min(inter_point_dist) >= 0.03:
                    valid_line = False
                    break
                tmp_input_dist += torch.min(inter_point_dist).cpu().item()
                inter_point_index = torch.argmin(inter_point_dist)
                tmp_input_line.append(features[inter_point_index])
            if not valid_line:
                continue
            tmp_input_line.append(input_line_features[i2])
            batch_input_line.append(torch.stack(tmp_input_line))
            batch_index_line.append([i1+1, i2+1])
            batch_index_dist.append(tmp_input_dist/point_num_in_line)
    
    batch_input_line = torch.stack(batch_input_line).transpose(1, 2)
    batch_output_line = line_net(batch_input_line)
    predicted_line_index = torch.sigmoid(batch_output_line.squeeze())
    predicted_line_list = []
    predicted_line_probs = []
    if len(predicted_line_index.shape) == 0:
        predicted_line_index = predicted_line_index.unsqueeze(0)
    for i, predicted_index in enumerate(predicted_line_index):
        if predicted_index > 0.5:
            predicted_line_list.append(batch_index_line[i])
            predicted_line_probs.append(predicted_index)
    return np.array(predicted_vertex_list), np.array(predicted_vertex_probs), np.array(predicted_line_list), np.array(predicted_line_probs)
    


if __name__ == "__main__":
    sigma = 0.01
    clip = 0.01
    patch_size = 50

    save_to_folder = os.path.join(curr_dir, 'run_test_result', f'patch{patch_size}sigma{sigma}clip{clip}')
    if not os.path.exists(save_to_folder):
        os.makedirs(save_to_folder)
    print(os.path.join(curr_dir, f'../abc_data/patches_{patch_size}_noise_sigma{sigma}clip{clip}/test/*.down'))
    test_file_list = glob(os.path.join(curr_dir, f'../abc_data/patches_{patch_size}_noise_sigma{sigma}clip{clip}/test/*.down'))
    test_file_list.sort()

    for test_file in tqdm(test_file_list):
        if os.path.exists(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_line.txt'))):
            continue
        predicted_vertex_list, predicted_vertex_probs, predicted_line_list, predicted_line_probs = predict(
            test_file,
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}_pretrained/backbone_patchSize{patch_size}_miniBatch512_nmsTh0.01_linePosTh0.01_lineNegTh0.05_lossweightP1.0V50.0L1.0_Val.pth'),
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}_pretrained/patchnet_patchSize{patch_size}_miniBatch512_nmsTh0.01_linePosTh0.01_lineNegTh0.05_lossweightP1.0V50.0L1.0_Val.pth'),
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}_pretrained/vertexnet_patchSize{patch_size}_miniBatch512_nmsTh0.01_linePosTh0.01_lineNegTh0.05_lossweightP1.0V50.0L1.0_Val.pth'),
            os.path.join(curr_dir, f'../checkpoint_sigma{sigma}clip{clip}_pretrained/linenet_patchSize{patch_size}_miniBatch512_nmsTh0.01_linePosTh0.01_lineNegTh0.05_lossweightP1.0V50.0L1.0_Val.pth'),
        )

        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_vertex.txt')), predicted_vertex_list)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_vprobs.txt')), predicted_vertex_probs)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_line.txt')), predicted_line_list)
        np.savetxt(os.path.join(save_to_folder, test_file.split('/')[-1].replace('.down', '_lprobs.txt')), predicted_line_probs)
