# coding: utf-8
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler

from utils import (get_normalized_adj, get_sp_adj_mat,
                   sparse_mx_to_torch_sparse_tensor)


# A class which is designed for loading various kinds of data
class DataLoader:
    max_time_num: int
    node2idx_dict: dict
    node_num: int
    has_cuda: bool

    def __init__(self, max_time_num, has_cuda=False):
        self.max_time_num = max_time_num
        self.has_cuda = has_cuda

    # get adjacent matrices for a graph list, this function supports Tensor type-based adj and sparse.coo type-based adj.
    def get_date_adj_list(self, origin_base_path, start_idx, duration, sep='\t', normalize=False, row_norm=False, add_eye=False, data_type='tensor'):
        assert data_type in ['tensor', 'matrix']
        date_dir_list = sorted(os.listdir(origin_base_path))
        # print('adj list: ', date_dir_list)
        date_adj_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            original_graph_path = os.path.join(origin_base_path, date_dir_list[i])
            spmat = get_sp_adj_mat(original_graph_path, sep=sep)
            # spmat = sp.coo_matrix((np.exp(alpha * spmat.data), (spmat.row, spmat.col)), shape=(self.node_num, self.node_num))
            if add_eye:
                spmat = spmat + sp.eye(spmat.shape[0])
            if normalize:
                spmat = get_normalized_adj(spmat, row_norm=row_norm)
            # data type
            if data_type == 'tensor':
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                date_adj_list.append(sptensor.cuda() if self.has_cuda else sptensor)
            else:  # data_type == matrix
                date_adj_list.append(spmat)
        # print(len(date_adj_list))
        return date_adj_list

    # get k-core sub-graph adjacent matrices for a graph list, it is a 2-layer nested list, outer layer for graph, inner layer for k-cores.
    # k-core subgraphs will be automatically normalized by 'renormalization trick'(add_eye=True)
    def get_core_adj_list(self, core_base_path, start_idx, duration, max_core=-1):
        core_adj_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            date_dir_path = os.path.join(core_base_path, '1-'+str(i))  
            f_list = sorted(os.listdir(date_dir_path))
            core_file_num = len(f_list)
            tmp_adj_list = []
            if max_core == -1:
                max_core = core_file_num
            f_list = f_list[:max_core]  # select 1 core to max core
            f_list = f_list[::-1]  # reverse order, max core, (max - 1) core, ..., 1 core

            # get k-core adjacent matrices at the i-th timestamp
            spmat_list = []
            for j, f_name in enumerate(f_list):
                spmat = sp.load_npz(os.path.join(date_dir_path, f_name))
                spmat_list.append(spmat)
                if j == 0:
                    spmat = spmat + sp.eye(spmat.shape[0])
                else:
                    delta = spmat - spmat_list[j - 1]    # reduce subsequent computation complexity and reduce memory cost!
                    if delta.sum() == 0:  # reduce computation complexity and memory cost!
                        continue
                # Normalization will reduce the self weight, hence affect its performance! So we omit normalization.
                sptensor = sparse_mx_to_torch_sparse_tensor(spmat)
                tmp_adj_list.append(sptensor.cuda() if self.has_cuda else sptensor)
            # print('time: ', i, 'core len: ', len(tmp_adj_list))
            core_adj_list.append(tmp_adj_list)
        return core_adj_list

    # get node co-occurrence pairs of random walk for a graph list, the node pair list is used for negative sampling
    def get_node_pair_list(self, walk_pair_base_path, start_idx, duration):
        walk_file_list = sorted(os.listdir(walk_pair_base_path))
        # print('walk file list: ', walk_file_list)
        node_pair_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            walk_file_path = os.path.join(walk_pair_base_path, '1-'+str(i)+'.npz')
            walk_spadj = sp.load_npz(walk_file_path)
            neighbor_arr = walk_spadj.tolil().rows #取出每个元素的行索引，每个节点的邻居节点
            node_pair_list.append(neighbor_arr)
        return node_pair_list

    # get node frequencies of random walk for a graph list, the node frequency list is used for negative sampling
    def get_node_freq_list(self, node_freq_base_path, start_idx, duration):
        freq_file_list = sorted(os.listdir(node_freq_base_path))
        # print('node freq list: ', freq_file_list)
        node_freq_list = []
        for i in range(start_idx, min(start_idx + duration, self.max_time_num)):
            freq_file_path = os.path.join(node_freq_base_path, '1-'+str(i)+'.json')
            # print(freq_file_path)
            with open(freq_file_path, 'r') as fp:
                node_freq_arr = json.load(fp)
                node_freq_list.append(node_freq_arr)
        return node_freq_list


    # load node features from file, or create one-hot node feature
    def get_feature_list(self,ip_num, data_path):
        sc=StandardScaler()
        file=data_path+'nodes_feats.pt'
        if os.path.exists(file):
            self.nodes_feats=torch.load(file)
        else:
            self.nodes_feats = torch.randn(ip_num,200) #从标准正态分布中随机采样
            torch.save(self.nodes_feats, file)
        self.nodes_feats=sc.fit_transform(self.nodes_feats)
        return torch.FloatTensor(self.nodes_feats), self.nodes_feats.shape[1]
