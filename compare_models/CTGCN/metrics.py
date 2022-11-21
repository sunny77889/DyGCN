# coding: utf-8
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable

from utils import accuracy


# Unsupervised loss classes
class NegativeSamplingLoss(nn.Module):
    def __init__(self, node_pair_list, neg_freq_list, neg_num=20):
        super(NegativeSamplingLoss, self).__init__()
        self.node_pair_list = node_pair_list
        self.neg_freq_list = neg_freq_list
        self.neg_sample_num = neg_num

    # Negative sampling loss used for unsupervised learning to preserve local connective proximity
    def forward(self, node_embedding, batch_indices, indices):
        node_embedding = [node_embedding] if not isinstance(node_embedding, list) and len(node_embedding.size()) == 2 else node_embedding
        bce_loss = nn.BCEWithLogitsLoss()
        neighbor_loss = Variable(torch.tensor([0.], device=node_embedding.device), requires_grad=True)
        timestamp_num = len(node_embedding)
        for i in range(timestamp_num):
            embedding_mat = node_embedding[i]   # tensor
            node_pairs = self.node_pair_list[i]  # list
            node_freqs = self.neg_freq_list[i]  # tensor
            sample_num, node_indices, pos_indices, neg_indices = self.__get_node_indices(batch_indices[i], node_pairs, node_freqs, node_embedding.device)
            if sample_num == 0:
                continue
            embedding_mat=embedding_mat[indices[i]]
            pos_score = torch.sum(embedding_mat[node_indices].mul(embedding_mat[pos_indices]), dim=1) #计算余弦相似性
            neg_score = torch.sum(embedding_mat[node_indices].matmul(torch.transpose(embedding_mat[neg_indices], 1, 0)), dim=1) #每个源节点对所有的不相连节点计算相似性并相加
            
            pos_loss = bce_loss(pos_score, torch.ones_like(pos_score))
            neg_loss = bce_loss(neg_score, torch.zeros_like(neg_score))
            loss_val = 3*pos_loss + neg_loss
            neighbor_loss = neighbor_loss + loss_val
        return neighbor_loss

    def __get_node_indices(self, batch_indices, node_pairs: np.ndarray, node_freqs: np.ndarray, device):
        dtype = batch_indices.dtype
        node_indices, pos_indices, neg_indices = [], [], []
        random.seed()

        sample_num = 0
        for node_idx in batch_indices:
            # print('node pair type: ', type(node_pairs))
            neighbor_num = len(node_pairs[node_idx])
            if neighbor_num <= self.neg_sample_num:
                pos_indices += node_pairs[node_idx]
                real_num = neighbor_num
            else:
                pos_indices += random.sample(node_pairs[node_idx], self.neg_sample_num)
                real_num = self.neg_sample_num
            node_indices += [node_idx] * real_num
            sample_num += real_num
        if sample_num == 0:
            return sample_num, None, None, None
        # print(len(node_freqs), len(pos_indices), self.neg_sample_num)
        neg_indices += random.sample(node_freqs, len(node_freqs)//2)
        

        node_indices = torch.tensor(node_indices, dtype=dtype, device=device)
        pos_indices = torch.tensor(pos_indices, dtype=dtype, device=device)
        neg_indices = torch.tensor(neg_indices, dtype=dtype, device=device)
        return sample_num, node_indices, pos_indices, neg_indices


# Reconstruction loss used for k-core based structure preserving methods(CGCN-S and CTGCN-S)
class ReconstructionLoss(nn.Module):

    def __init__(self):
        super(ReconstructionLoss, self).__init__()

    def forward(self, input_list):
        assert len(input_list) == 3
        node_embedding, structure_embedding, batch_indices = input_list[0], input_list[1], input_list[2]
        node_embedding = [node_embedding] if not isinstance(node_embedding, list) and len(node_embedding.size()) == 2 else node_embedding
        structure_embedding = [structure_embedding] if not isinstance(structure_embedding, list) and len(structure_embedding.size()) == 2 else structure_embedding
        return self.__reconstruction_loss(node_embedding, structure_embedding, batch_indices)

    # Reconstruction loss used for unsupervised learning to preserve local connective proximity
    @staticmethod
    def __reconstruction_loss(node_embedding, structure_embedding, batch_indices=None):
        mse_loss = nn.MSELoss()
        structure_loss = 0
        timestamp_num = len(node_embedding)
        for i in range(timestamp_num):
            embedding_mat = node_embedding[i]
            structure_mat = structure_embedding[i]

            if batch_indices is not None:
                structure_loss = structure_loss + mse_loss(structure_mat[batch_indices], embedding_mat[batch_indices])
            else:
                structure_loss = structure_loss + mse_loss(structure_mat, embedding_mat)
        return structure_loss
