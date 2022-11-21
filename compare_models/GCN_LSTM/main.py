import argparse
import os
from operator import imod
from statistics import mode
from time import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_process import (Cic2018_Dataset, Cic_Dataset, UNSW_Dataset,
                          USTC_Dataset)
from gcn_lstm import GCN_LSTM
from util import Classifier

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)

def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feats_per_node', default=200, type=int)
    parser.add_argument('--layer_1_feats', default=128, type=int)
    parser.add_argument('--layer_2_feats', default=64, type=int)
    parser.add_argument('--num_gcn_layers', default=2, type=int)
    parser.add_argument('--lstm_feats', default=32, type=int)
    parser.add_argument('--num_lstm_layers', default=1, type=int)
    parser.add_argument('--cls_in_feats', default=64, type=int)
    parser.add_argument('--k', default=200, type=int)
    parser.add_argument('--ck_path', default='model.pt', type=str)
    parser.add_argument('--graph_embs_path', default='graph_embs.pt',type=str)
    return parser.parse_args()
    
def neg_sample(adj, neg_num):
    weights = torch.sum(adj.to_dense(),1)
    hat_a=adj.to_dense().cpu()
    neg_indices=torch.where(hat_a==0)
    dst = weights.multinomial(neg_num, replacement=True) # 多项式分布采样，第一个参数表示采样的数目。采样的是self.weights的位置，self.weights的每个值表示采样到该元素的权重，
    return dst

def get_edge_embs(out, adj):
    edge_indices = adj._indices()
    neg_dst=neg_sample(adj, len(edge_indices[0]))
    
    pos_edges = torch.cat([out[edge_indices[0]], out[edge_indices[1]]], dim=1)
    neg_edges = torch.cat([out[edge_indices[0]], out[neg_dst]], dim=1)
    edge_embs = torch.cat([pos_edges, neg_edges])
    label = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).long()
    return edge_embs, label

def train(args, seq_len,graphs, node_feats, device, train_len, epochs):
    model = GCN_LSTM(args, activation = torch.nn.RReLU(), device=device)
    cls=Classifier(args)
    model.to(device)
    cls.to(device)
    gcn_opt=torch.optim.Adam(model.parameters(), lr=0.0001)
    cls_opt=torch.optim.Adam(cls.parameters(), lr=0.0001)
    loss_gcn=nn.BCEWithLogitsLoss()
    model.train()
    for i in range(epochs):
        el=0
        for idx in tqdm(range(train_len[0],train_len[1]-seq_len)):
            gcn_opt.zero_grad()
            cls_opt.zero_grad()
            gs=[graphs[i] for i in range(idx,idx+seq_len)]
            adjs=[g['adj'] for g in gs]
            mask=[g['node_idx'] for g in gs]
            out=model(adjs, node_feats, mask)
            edge_embs, labels=get_edge_embs(out, adjs[-1])
            pred=cls(edge_embs)
            labels=labels.unsqueeze(1)
            labels=torch.cat([1-labels, labels], 1)
            loss=loss_gcn(pred.cpu(), labels.float())
            loss.backward()
            el+=loss.item()
            gcn_opt.step()
            cls_opt.step()
        print('epoch:{:.4f}, loss:{:.4f}'.format(i, el/idx))
        torch.save({
            'gcn_model':model,
            'cls':cls.state_dict(),
        }, args.ck_path)
    return model
    
def predict(args, graphs, node_feats):
    model = GCN_LSTM(args, activation = torch.nn.RReLU(), device=device)
    ck=torch.load(args.ck_path)
    model=ck['gcn_model']
    model.to(device)
    graph_embs=[]
    model.eval()
    for idx in tqdm(range(len(graphs)-seq_len)):
        gs=[graphs[i] for i in range(idx,idx+seq_len)]
        adjs=[g['adj'] for g in gs]
        mask=[g['node_idx'] for g in gs]
        out=model(adjs, node_feats, mask)
        edge_indices = adjs[-1]._indices()
        pos_edges = torch.cat([out[edge_indices[0]], out[edge_indices[1]]], dim=1)
        graph_embs.append(torch.mean(pos_edges.detach().cpu(), 0))
    torch.save(torch.stack(graph_embs), args.graph_embs_path)

if __name__ == '__main__':
    args=parse()
    seq_len=5
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    t0=time()
    cd =Cic_Dataset()
    graphs=cd.gen_graphs()
    node_feats=cd.gen_node_feats()
    t1=time()
    print('预训练时间',t1-t0)

    # train(args, seq_len, graphs=graphs, node_feats=node_feats, device=device, train_len=[0, 529], epochs=1) #cic[0,529], unsw[200:600], ustc[10:100]
    # t2=time()
    # print('训练时间', t2-t1)
    # predict(args, graphs=graphs, node_feats=node_feats)
    # t3=time()
    # print('测试时间', t3-t2)
