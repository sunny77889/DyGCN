import argparse
import os
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
from torch import softmax
from tqdm import tqdm

from data_process import (Cic2018_Dataset, Cic_Dataset, UNSW_Dataset,
                          USTC_Dataset)
from gcn import GCN
from util import Classifier

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)

def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--feats_per_node', default=200, type=int)
    parser.add_argument('--layer_1_feats', default=128, type=int)
    parser.add_argument('--layer_2_feats', default=64, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--cls_in_feats', default=128, type=int)
    parser.add_argument('--ck_path', default='model.pt', type=str)
    parser.add_argument('--graph_embs_path', default='data/cic2018/graph_embs.pt',type=str)
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


def neg_sample_com(adj, neg_num):
    hat_a = torch.eye(adj.shape[0])+adj.cpu()
    hat_a=adj.to_dense().cpu()
    neg_indices=torch.where(hat_a==0)
    degs = hat_a.sum(0)
    degs_src, degs_dst=degs[neg_indices[0]],degs[neg_indices[1]]
    k=torch.min(degs_src,degs_dst)/torch.max(degs_src,degs_dst)
    neg_sam_scores=(degs_src+degs_dst)*k
    
    neg_idx=neg_sam_scores.sort().indices
    neg_idx=neg_idx[-neg_num:] #选取负样本
    return neg_indices[0][neg_idx], neg_indices[1][neg_idx]


def get_edge_embs_com(x, adj):
    edge_indices = adj._indices()
    neg_indices=neg_sample_com(adj, len(edge_indices[0]))
    
    #边的特征：[源节点的出边特征、目的节点的入边特征]  对比损失，有边相连的节点表示尽可能相似
    pos_edges = torch.cat([x[edge_indices[0]], x[edge_indices[1]]], dim=1)
    neg_edges = torch.cat([x[neg_indices[0]], x[neg_indices[1]]], dim=1)
    edge_embs = torch.cat([pos_edges, neg_edges])
    label = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))]).long()
    return edge_embs, label


def train(args,graphs, node_feats, device, train_len):
    model = GCN(args, activation = torch.nn.RReLU(), device=device)
    cls=Classifier(args)
    model.to(device)
    cls.to(device)
    gcn_opt=torch.optim.Adam(model.parameters(), lr=0.0001)
    cls_opt=torch.optim.Adam(cls.parameters(), lr=0.0001)
    loss_gcn=nn.BCEWithLogitsLoss()
    model.train()
    for i in range(50):
        el=0
        for idx in tqdm(range(train_len[0],train_len[1])):
            gcn_opt.zero_grad()
            cls_opt.zero_grad()
            gs=graphs[i]
            adj, nx=gs['adj'], node_feats[gs['node_idx']]
            out=model(adj, nx)
            edge_embs, labels=get_edge_embs(out, adj)
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
            'gcn_model':model.state_dict(),
            'cls':cls.state_dict(),
        }, args.ck_path)
    return model
    
def predict(args, graphs, node_feats, device):
    model = GCN(args, activation = torch.nn.RReLU(), device=device)
    cls = Classifier(args).to(device)
    ck=torch.load(args.ck_path)
    model.load_state_dict(ck['gcn_model'])
    cls.load_state_dict(ck['cls'])
    model.to(device)
    graph_embs=[]
    model.eval()
    cls.eval()
    for i in tqdm(range(len(graphs))):
        gs=graphs[i]
        adj, nx=gs['adj'], node_feats[gs['node_idx']]
        out = model(adj, nx)
        edge_indices = adj._indices()
        pos_edges = torch.cat([out[edge_indices[0]], out[edge_indices[1]]], dim=1)
        # graph_embs.append(torch.mean(pos_edges.detach().cpu(), 0))
        pred=cls(pos_edges)
        pred = softmax(pred,0)
        # 边=边*异常概率q
        pos_edges=pos_edges*pred[:,0].unsqueeze(1)
        graph_embs.append(pos_edges.sum(0))
    torch.save(torch.stack(graph_embs), args.graph_embs_path)


if __name__ == '__main__':
    args=parse()
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    t0=time()
    cd =Cic_Dataset()
    graphs=cd.gen_graphs()
    node_feats=cd.gen_node_feats()
    print('预训练时间', time()-t0)
    t1=time()
    train(args, graphs=graphs, node_feats=node_feats, device=device, train_len=[0, 529]) #cic[0,529], unsw[200:600], ustc[10:100]
    print('训练时间', time()-t1)
    predict(args, graphs=graphs, node_feats=node_feats, device=device)
