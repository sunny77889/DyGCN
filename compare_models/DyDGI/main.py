import argparse
import os
import sys
from time import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_process import (Cic2018_Dataset, Cic_Dataset, UNSW_Dataset,
                          USTC_Dataset)
from dgi import DGI

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)

def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ip_feats_dim', default=200, type=int)
    parser.add_argument('--edge_feats_dim', default=77, type=int)
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('--out_dim', default=16, type=int)
    parser.add_argument('--num_layers', default=2, type=int)
    parser.add_argument('--cls_in_feats', default=128, type=int)
    parser.add_argument('--ck_path', default='data/cic/model.pt', type=str)
    parser.add_argument('--graph_embs_path', default='data/cic/graph_embs.pt',type=str)
    return parser.parse_args()
    

def gen_data(ip_x, e_x):

    ip_nodes, e_nodes=ip_x.shape[0], e_x.shape[0]
    idx = np.random.permutation(ip_nodes)
    fu_ip_x = ip_x[idx] #图进行扰乱
    idx = np.random.permutation(e_nodes)
    fu_e_x = e_x[idx] #图进行扰乱
    lbl_1 = torch.ones(1, ip_nodes)
    lbl_2 = torch.zeros(1, ip_nodes)
    ip_labels = torch.cat((lbl_1, lbl_2), 1)

    lbl_1 = torch.ones(1, e_nodes)
    lbl_2 = torch.zeros(1, e_nodes//2)
    e_labels = torch.cat((lbl_1, lbl_2), 1) 

    return fu_ip_x, ip_labels, fu_e_x, e_labels

def train(args,ip_adjs, edge_adjs, ip_feats, edge_feats, device, train_len, seq_len, epochs):
    model = DGI(args.ip_feats_dim,args.edge_feats_dim,args.hidden_dim,args.out_dim, activation = torch.nn.RReLU())
    model.to(device)
    gcn_opt=torch.optim.Adam(model.parameters(), lr=0.0001)
    b_xent = nn.BCEWithLogitsLoss()
    model.train()
    for i in range(epochs):
        el=0
        for idx in range(train_len[0],train_len[1]-seq_len):
            gcn_opt.zero_grad()
            gs=[ip_adjs[i] for i in range(idx,idx+seq_len)]
            ip_adj_list, e_adj_list=[g['adj'].to(device) for g in gs], [edge_adjs[i].to(device) for i in range(idx, idx+seq_len)]
            ip_x_list, e_x_list=[ip_feats[g['node_idx']].to(device) for g in gs], [torch.Tensor(edge_feats[i]).to(device) for i in range(idx, idx+seq_len)]

            fu_ip_x, ip_labels, fu_e_x, e_labels=gen_data(ip_x_list[-1], e_x_list[-1])
            labels=torch.cat((ip_labels, e_labels),1).to(device)
            logits, his_mean, out=model(ip_adj_list, ip_x_list, fu_ip_x, e_adj_list, e_x_list, fu_e_x, True,None, None, None)
            dgi_loss=b_xent(logits, labels)
            lstm_loss=torch.sqrt(torch.sum((out-his_mean)**2, dim=1))
            loss =dgi_loss+lstm_loss

            loss.backward()
            el+=loss.item()
            gcn_opt.step()
        print('epoch:{:.4f}, loss:{:.4f}'.format(i, el/idx))
        torch.save({
            'dgi_model':model.state_dict(),
        }, args.ck_path)
    return model
    
def predict(args,ip_adjs, edge_adjs, ip_feats, edge_feats, device, seq_len):
    model = DGI(args.ip_feats_dim,args.edge_feats_dim,args.hidden_dim,args.out_dim, activation = torch.nn.RReLU())
    ck=torch.load(args.ck_path)
    model.load_state_dict(ck['dgi_model'])
    model.to(device)
    graph_embs=[]
    model.eval()
    for idx in tqdm(range(len(ip_adjs)-seq_len)):
        gs=[ip_adjs[i] for i in range(idx,idx+seq_len)]
        ip_adj_list, e_adj_list=[g['adj'].to(device) for g in gs], [edge_adjs[i].to(device) for i in range(idx, idx+seq_len)]
        ip_x_list, e_x_list=[ip_feats[g['node_idx']].to(device) for g in gs], [torch.Tensor(edge_feats[i]).to(device) for i in range(idx, idx+seq_len)]

        fu_ip_x, ip_labels, fu_e_x, e_labels=gen_data(ip_x_list[-1], e_x_list[-1])
        logits, his_mean, out=model(ip_adj_list, ip_x_list, fu_ip_x, e_adj_list, e_x_list, fu_e_x, True,None, None, None)
        graph_embs.append(out.detach().cpu())
    torch.save(torch.stack(graph_embs).squeeze(1), args.graph_embs_path)


if __name__ == '__main__':
    args=parse()
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    t0=time()
    cd=Cic_Dataset()
    ip_adj_list, edge_adj_list=cd.gen_graphs()
    ip_feats=cd.gen_node_feats()
    edge_feats=cd.gen_edge_feats()
    t1=time()
    print('预训练时间',t1-t0)
 
    train(args, ip_adj_list, edge_adj_list, ip_feats, edge_feats, device=device, train_len=[0,529], seq_len=5, epochs=50) #cic[0,529], unsw[200:600], ustc[10:100]
    t2=time()
    print('训练时间', t2-t1)
    predict(args, ip_adj_list, edge_adj_list, ip_feats, edge_feats, device=device, seq_len=5)
    print('测试时间', time()-t2)




