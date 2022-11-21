import random

import numpy as np
import torch
import torch.nn as nn

from layers import GCN, AvgReadout, Discriminator


class DGI(nn.Module):
    def __init__(self, ip_in,e_in, n_h, outd, activation):
        super(DGI, self).__init__()
        self.ip_gcn = GCN(ip_in, n_h, activation)
        self.e_gcn=GCN(e_in, n_h, activation)
        self.read = AvgReadout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)
        self.lstm=nn.LSTM(n_h, outd)
    
    def forward(self, ip_adj_list, ip_x_list, fu_ip_x, e_adj_list, e_x_list, fu_e_x, sparse, msk,samp_bias1, samp_bias2):
        graph_embs=[]
        for i in range(len(ip_adj_list)):
            ip1 = self.ip_gcn(ip_x_list[i], ip_adj_list[i], sparse)
            e1=self.e_gcn(e_x_list[i], e_adj_list[i],sparse)
        
            true_h=torch.cat((ip1, e1),1)

            c = self.read(true_h,msk)
            c = self.sigm(c)
            graph_embs.append(c)

        graph_embs=torch.stack(graph_embs)
        out,_=self.lstm(graph_embs)
        
        ip2 = self.ip_gcn(fu_ip_x, ip_adj_list[-1], sparse)
        e2=self.e_gcn(fu_e_x, e_adj_list[-1], sparse)
        
        #利用当前状态对边进行采样，使得越可能异常的边进入模型越多
        dc=c.expand_as(e2)
        d=torch.sqrt(torch.sum((e2-dc)**2,2))
        idx=random.choices([i for i in range(e2.shape[1])],weights=d.squeeze(0), k=e2.shape[1]//2)
        e2=e2[:,idx]
        
        false_h=torch.cat((ip2, e2),1)
        ret = self.disc(c, true_h, false_h, samp_bias1, samp_bias2)
        his_mean=out[:-1].mean(0)
        return ret, his_mean, out[-1]

    # Detach the return variables
    def embed(self, seq, adj, sparse, msk):
        h_1 = self.gcn(seq, adj, sparse)
        c = self.read(h_1, msk)

        return h_1.detach(), c.detach()

