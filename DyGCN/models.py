import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import Tanh
from torch.nn.parameter import Parameter
from torch.utils.data import dataset
from torch.nn.functional import softmax
import DyGCN.utils as u
from DyGCN.layers import GraphConvolution, GraphConvolution2, GraphConvolution3


class DGCN3(nn.Module):
    def __init__(self, args):
        super(DGCN3, self).__init__()
        self.args=args
        self.gc1 = GraphConvolution(args.in_dim, args.out_dim) # 聚合出边特征
        self.gc2 = GraphConvolution(args.in_dim, args.out_dim) # 聚合入边特征
        self.gc3 = GraphConvolution(args.in_dim, args.out_dim) # 边权重为结构特征的图
        # self.gc4 = GraphConvolution(args.in_dim, args.hid_dim) # 流-k近邻图
        
        self.lstm=nn.LSTM(input_size=args.hid_dim, hidden_size=args.out_dim ) #学习节点的时序性
        self.dropout = 0.5
        self.ln=nn.LayerNorm(args.hid_dim)
        
    def node_history(self, ips, cur_ips, output):
        '''根据当前时刻的节点集cur_ips,获取历史i时刻节点的属性'''
        idx1 = np.where(np.in1d(ips, cur_ips))[0]
        idx2 = np.where(np.in1d(cur_ips, ips))[0]
        aa = torch.zeros(len(cur_ips), self.args.hid_dim).to(output.device)
        aa[idx2]=output[idx1]
        return aa.unsqueeze(0)

    def forward(self, x_list, Ain_list, Aout_list,A_list, ips_list, cur_ips, node_X, struct_adj):
        '''
        x_list: 边的特征向量
        Ain_list: 入向邻接矩阵
        Aout_list: 出边邻接矩阵
        A_list: 节点邻接矩阵
        ips_list: 节点集序列
        cur_ips: 当前时刻节点集
        '''
        seqs=[]
        struct_weight=[]
        for i in range(len(Ain_list)):
            x, Ain, Aout, Adj = x_list[i], Ain_list[i], Aout_list[i], A_list[i]
            node_in= self.gc1(x, Ain) # 聚合节点的入边特征
            node_out = self.gc2(x, Aout) # 聚合节点的出边特征
            # struct_x = self.gc3(node_X[i].to(x.device), struct_adj[i].to(x.device))
            
            # node_feat=torch.stack((node_in, node_out, struct_x), 0)
            # alpha_x=softmax(node_feat) # 输出不同类型节点特征的权重
            # node_feat=alpha_x*node_feat
            # node_feat=torch.sum(node_feat, 0) # 加权和

            node_feat=torch.cat((node_in, node_out),1) # 拼接节点的入边特征-出边特征作为节点的聚合特征
            node_feat = F.dropout(node_feat, 0.5)
            # node_feat=self.ln(node_feat)
            seqs.append(self.node_history(ips_list[i], cur_ips, node_feat))
        seqs=torch.vstack(seqs)
        output, _ =self.lstm(seqs) # 学习节点的时序
        # flow_knn_output=self.gc4(x, flow_knn_adj.to(x.device))
        return output[-1]

class DGCN2(nn.Module):
    def __init__(self,nfeat, nhid,outd, dropout):
        super(DGCN2, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid) # 聚合出边特征
        self.gc2 = GraphConvolution(nhid, outd)#, bias=False, type='node')#聚合入边特征
        self.lstm=nn.LSTM(input_size=outd, hidden_size=16)
        self.dropout = dropout
        self.ln=nn.LazyBatchNorm1d(outd)
    def get_hisNode(self, ips,cur_ips, output):
        '''根据当前时刻的节点集cur_ips,获取历史i时刻节点的属性'''
        idx1 = np.where(np.in1d(ips, cur_ips))[0]
        idx2 = np.where(np.in1d(cur_ips, ips))[0]
        aa = torch.zeros(len(cur_ips),32)
        aa[idx2]=output[idx1]
        return aa.unsqueeze(0)
    def forward(self, x_list, ifa_list, adj_list, ips_list, cur_ips):
        seqs=[]
        for i in range(len(ifa_list)):
            x, ifa, adj = x_list[i],ifa_list[i], adj_list[i]
            x = F.relu(self.gc1(x,ifa))# 源节点聚合出边特征生成节点特征
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.relu(self.gc2(x, adj))#聚合邻居节点特征并与自身特征做加和
            x = self.ln(x)
            seqs.append(self.get_hisNode(ips_list[i], cur_ips, x))
        output,_=self.lstm(torch.vstack(seqs))
        return output[-1]

class DGCN(nn.Module):
    def __init__(self, nfeat, nhid, outd, dropout):
        super(DGCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, outd)
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(outd)

    def forward(self, x, IFadj, adj):
        x = F.relu(self.gc1(x, IFadj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.bn(x)
        return x

class LSTM_AE(nn.Module):
    def __init__(self, fea_dim):
        super(LSTM_AE,self).__init__()
        
        self.l1 = nn.LSTM(input_size=fea_dim, hidden_size=16, num_layers=1, batch_first=True)
        self.l2 = nn.LSTM(input_size=16, hidden_size=fea_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        encoded, _ = self.l1(x)
        decoded, _ = self.l2(encoded)
        return encoded, decoded     
        
class Classifier(torch.nn.Module):
    def __init__(self,indim,out_features=2):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = indim, out_features =16),
                                       activation,
                                       torch.nn.Linear(in_features =16,out_features = out_features))

    def forward(self,x):
        return self.mlp(x)

class AutoEncoder(torch.nn.Module):
    def __init__(self,in_dim):
        super(AutoEncoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh()
        )
        self.decoder=nn.Sequential(
            nn.Linear(32, in_dim),
        )
    def forward(self,x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat
