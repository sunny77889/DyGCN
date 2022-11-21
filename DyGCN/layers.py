import math

import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_sequence


# 衰减的时序聚合 
class GraphConvolution3(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution3, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm = nn.LSTM(32, 32,1)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        '''为了让每次训练产生的初始参数尽可能相同，从而便于实验结果的复现，可以设置固定的随机数生成种子'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.lstm.reset_parameters()
    
    def decay(self, i, flow):
        if i==0:
            return flow[i]
        return flow[i]+0.5*self.decay(i-1,flow)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support) #稀疏矩阵相乘
        seq_node=[]
        aa = adj.to_dense()
        for a in aa:
            idx = torch.nonzero(a).squeeze(1)
            if len(idx)==0:
                seq_node.append(torch.zeros(self.out_features))
            else:
                idx = idx[len(idx)-100:len(idx)]
                seq_node.append(self.decay(len(idx)-1, support[idx]))
        output=torch.stack(seq_node)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        '''该方法是类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”'''
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# LSTM学习边的时序得到节点的特征d
class GraphConvolution2(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lstm = nn.LSTM(out_features, out_features, 1)
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        '''为了让每次训练产生的初始参数尽可能相同，从而便于实验结果的复现，可以设置固定的随机数生成种子'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        self.lstm.reset_parameters()

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support) #稀疏矩阵相乘
        seq_node=[]
        aa = adj.to_dense()
        for a in aa:
            idx = torch.nonzero(a).squeeze(1)
            seq_node.append(support[idx[len(idx)-10:len(idx)]]) # 选取后10个流输入到lstm中
        seq_node = pad_sequence(seq_node)
        output,_=self.lstm(seq_node)
        output=output[-1]
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        '''该方法是类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”'''
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

# 原始GCN层
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, type=None):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.type=type
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        '''为了让每次训练产生的初始参数尽可能相同，从而便于实验结果的复现，可以设置固定的随机数生成种子'''
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj.to(support.device), support) #聚合邻居特征
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        '''该方法是类的实例化对象用来做“自我介绍”的方法，默认情况下会返回当前对象的“类名+object at+内存地址”'''
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
