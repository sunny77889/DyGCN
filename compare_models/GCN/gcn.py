import math
from turtle import forward

import torch
import torch.nn as nn
from matplotlib.pyplot import cla
from torch.nn.parameter import Parameter


def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)

class GCN(nn.Module):
    def __init__(self, args, activation, device):
        super().__init__()
        self.activation=activation
        self.num_layers=args.num_layers
        self.device=device

        self.w_list=nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i=Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                reset_param(w_i)
            else:
                w_i=Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                reset_param(w_i)
            self.w_list.append(w_i)
    def forward(self, A_hat, nodefeat):
        A_hat, nodefeat=A_hat.to(self.device), nodefeat.to(self.device)
        last_l=self.activation(A_hat.matmul(nodefeat.matmul(self.w_list[0])))
        for i in range(1, self.num_layers):
            last_l=self.activation(A_hat.matmul(last_l.matmul(self.w_list[i])))
        return last_l
    

