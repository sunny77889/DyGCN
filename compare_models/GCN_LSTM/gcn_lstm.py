import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import util as u


def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)

class GCN_LSTM(nn.Module):
    def __init__(self, args, activation, device='cpu'):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=args.layer_2_feats,
            hidden_size=args.lstm_feats,
            num_layers=args.num_lstm_layers
        )
        # self.lstm = nn.GRU(
        #         input_size=args.layer_2_feats,
        #         hidden_size=args.lstm_l2_feats,
        #         num_layers=args.lstm_l2_layers
        # )
        self.device=device
        self.activation=activation
        self.num_layers=args.num_gcn_layers
        self.choose_top_k=TopK(args.layer_2_feats, args.k)
        self.w_list=nn.ParameterList()
        for i in range(self.num_layers):
            if i==0:
                w_i=Parameter(torch.Tensor(args.feats_per_node, args.layer_1_feats))
                reset_param(w_i)
            else:
                w_i=Parameter(torch.Tensor(args.layer_1_feats, args.layer_2_feats))
                reset_param(w_i)
            self.w_list.append(w_i)
    def forward(self, A_list, node_feats, mask_list):
        last_l_seq=[]
        for t, Ahat in enumerate(A_list):
            idx=mask_list[t]
            Ahat, x=Ahat.to(self.device), node_feats.to(self.device)
            x=x.matmul(self.w_list[0])
            x[idx]=self.activation(Ahat.matmul(x[idx]))
            for i in range(1, self.num_layers):
                x=x.matmul(self.w_list[i])
                x[idx]=self.activation(Ahat.matmul(x[idx]))
            last_l_seq.append(x)
        
        last_l_seq=torch.stack(last_l_seq)


        out, _=self.lstm(last_l_seq, None)
        return out[-1]

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        ll=node_embs.shape[0]
        tanh = torch.nn.Tanh()
        out=node_embs * tanh(scores.view(-1,1))
        if ll<self.k:
            t=node_embs[-1] * tanh(scores[-1])
            t=t.unsqueeze(0).repeat(self.k-ll,1)
            out =torch.cat([out, t], 0)
        out=out[:self.k]

        #we need to transpose the output
        return out
                
                
