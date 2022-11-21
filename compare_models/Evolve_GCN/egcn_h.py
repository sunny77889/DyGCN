import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import util as u


class EGCN(torch.nn.Module):
    def __init__(self, args, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = u.Namespace({})

        feats = [args.feats_per_node,
                 args.layer_1_feats,
                 args.layer_2_feats]
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = []
        self._parameters = nn.ParameterList()
        for i in range(1,len(feats)):
            GRCU_args = u.Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            #print (i,'grcu_i', grcu_i)
            self.GRCU_layers.append(grcu_i.to(self.device))
            self._parameters.extend(list(self.GRCU_layers[-1].parameters()))

    def parameters(self):
        return self._parameters

    def forward(self,A_list, Nodes_list, nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list, nodes_mask_list, self.device)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = u.Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list,mask_list, device):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            Ahat=Ahat.to(device)
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights, node_embs, mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q,prev_z, mask):
        z_topk=self.choose_topk(prev_z,mask)

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):

        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

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

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        ll=mask.shape[0]
        tanh = torch.nn.Tanh()
        out=node_embs * tanh(scores.view(-1,1))
        if ll<self.k:
            t=node_embs[-1] * tanh(scores[-1])
            t=t.unsqueeze(0).repeat(self.k-ll,1)
            out =torch.cat([out, t], 0)
        out=out[:self.k]

        #we need to transpose the output
        return out.t()
