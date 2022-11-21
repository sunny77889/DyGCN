# coding: utf-8
import gc
import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from helper import DataLoader
from metrics import NegativeSamplingLoss
from models import CGCN, CTGCN

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)

def get_data_loader(args):
    base_path = args['base_path']
    origin_folder = args['origin_folder']
    core_folder = args.get('core_folder', None)
    nfeature_folder = args.get('nfeature_folder', None)
    node_file = args['node_file']
    has_cuda = args['has_cuda']

    origin_base_path = os.path.abspath(os.path.join(base_path, origin_folder)) if origin_folder else None
    core_base_path = os.path.abspath(os.path.join(base_path, core_folder)) if core_folder else None
    node_feature_path = os.path.abspath(os.path.join(base_path, nfeature_folder)) if nfeature_folder else None
    max_time_num = len(os.listdir(origin_base_path)) if origin_base_path else len(os.listdir(core_base_path))
    assert max_time_num > 0

    data_loader = DataLoader(max_time_num, has_cuda=has_cuda)
    args['origin_base_path'] = origin_base_path
    args['core_base_path'] = core_base_path
    args['nfeature_path'] = node_feature_path
    return data_loader


def get_loss(idx, time_length, data_loader, args):
    base_path = args['base_path']
    walk_pair_folder = args['walk_pair_folder']
    node_freq_folder = args['node_freq_folder']
    neg_num = args['neg_num']
    walk_pair_base_path = os.path.abspath(os.path.join(base_path, walk_pair_folder))
    node_freq_base_path = os.path.abspath(os.path.join(base_path, node_freq_folder))
    node_pair_list = data_loader.get_node_pair_list(walk_pair_base_path, start_idx=idx, duration=time_length)
    neg_freq_list = data_loader.get_node_freq_list(node_freq_base_path, start_idx=idx, duration=time_length)
    loss = NegativeSamplingLoss(node_pair_list, neg_freq_list, neg_num=neg_num)
    return loss


def gnn_embedding(method, args, ip_indices, ip_num, train_len, epoches):
    # common params
    base_path = args['base_path']
    model_folder = args['model_folder']
    model_file = args['model_file']
    duration = args['duration']
    hidden_dim = args['hid_dim']
    embed_dim = args['embed_dim']
    lr = args['lr']
    input_dim = args['input_dim']
    shuffle = args['shuffle']
    weight_decay = args['weight_decay']
    trans_num = args['trans_layer_num']
    diffusion_num = args['diffusion_layer_num']
    hidden_dim = args['hid_dim']
    model_type = args['model_type']
    rnn_type = args['rnn_type']
    trans_activate_type = args['trans_activate_type']
    bias = args.get('bias', None)
    data_loader = get_data_loader(args)
    core_base_path = args['core_base_path']
    max_core = args['max_core']
    model_base_path=os.path.abspath(os.path.join(base_path, model_folder))

    t1 = time.time()
    
    print('start ' + method + ' embedding!')
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    time_length = 5
    model = CTGCN(input_dim, hidden_dim, embed_dim, ip_num, trans_num=trans_num, diffusion_num=diffusion_num, duration=time_length, bias=bias, rnn_type=rnn_type,
                         model_type=model_type, trans_activate_type=trans_activate_type)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    x, input_dim=data_loader.get_feature_list(ip_num=ip_num,data_path=base_path)
    x=x.to(device)
    args['input_dim'] = input_dim
    model.train()
    for epoch in range(epoches):
        epoch_loss=0
        for idx in range(train_len[0],train_len[1]-time_length):
            optimizer.zero_grad()
            adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_length, max_core=max_core)
            x_list=[x[ip_indices[i]] for i in range(idx, idx+duration)]
            loss_model = get_loss(idx, time_length, data_loader, args).to(device)
            indices=[ip_indices[i] for i in range(idx, idx+duration)]
            node_indices=[]
            for i in range(duration):
                node_num=x_list[i].shape[0]
                all_nodes = torch.arange(node_num, device=device)
                node_indices.append(all_nodes[torch.randperm(node_num)] if shuffle else all_nodes)  # Tensor

            embedding_list = model(x_list, adj_list, indices, device)
            loss = loss_model(embedding_list, node_indices, indices)
            loss.backward()
            optimizer.step()  
            epoch_loss+=loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            else:
                gc.collect()
        torch.save(model.state_dict(), os.path.join(model_base_path, model_file))
        print('epoch:', epoch, 'loss:', epoch_loss.item()/idx)
    t2 = time.time()
    print('finish ' + method + ' embedding! cost time: ', t2 - t1, ' seconds!')

def gnn_embedding_predict(method, args, ip_indices, ip_num, ):
    # common params
    base_path = args['base_path']
    model_folder = args['model_folder']
    model_file = args['model_file']
    hidden_dim = args['hid_dim']
    embed_dim = args['embed_dim']
    input_dim = args['input_dim']
    trans_num = args['trans_layer_num']
    diffusion_num = args['diffusion_layer_num']
    hidden_dim = args['hid_dim']
    model_type = args['model_type']
    rnn_type = args['rnn_type']
    trans_activate_type = args['trans_activate_type']
    bias = args.get('bias', None)
    data_loader = get_data_loader(args)
    core_base_path = args['core_base_path']
    max_core = args['max_core']
    model_base_path=os.path.abspath(os.path.join(base_path, model_folder))
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    t1 = time.time()
    
    data_len=len(ip_indices)
    x, input_dim=data_loader.get_feature_list(ip_num=ip_num,data_path=base_path)
    x=x.to(device)
    args['input_dim'] = input_dim
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    time_length = 5
    model = CTGCN(input_dim, hidden_dim, embed_dim, ip_num, trans_num=trans_num, diffusion_num=diffusion_num, duration=time_length, bias=bias, rnn_type=rnn_type,
                         model_type=model_type, trans_activate_type=trans_activate_type)

    model = model.to(device)
    model_path = os.path.join(model_base_path, model_file)
    model.load_state_dict(torch.load(model_path))
    # data_len-time_length
    model.eval()
    graph_embs=[]
    for idx in tqdm(range(data_len-time_length)):
        # 读数据
        adj_list = data_loader.get_core_adj_list(core_base_path, start_idx=idx, duration=time_length, max_core=max_core)
        x_list=[x[ip_indices[i]] for i in range(idx, idx+time_length)]
        indices= [ip_indices[i] for i in range(idx, idx+time_length)]
        embedding_list = model(x_list, adj_list, indices, device)
    
        embedding = embedding_list[-1][indices[-1]]
        graph_emb=embedding.cpu().detach().mean(0)
        graph_embs.append(graph_emb)
    torch.save(torch.stack(graph_embs),os.path.join(base_path,'graph_embs.pt'))
    t2 = time.time()
    print('finish ' + method + ' embedding! cost time: ', t2 - t1, ' seconds!')
    