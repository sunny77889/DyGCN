# coding: utf-8
import argparse
import json
import os
import sys
from time import time

from data_process import Cic_Dataset
from preprocessing import preprocess
from train import gnn_embedding, gnn_embedding_predict

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)

def parse_args(args):
    parser = argparse.ArgumentParser(prog='CTGCN', description='K-core based Temporal Graph Convolutional Network')
    parser.add_argument('--config', nargs=1, default=['config/uci.json'],type=str, help='configuration file path')
    parser.add_argument('--task', type=str, default='embedding', help='task name which is needed to run')
    parser.add_argument('--method', type=str, default="CTGCN-C", help='graph embedding method, only used for embedding task')
    return parser.parse_args(args)


def parse_json_args(file_path):
    config_file = open(file_path)
    json_config = json.load(config_file)
    config_file.close()
    return json_config
import numpy as np

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    print('args:', args)
    print(args.config[0])
    config_dict = parse_json_args(args.config[0])
    t0=time()
    Cic_Dataset().gen_graphs()
    # preprocess(config_dict['preprocessing'][args.method]) #模型训练时不需要

    ip_indices=np.load('data/cic/ip_indices.npy', allow_pickle=True)
    ip_num=np.concatenate(ip_indices).max()+1
    param_dict =config_dict['embedding'][args.method]
    t1=time()
    print('预处理时间',t1-t0)
    gnn_embedding(args.method, param_dict, ip_indices,ip_num,train_len=[0, 529], epoches=50) #cic[0,529], unsw[200:600], ustc[10:100]
    t2=time()
    print('训练时间', t2-t1)
    gnn_embedding_predict(args.method, param_dict, ip_indices, ip_num)

