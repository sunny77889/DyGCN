#%%
import argparse
import os
import sys
sys.path[0]='/home/xiaoqing/gitpro/GNNPro/DyGCN'

from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from DyGCN.read_data import Dataset
from DyGCN.train_test import predict5, train_gcn_lstm5
from DyGCN.utils import set_seed, Namespace, plot_train_loss


def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',default='train', type=str) # 具体进行训练还是测试
    parser.add_argument('--ck_path',default='DyGCN/savedmodel/model.pt', type=str) # 模型保存路径
    parser.add_argument('--embs_path', default='DyGCN/data/graph_embs.pt',type=str) # 子图嵌入结果保存路径
    parser.add_argument('--dataset', default='cic2017', type=str) # 采用cic2017数据集还是cic2018数据集进行实验
    parser.add_argument('--learning_rate', default=0.0001,type=float)
    parser.add_argument('--in_dim', default=77,type=int) # 边属性特征初始维度，cic：77, cic2018：77
    parser.add_argument('--hid_dim', default=32,type=int)
    parser.add_argument('--out_dim', default=16,type=int)
    parser.add_argument('--seq_len', default=5, type=int) # 时序学习序列长度
    return parser.parse_args()

if __name__ =='__main__':
    args=parse()
    set_seed(2) # 设置随机数种子


    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    args.device=device
    
    # 读取数据
    dataset = Dataset(args, os.path.join('DyGCN/data/', args.dataset))
    data=dataset.gen_graphs()
    dataset.degree_adj()
    data=Namespace(data)

    ip_lens=[len(ips) for ips in data.ip_list]
    print("平均每张图包含的节点数目", sum(ip_lens)/len(ip_lens))

    if args.mode=='train':
        t0=time()
        print('model tarin.....')
        losses=train_gcn_lstm5(args, data, epochs=1, train_len=[0, 529]) # cic[0,529], unsw[200:600], ustc[10:100]
        plot_train_loss(losses)
        t1=time()
        print('DyGCN 训练时间：', t1-t0)
    else:
        t0=time()
        print('model test....')
        predict5(args, data)
        print('DyGCN 测试时间：', time()-t0)
    