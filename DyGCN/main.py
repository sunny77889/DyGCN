import argparse
import os
import sys
sys.path[0]='/home/ypd-23-teacher-2/xiaoqing/DyGCN/'

from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import numpy as np
from DyGCN.read_data import Dataset
from DyGCN.train_test import predict5, train_gcn_lstm5
from DyGCN.utils import set_seed, Namespace, plot_train_loss
# _struct_indegree,_struct_indegree_weight,_struct_indegree_centrality

save_name='_struct_degree' 
def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',default= 'test', type=str) # 具体进行训练还是测试
    parser.add_argument('--ck_path',default='DyGCN/savedmodel/cic2018/model'+save_name+'.pt', type=str) # 模型保存路径
    parser.add_argument('--embs_path', default='DyGCN/data/cic2018/graph_embs'+save_name+'.pt',type=str) # 子图嵌入结果保存路径
    parser.add_argument('--dataset', default='cic2018', type=str) # 采用cic2017数据集还是cic2018数据集进行实验
    parser.add_argument('--learning_rate', default=0.0001,type=float)
    parser.add_argument('--in_dim', default=77,type=int) # 边属性特征初始维度，cic：77, cic2018：77
    parser.add_argument('--hid_dim', default=32,type=int)
    parser.add_argument('--out_dim', default=16,type=int)
    parser.add_argument('--seq_len', default=5, type=int) # 时序学习序列长度
    return parser.parse_args()
# te={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0, 24: 0.0, 25: 0.0, 26: 0.0, 27: 0.0, 28: 0.017871144729886424, 29: 0.0, 30: 0.0, 31: 0.0, 32: 0.006274654548248016, 33: 0.0, 34: 0.0, 35: 0.0, 36: 0.0, 37: 0.03762527233115466, 38: 0.0666615330678887, 39: 0.13764579497983095, 40: 0.13090638165804516, 41: 0.05306513409961686, 42: 0.2753630579847618, 43: 0.05311302681992337, 44: 0.00934037280301648, 45: 0.32935394784746264, 46: 0.41141163499936306, 47: 0.2013868842659629, 48: 0.0, 49: 0.0005695790449002075, 50: 0.3649125313746678, 51: 0.0, 52: 0.0, 53: 0.0, 54: 0.0, 55: 0.0, 56: 0.0, 57: 0.0, 58: 0.0, 59: 0.0, 60: 0.06147427372094644, 61: 0.0, 62: 0.001628352490421456, 63: 0.0067519678681203146, 64: 0.001628352490421456, 65: 0.0, 66: 0.0, 67: 0.0, 68: 0.0022932490869574966, 69: 0.0, 70: 0.0, 71: 0.0, 72: 0.0, 73: 0.0, 74: 0.0, 75: 0.0, 76: 0.0, 77: 0.0, 78: 0.0, 79: 0.0, 80: 0.0, 81: 0.0, 82: 0.008003505731122181, 83: 0.0, 84: 0.0, 85: 0.0, 86: 0.0, 87: 0.0, 88: 0.0, 89: 0.026875588038535996, 90: 0.0, 91: 0.0, 92: 0.0, 93: 0.0, 94: 0.0, 95: 0.0, 96: 0.0, 97: 0.02112006509768155, 98: 0.0, 99: 0.0, 100: 0.0, 101: 0.0, 102: 0.0, 103: 0.0, 104: 0.0, 105: 0.0, 106: 0.0, 107: 0.0, 108: 0.0, 109: 0.0, 110: 0.0, 111: 0.0, 112: 0.0, 113: 0.0, 114: 0.0, 115: 0.0, 116: 0.0, 117: 0.0, 118: 0.0, 119: 0.0, 120: 0.0, 121: 0.0, 122: 0.0, 123: 0.0, 124: 0.053746683426810465, 125: 0.0, 126: 0.0, 127: 0.0, 128: 0.0, 129: 0.0, 130: 0.0, 131: 0.0, 132: 0.0, 133: 0.0, 134: 0.0, 135: 0.0, 136: 0.0, 137: 0.0, 138: 0.0, 139: 0.0, 140: 0.0, 141: 0.0, 142: 0.0, 143: 0.0, 144: 0.0, 145: 0.0}
# print(max(te.values()))
if __name__ =='__main__':
    args=parse()
    set_seed(2) # 设置随机数种子

    device=torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    args.device=device
    
    # 读取数据
    print(args.dataset, '读取数据....')
    dataset = Dataset(args, os.path.join('DyGCN/data/', args.dataset))

    data=dataset.gen_graphs()
    t0=time()
    # struct_adj=dataset.degree_adj(type='degree')
    # struct_adj=dataset.struct_feture_adj(type='betweenness_centrality')
    print('结构特征图生成时间为', time()-t0)
    # flow_knn_adj=torch.load('flow_knn_graphs.pt')
    data=Namespace(data)

    ip_lens=[len(ips) for ips in data.ip_list]
    print("平均每张图包含的节点数目", sum(ip_lens)/len(ip_lens))
    train_len=[0, 529] if args.dataset=='cic2017' else [0,4600] # cic2017[0,529], cic2018[0, 4600] unsw[200:600], ustc[10:100]

    if args.mode=='train':
        t0=time()
        print('model tarin .....')
        losses=train_gcn_lstm5(args, data, epochs=50, train_len=train_len)
        plot_train_loss(losses)
        print('DyGCN 训练时间：', time()-t0)    
        t0=time()   
        print('model test .....')
        predict5(args, data)
        print('DyGCN 测试时间：', time()-t0)
    else: 
        t0=time()
        print('model test .....')           
        predict5(args, data)
        print('DyGCN 测试时间：', time()-t0)
