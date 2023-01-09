import datetime
import glob
import math
import os
from collections import Counter, defaultdict
from time import time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

FLOW_NUM=1000 # 每个通信图中包含的流的数目

class Dataset():
    def __init__(self, args, save_path):
        self.adj_list=[]
        self.label_list=[]
        self.Ain_list=[]
        self.Aout_list=[]
        self.A_list=[]
        self.ip_list=[] # 每个图的ip
        self.node_feats={} # 随机生成的节点特征
        self.dataset=args.dataset

        self.edge_file=os.path.join(save_path, 'edges.npy')
        self.feat_file=os.path.join(save_path, 'feats.npy')
        self.label_types_file=os.path.join(save_path, 'label_types.npy')
        self.label_file=os.path.join(save_path, 'labels.npy')
    
    def normalize_random(self,adj):
        """邻接矩阵归一化：随机游走归一化"""
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -1).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        return adj
    
    def normalize_sym(self,adj):
        """邻接矩阵归一化：对称归一化"""
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        return torch.mm(adj, d_mat_inv_sqrt)
    
    def gen_node_feats(self, ips):
        for ip in ips:
            if ip not in self.node_feats:
                self.node_feats[ip]=torch.randn(77)

    def gen_graphs(self):
        dataset_name=self.dataset.split('/')[-1]
        assert dataset_name in ['cic2017', 'cic2018']
        if os.path.exists(self.edge_file):
            self.edges = np.load(self.edge_file, allow_pickle=True)
            self.feat_list=np.load(self.feat_file, allow_pickle=True)
            self.label_types=np.load(self.label_types_file, allow_pickle=True)
            self.labels=np.load(self.label_file, allow_pickle=True)
        elif dataset_name=='cic2017':
            self.process_cic2017()
        else:
            self.process_cic2018()
        

        le = LabelEncoder()
        for i in tqdm(range(len(self.edges))):
            e = self.edges[i]
            edges = e[:,:2] #源IP，目的IP
            cc=dict(Counter(edges[:,0]))
            
            labels = e[:, 2].astype(np.long)
            
            ips = le.fit_transform(edges.reshape(-1))
            t='205.174.165.73'
            if t in cc and cc[t]==72:
                print(i)
                print(le.transform([t]))
            self.ip_list.append(le.classes_.tolist())
            self.gen_node_feats(le.classes_.tolist()) # 为当前IP生成节点特征
            
            edges=ips.reshape(-1,2)
            n = len(le.classes_)
            A_out = torch.sparse_coo_tensor([edges[:,0],range(FLOW_NUM)], torch.ones(FLOW_NUM), size=[n, FLOW_NUM])
            A_in = torch.sparse_coo_tensor([edges[:,1],range(FLOW_NUM)], torch.ones(FLOW_NUM), size=[n, FLOW_NUM])
            adj = torch.sparse_coo_tensor(edges.T, torch.ones(FLOW_NUM),size=[n, n])
            self.A_list.append(adj)
            adj = self.normalize_sym(torch.eye(adj.shape[0])+adj)
            A_out=self.normalize_random(A_out.to_dense())
            A_in=self.normalize_random(A_in.to_dense())
            self.Aout_list.append(A_out.to_sparse())#源IP-Flow邻接矩阵：有向图
            self.Ain_list.append(A_in.to_sparse()) #目的IP-Flow邻接矩阵: 有向图
            self.adj_list.append(adj.to_sparse()) # ip节点邻接矩阵：无向图
            self.label_list.append(labels) # 图中每条流的边
        
        # df = pd.DataFrame({'types':np.array(mal_types)[idx].tolist(),'num':np.array(mal_num)[idx].astype(int).tolist()}) 
        # print(df.groupby('types').agg('num'))
        self.degree_adj()
        return {
            'Ain_list':self.Ain_list,
            'Aout_list': self.Aout_list,
            'A_list':self.A_list,
            'adj_list':self.adj_list,
            'feat_list':self.feat_list,
            'ip_list': self.ip_list,
            'node_feats': self.node_feats,
            'struct_adj':self.S_adjs
        }

    
    def degree_adj(self, type='out'):
        # 事实证明 入度中心性的效果最好
        self.S_adjs=[]
        for adj in self.A_list:
            adj=torch.eye(adj.shape[0])+adj.to_dense()
            adj[adj>1]=1
            if type=='out':
                degrees=adj.sum(1)
            elif type=='in':
                degrees=adj.sum(0)
            elif type=='out_weight':
                degrees=adj.sum(1)/torch.sum(adj)
            elif type=='in_weight':
                degrees=adj.sum(0)/torch.sum(adj)
            elif type=='out_centrality':
                degrees=adj.sum(1)/adj.shape[0]
            elif type=='in_centrality':
                degrees=adj.sum(0)/adj.shape[0]
            elif type=='degree':
                degrees=1/(1+abs(adj.sum(0)-adj.sum(1)))
            
            
            adj = torch.mul(adj, degrees)
            self.S_adjs.append(self.normalize_sym(adj))
        return self.S_adjs

    def struct_feture_adj(self, type='betweenness_centrality'):
        self.S_adjs=[]
        i=0
        for adj in self.A_list:
            adj=torch.eye(adj.shape[0])+adj.to_dense()
            adj[adj>1]=1
            G=nx.from_numpy_array(adj.numpy())
            score = nx.betweenness_centrality(G)
            feas=list(score.values())
            adj = torch.mul(adj, torch.Tensor(feas))
            self.S_adjs.append(self.normalize_sym(adj))
            if i==2746:
                sg=nx.from_numpy_array(adj.numpy())
                nx.draw(sg, with_labels=sg.nodes)
                plt.savefig('nx.png')
                print(score)
            i+=1
        return self.S_adjs



    def process_cic2017(self):
        t0=time()
        csv_list=glob.glob('data/cic2017/'+'*.csv')
        print("共发现%s个csv文件"%len(csv_list))
        dataframe_list=[]
        for file in csv_list:
            df=pd.read_csv(file, index_col=None, encoding='unicode_escape')
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
            k=math.floor(len(df)/FLOW_NUM)
            # 按照时间戳进行排序，原始的时间戳有问题
            td = datetime.timedelta(hours=12)
            aa=datetime.datetime.strptime(df[' Timestamp'][0].split(' ')[0]+' 8:00:00',"%d/%m/%Y %H:%M:%S")
            if file.split('/')[-1].split('-')[0] != '1Monday':
                df[' Timestamp']=[datetime.datetime.strptime(t, "%d/%m/%Y %H:%M") for t in df[' Timestamp']] #先将字符串类型的时间戳转换为时间戳
            else:
                self.train_len=k
                df[' Timestamp']=[datetime.datetime.strptime(t, "%d/%m/%Y %H:%M:%S") for t in df[' Timestamp']] #先将字符串类型的时间戳转换为时间戳

            df[' Timestamp']=df[' Timestamp'].apply(lambda x:x if x>aa else x+td) #把下午的时间改为24小时，1点改为13点
            df = df.sort_values(by=[' Timestamp']) #首先按照时间戳排序
            dataframe_list.append(df.iloc[:k*FLOW_NUM,:])
        df = pd.concat(dataframe_list)
        feats=df.drop(columns=['Flow ID', ' Timestamp', ' Fwd Header Length.1', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Label'])
        df = df.loc[:, [' Source IP', ' Destination IP',' Label']]
        self.mal_types=df[' Label'].values.reshape(-1, FLOW_NUM)
        np.save(self.label_types_file, self.mal_types)
        df.loc[df[' Label']!='BENIGN', ' Label']=1
        df.loc[df[' Label']=='BENIGN', ' Label']=0
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        graph_labels =df[' Label'].values.reshape(-1, FLOW_NUM)
        graph_labels = np.max(graph_labels, axis=1)

        self.edges=df.values.reshape(-1,FLOW_NUM,3)
        self.feat_list=feats.reshape(-1, FLOW_NUM, feats.shape[1])

        np.save(self.label_file, graph_labels)
        np.save(self.edge_file, self.edges)
        np.save(self.feat_file, self.feat_list)
        print('cic2017 预处理时间:', time()-t0)
        

    def process_cic2018(self):
        t0=time()
        file=os.path.join('data/cic2018/', 'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv')
        df = pd.read_csv(file, encoding='unicode_escape')
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
        # df = df[-(df['Src IP']==df['Dst IP'])] #删除源IP和目的IP相同的流
        k=math.floor(len(df)/FLOW_NUM)
        df=df.iloc[:k*FLOW_NUM,:]
        df = df.sort_values(by=['Timestamp'])
        df=df[400000:] # 前40万条流中包含异常，由于是基于时序的异常检测，因此忽略这部分流
        feats=df.drop(columns=['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'Label'])
        df = df.loc[:, ['Src IP','Dst IP','Label']]
        self.mal_types=df['Label'].values.reshape(-1, FLOW_NUM)
        np.save(self.label_types_file, self.mal_types)
        df.loc[df['Label']!='Benign', 'Label']=1
        df.loc[df['Label']=='Benign', 'Label']=0
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        graph_labels =df['Label'].values.reshape(-1, FLOW_NUM)
        graph_labels = np.max(graph_labels, axis=1)

        self.edges=df.values.reshape(-1,FLOW_NUM,3)
        self.feat_list=feats.reshape(-1, FLOW_NUM, feats.shape[1])

        np.save(self.label_file, graph_labels)
        np.save(self.edge_file, self.edges)
        np.save(self.feat_file,self.feat_list)
        print('cic2018 预处理时间:', time()-t0)


