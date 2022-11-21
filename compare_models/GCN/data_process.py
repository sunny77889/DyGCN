import datetime
import glob
import json
import math
import os
from collections import Counter, defaultdict
from datetime import timedelta
from imghdr import tests
from pydoc import pager
from time import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from matplotlib.pyplot import axis
from numpy.core.fromnumeric import trace
from numpy.core.numeric import indices
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)
class Cic_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000
        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/GCN/data/cic/'
    
    def normalize_sym(self,adj):
        """原始归一化方式，对称归一化"""
        # adj = sp.coo_matrix(adj)
        adj=adj.to_dense()
        rowsum = np.array(1+adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        adj=sp.csr_matrix(torch.mm(adj, d_mat_inv_sqrt))
        coo = adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def gen_node_feats(self):
        file=self.data_path+'nodes_feats.pt'
        if os.path.exists(file):
            self.nodes_feats=torch.load(file)
        else:
            self.nodes_feats = torch.randn(self.ip_num,200) #从标准正态分布中随机采样
            torch.save(self.nodes_feats, self.data_path+'nodes_feats.pt')
        return self.nodes_feats
    
    def gen_graphs(self):
        data_path=self.data_path+'data.npy'
        if os.path.exists(data_path):
            self.graphs=np.load(data_path, allow_pickle=True)
            return self.graphs

        csv_list=glob.glob('/home/xiaoqing/gitpro/GNNPro/DyGCN/data/CIC2017/rawdata/'+'*.csv')
        print("共发现%s个csv文件"%len(csv_list))
        i=0
        dataframe_list=[]
        for file in csv_list:
            print(file)
            df=pd.read_csv(file, index_col=None, encoding='unicode_escape')
            print(len(df))
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
            k=math.floor(len(df)/self.graph_len)
            #按照时间戳进行排序，原始的时间戳有问题
            td = timedelta(hours=12)
            aa=datetime.datetime.strptime(df[' Timestamp'][0].split(' ')[0]+' 8:00:00',"%d/%m/%Y %H:%M:%S")
            if file.split('/')[-1].split('-')[0] != '1Monday':
                df[' Timestamp']=[datetime.datetime.strptime(t, "%d/%m/%Y %H:%M") for t in df[' Timestamp']] #先将字符串类型的时间戳转换为时间戳
            else:
                self.train_len=k
                df[' Timestamp']=[datetime.datetime.strptime(t, "%d/%m/%Y %H:%M:%S") for t in df[' Timestamp']] #先将字符串类型的时间戳转换为时间戳

            df[' Timestamp']=df[' Timestamp'].apply(lambda x:x if x>aa else x+td) #把下午的时间改为24小时，1点改为13点
            df = df.sort_values(by=[' Timestamp']) #首先按照时间戳排序
            print(k)
            dataframe_list.append(df.iloc[:k*self.graph_len,:])

        df = pd.concat(dataframe_list)
        df = df.loc[:, [' Source IP', ' Destination IP',' Label']]
        df.loc[df[' Label']!='BENIGN', ' Label']=1
        df.loc[df[' Label']=='BENIGN', ' Label']=0
        graph_labels =df[' Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1)
        np.save(self.data_path+'labels.npy', graph_labels.astype(np.int))

        ips=pd.concat([df[' Source IP'], df[' Destination IP']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df[' Source IP'].values), le.transform(df[' Destination IP'].values)
        adj_list=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices)
            adj=torch.sparse_coo_tensor([le.transform(sip), le.transform(dip)], values=torch.ones(1000), size=(ip_num, ip_num))
            adj=self.normalize_sym(adj)
            graph={'adj':adj, 'node_idx':ip_indices}
            adj_list.append(graph)
        
        self.graphs=np.array(adj_list)
        np.save(data_path, self.graphs)
        return self.graphs
class UNSW_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000
        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/GCN/data/unsw/'
    
    def normalize_sym(self,adj):
        """原始归一化方式，对称归一化"""
        # adj = sp.coo_matrix(adj)
        adj=adj.to_dense()
        rowsum = np.array(1+adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        adj=sp.csr_matrix(torch.mm(adj, d_mat_inv_sqrt))
        coo = adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def gen_node_feats(self):
        file=self.data_path+'nodes_feats.pt'
        if os.path.exists(file):
            self.nodes_feats=torch.load(file)
        else:
            self.nodes_feats = torch.randn(self.ip_num,200) #从标准正态分布中随机采样
            torch.save(self.nodes_feats, self.data_path+'nodes_feats.pt')
        return self.nodes_feats
    
    def gen_graphs(self):
        data_path=self.data_path+'data.npy'
        if os.path.exists(data_path):
            self.graphs=np.load(data_path, allow_pickle=True)
            return self.graphs

        csv_list=glob.glob('/home/xiaoqing/gitpro/GNNPro/DyGCN/data/UNSWNB15/rawdata/'+'*.csv')
        cols = pd.read_csv('/home/xiaoqing/gitpro/GNNPro/DyGCN/data/UNSWNB15/NUSW-NB15_features.csv', index_col=None)['Name'].values.tolist()
        print("共发现%s个csv文件"%len(csv_list))
        i=0
        dataframe_list=[]
        for file in sorted(csv_list):
            print(file)
            df = pd.read_csv(file, encoding='unicode_escape')
            df.columns = cols
            df.loc[df['Label'] == 0, 'Label'] = 'BENIGN'
            df['is_ftp_login'] = df['is_ftp_login'].replace(np.nan, 0)
            df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace(np.nan, 0)
            df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].replace(np.nan, 0)
            # print(df[(df['srcip']==df['dstip'])])
            # df = df[-(df['srcip']==df['dstip'])]
            k=math.floor(len(df)/self.graph_len)
            dataframe_list.append(df.iloc[:k*self.graph_len,:])

        df = pd.concat(dataframe_list)
        df = df.loc[:, ['srcip', 'dstip','Label']]
        df.loc[df['Label']!='BENIGN', 'Label']=1
        df.loc[df['Label']=='BENIGN', 'Label']=0
        graph_labels =df['Label'].values.reshape(-1, self.graph_len)
        mal_nums=[Counter(gl)[1] for gl in graph_labels]
        labels=np.zeros_like(mal_nums)
        labels[np.array(mal_nums)>100]=1
        np.save(self.data_path+'labels_100.npy', labels.astype(np.int))

        ips=pd.concat([df['srcip'], df['dstip']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df['srcip'].values), le.transform(df['dstip'].values)
        adj_list=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices)
            sip, dip=le.transform(sip), le.transform(dip)
            adj=torch.sparse_coo_tensor([sip, dip], values=torch.ones(1000), size=(ip_num, ip_num))
            adj=self.normalize_sym(adj)
            graph={'adj':adj, 'node_idx':ip_indices}
            adj_list.append(graph)
        
        self.graphs=np.array(adj_list)
        np.save(data_path, self.graphs)
        return self.graphs

class USTC_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000
        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/GCN/data/ustc/'
    
    def normalize_sym(self,adj):
        """原始归一化方式，对称归一化"""
        # adj = sp.coo_matrix(adj)
        adj=adj.to_dense()
        rowsum = np.array(1+adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        adj=sp.csr_matrix(torch.mm(adj, d_mat_inv_sqrt))
        coo = adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def gen_node_feats(self):
        file=self.data_path+'nodes_feats.pt'
        if os.path.exists(file):
            self.nodes_feats=torch.load(file)
        else:
            self.nodes_feats = torch.randn(self.ip_num,200) #从标准正态分布中随机采样
            torch.save(self.nodes_feats, self.data_path+'nodes_feats.pt')
        return self.nodes_feats
    
    def gen_graphs(self):
        data_path=self.data_path+'data.npy'
        # if os.path.exists(data_path):
        #     self.graphs=np.load(data_path, allow_pickle=True)
        #     return self.graphs

        csv_list=glob.glob('/home/xiaoqing/gitpro/GNNPro/DyGCN/data/USTCTFC/rawdata/'+'*.csv')
        print("共发现%s个csv文件"%len(csv_list))
        i=0
        dataframe_list=[]
        for file in sorted(csv_list):
            print(file)
            df = pd.read_csv(file, encoding='unicode_escape')
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
            df = df[-(df['Src IP']==df['Dst IP'])] #删除源IP和目的IP相同的流
            k=math.floor(len(df)/self.graph_len)
            dataframe_list.append(df.iloc[:k*self.graph_len,:])
        df = pd.concat(dataframe_list)
        df = df.sort_values(by=['Timestamp'])
        df = df.loc[:, ['Src IP','Dst IP','Label']]
        df.loc[df['Label']!='BENIGN', 'Label']=1
        df.loc[df['Label']=='BENIGN', 'Label']=0
        graph_labels =df['Label'].values.reshape(-1, self.graph_len)
        mal_nums=[Counter(gl)[1] for gl in graph_labels]
        labels=np.zeros_like(mal_nums)
        labels[np.array(mal_nums)>100]=1
        np.save(self.data_path+'labels_100.npy', labels.astype(np.int))

        ips=pd.concat([df['Src IP'], df['Dst IP']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df['Src IP'].values), le.transform(df['Dst IP'].values)
        adj_list=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices)
            sip, dip=le.transform(sip), le.transform(dip)
            adj=torch.sparse_coo_tensor([sip, dip], values=torch.ones(1000), size=(ip_num, ip_num))
            adj=self.normalize_sym(adj)
            graph={'adj':adj, 'node_idx':ip_indices}
            adj_list.append(graph)
        
        self.graphs=np.array(adj_list)
        np.save(data_path, self.graphs)
        return self.graphs

class Cic2018_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000
        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/GCN/data/cic2018/'
    
    def normalize_sym(self,adj):
        """原始归一化方式，对称归一化"""
        # adj = sp.coo_matrix(adj)
        adj=adj.to_dense()
        rowsum = np.array(1+adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        adj=sp.csr_matrix(torch.mm(adj, d_mat_inv_sqrt))
        coo = adj.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def gen_node_feats(self):
        file=self.data_path+'nodes_feats.pt'
        if os.path.exists(file):
            self.nodes_feats=torch.load(file)
        else:
            self.nodes_feats = torch.randn(self.ip_num,200) #从标准正态分布中随机采样
            torch.save(self.nodes_feats, self.data_path+'nodes_feats.pt')
        return self.nodes_feats
    
    def gen_graphs(self):
        data_path=self.data_path+'data.npy'
        if os.path.exists(data_path):
            self.graphs=np.load(data_path, allow_pickle=True)
            return self.graphs

        file='/home/xiaoqing/gitpro/GNNPro/DyGCN/data/cicids2018/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'
        df = pd.read_csv(file, encoding='unicode_escape')
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
        # df = df[-(df['Src IP']==df['Dst IP'])] #删除源IP和目的IP相同的流
        k=math.floor(len(df)/self.graph_len)
        df=df.iloc[:k*self.graph_len,:]
        df = df.sort_values(by=['Timestamp'])
        df=df[400000:]
        df = df.loc[:, ['Src IP','Dst IP','Label']]
        df.loc[df['Label']!='Benign', 'Label']=1
        df.loc[df['Label']=='Benign', 'Label']=0
    
        graph_labels =df['Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1)
        np.save(self.data_path+'labels.npy', graph_labels.astype(np.int))
        # self.analysis(df, graph_labels)

        ips=pd.concat([df['Src IP'], df['Dst IP']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df['Src IP'].values), le.transform(df['Dst IP'].values)
        adj_list=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices)
            sip, dip=le.transform(sip), le.transform(dip)
            adj=torch.sparse_coo_tensor([sip, dip], values=torch.ones(1000), size=(ip_num, ip_num))
            adj=self.normalize_sym(adj)
            graph={'adj':adj, 'node_idx':ip_indices}
            adj_list.append(graph)
        
        self.graphs=np.array(adj_list)
        np.save(data_path, self.graphs)
        return self.graphs

    def gen_IPJson(self,src, dst, i):
        ips = list(Counter(src).keys())+list(Counter(dst).keys())
        ips = Counter(ips).keys() #节点的种类
        nodes = [] #记录图中的节点
        edges=[] #记录图中的边
        attack_ip, victim_ip=['52.14.136.135', '18.218.55.126', '18.218.229.235', '18.219.9.1', '18.216.200.189', '18.218.115.60', '18.219.5.43', '18.216.24.42', '18.218.11.51', '18.219.32.43'], ['172.31.69.25']#周二攻击者IP, 受害者IP
        # attack_ip, victim_ip=['205.174.165.73', '52.6.13.28'], ['192.168.10.15','192.168.10.8','192.168.10.9','192.168.10.14','192.168.10.5','192.168.10.12','192.168.10.17'] #周五攻击者IP，受害者IP
        for ip in ips:
            if ip in attack_ip:
                node = {'id':ip, 'name': 'Attacker', 'label': "Attacker"}
            elif ip in victim_ip:
                node = {'id':ip, 'name':'Victim', "label": "Victim"}
            else:
                node = {"id":ip, "name":"Normal"}
            nodes.append(node)
        src_nodes, dst_nodes = src, dst
        flows = [src_nodes[i]+"-"+dst_nodes[i] for i in range(len(src_nodes))]
        flows = dict(Counter(flows))
        for flow in flows:
            src,dst = flow.split('-')[0], flow.split('-')[1]
            edge = {"source":src, "target":dst, 'weight':flows[flow]} #边的权重表示多重边的数量
            edges.append(edge)
        kk = {"nodes":nodes, "edges":edges}
        f = open('json/'+str(i)+'_ip_graph.json', 'w')
        b = json.dumps(kk, separators=(',',':'), indent=4)
        f.write(b)
        f.close()
        # print("write IP graph json complete")
    def analysis(self, gs, labels):
       
        """
        将csv特征文件按照时间顺序切割成dgl子图：IP作为节点生成同质图，并写入文件
        """
        def knkn(gs):
            gs=gs.reshape(-1,3)
            p=Counter(gs[:,1][gs[:,2]==1])
            t=list(dict(p).keys())
            t.sort()
            print(t)
        # print(np.where(labels)>0)
        graphs=gs.values.reshape(-1,self.graph_len,3)
        mal_graphs=graphs[labels==1]
        knkn(mal_graphs)

        for i in tqdm(range(4000, 5000)):
            if labels[i]==0:
                print(Counter(graphs[i][:,1][graphs[i][:,2]==1]))
            g_data =graphs[i]
            self.gen_IPJson(g_data[:,0],g_data[:,1], i)

