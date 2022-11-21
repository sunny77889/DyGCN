import datetime
import glob
import math
import os
import random
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)

class Cic_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000
        self.train_len=529

        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/DyDGI/data/cic/'
        self.edata_path=self.data_path
    
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
    def gen_edge_feats(self):
        if os.path.exists(self.edata_path+'cic_feats.npy'):
            self.edge_feats=np.load(self.edata_path+'cic_feats.npy', allow_pickle=True)
            return self.edge_feats
        return None

    def gen_graphs(self):
        vdata_path=self.data_path+'data.npy'
        edata_path=self.edata_path+'eadj.npy'
        if os.path.exists(vdata_path) and os.path.exists(edata_path):
            self.ip_graphs=np.load(vdata_path, allow_pickle=True)
            self.edge_graphs=np.load(edata_path, allow_pickle=True)
            return self.ip_graphs, self.edge_graphs

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
        feats=df.drop(columns=['Flow ID', ' Timestamp', ' Fwd Header Length.1', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Label'])
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        np.save(self.edata_path+'cic_feats.npy',feats.reshape(-1, self.graph_len, feats.shape[1]))

        df = df.loc[:, [' Source IP', ' Destination IP',' Label']]
        df.loc[df[' Label']!='BENIGN', ' Label']=1
        df.loc[df[' Label']=='BENIGN', ' Label']=0
        graph_labels =df[' Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1)
        np.save(self.edata_path+'labels.npy', graph_labels.astype(np.int))

        ips=pd.concat([df[' Source IP'], df[' Destination IP']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df[' Source IP'].values), le.transform(df[' Destination IP'].values)
        adj_list=[]
        edge_graphs=[]
        for i in tqdm(range(0,len(src), self.graph_len)):      
            sip,dip=src[i:i+self.graph_len], dst[i:i+self.graph_len]
            ip_indices=np.unique(np.concatenate([sip, dip]))  #图中节点在整个数据集中的索引：后续会用以求特征矩阵
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices) #重新编码节点特征：当前时刻的图只是一个小图，所以我们重新定义邻接矩阵
            sip, dip=le.transform(sip), le.transform(dip)
            adj=torch.sparse_coo_tensor([sip, dip], values=torch.ones(self.graph_len), size=(ip_num, ip_num))
            adj_norm=self.normalize_sym(adj) #邻接矩阵归一化

            ip_graph={'adj':adj_norm, 'node_idx':ip_indices}
            adj_list.append(ip_graph)
            #生成线图
            graph=pd.DataFrame({'sip':sip, 'dip':dip, 'eid':[i for i in range(self.graph_len)]})
            sg=graph.groupby('sip')
            dg=graph.groupby('dip')
            se_edge=self.gen_edges(sg)
            de_edge=self.gen_edges(dg)
            e_edge=np.concatenate([se_edge, de_edge])

            e_adj=torch.sparse_coo_tensor([e_edge[:,0], e_edge[:,1]], values=torch.ones(len(e_edge)), size=(self.graph_len, self.graph_len))
            e_adj_norm=self.normalize_sym(e_adj)
            edge_graphs.append(e_adj_norm)

        self.eadj=np.array(edge_graphs)
        self.graphs=np.array(adj_list)
        np.save(edata_path,self.eadj)
        np.save(vdata_path, self.graphs)
        return self.graphs, self.eadj
    def gen_edges(self,gps):
        e_edge=[]
        for name, group in gps:
            eid=group['eid'].values
            n=len(eid)
            for i in range(n):
                for j in range(i+1,n):
                    e_edge.append([eid[i], eid[j]])
                    e_edge.append([eid[j], eid[i]])
        e_edge=np.array(e_edge)
        return np.array(e_edge)


class UNSW_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000

        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/DyDGI/data/unsw/'
        self.efeat_path=self.data_path+'feats.npy'
    
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
            torch.save(self.nodes_feats, file)
        sc=MinMaxScaler()
        node_feats= sc.fit_transform(self.nodes_feats)
        return torch.from_numpy(node_feats).to(torch.float32)
    
    def gen_edge_feats(self):
        if os.path.exists(self.efeat_path):
            self.edge_feats=np.load(self.efeat_path, allow_pickle=True)
            return self.edge_feats
        return None

    def gen_graphs(self):
        vdata_path=self.data_path+'data.npy'
        edata_path=self.data_path+'eadj.npy'
        if os.path.exists(vdata_path) and os.path.exists(edata_path):
            self.ip_graphs=np.load(vdata_path, allow_pickle=True)
            self.edge_graphs=np.load(edata_path, allow_pickle=True)
            return self.ip_graphs, self.edge_graphs
        
        csv_list=glob.glob('/home/xiaoqing/gitpro/GNNPro/DyGCN/data/UNSWNB15/rawdata/'+'*.csv')
        cols = pd.read_csv('/home/xiaoqing/gitpro/GNNPro/DyGCN/data/UNSWNB15/NUSW-NB15_features.csv', index_col=None)['Name'].values.tolist()
        print("共发现%s个csv文件"%len(csv_list))
        dataframe_list=[]
        for file in sorted(csv_list):
            print(file)
            df = pd.read_csv(file, encoding='unicode_escape')
            df.columns = cols
            df.loc[df['Label'] == 0, 'Label'] = 'BENIGN'
            df['is_ftp_login'] = df['is_ftp_login'].replace(np.nan, 0)
            df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace(np.nan, 0)
            df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].replace(np.nan, 0)
            # df = df[-(df['srcip']==df['dstip'])]
            k=math.floor(len(df)/self.graph_len)
            dataframe_list.append(df.iloc[:k*self.graph_len,:])

        df = pd.concat(dataframe_list)
        df = pd.get_dummies(data=df, columns=['proto', 'service', 'state'])
        feats=df.drop(columns=['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'Label', 'attack_cat'])
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        np.save(self.data_path+'feats.npy',feats.reshape(-1, self.graph_len, feats.shape[1]))

        df = df.loc[:, ['srcip', 'dstip','Label']]
        df.loc[df['Label']!='BENIGN', 'Label']=1
        df.loc[df['Label']=='BENIGN', 'Label']=0

        graph_labels =df['Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1)
        np.save(self.data_path+'labels.npy', graph_labels.astype(np.int))

        ips=pd.concat([df['srcip'], df['dstip']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df['srcip'].values), le.transform(df['dstip'].values)
        adj_list=[]
        edge_graphs=[]
        for i in tqdm(range(0,len(src), self.graph_len)):      
            sip,dip=src[i:i+self.graph_len], dst[i:i+self.graph_len]
            ip_indices=np.unique(np.concatenate([sip, dip]))  #图中节点在整个数据集中的索引：后续会用以求特征矩阵
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices) #重新编码节点特征：当前时刻的图只是一个小图，所以我们重新定义邻接矩阵
            sip, dip=le.transform(sip), le.transform(dip)
            adj=torch.sparse_coo_tensor([sip, dip], values=torch.ones(self.graph_len), size=(ip_num, ip_num))
            adj_norm=self.normalize_sym(adj) #邻接矩阵归一化

            ip_graph={'adj':adj_norm, 'node_idx':ip_indices}
            adj_list.append(ip_graph)
            #生成线图
            graph=pd.DataFrame({'sip':sip, 'dip':dip, 'eid':[i for i in range(self.graph_len)]})
            sg=graph.groupby('sip')
            dg=graph.groupby('dip')
            se_edge=self.gen_edges(sg)
            de_edge=self.gen_edges(dg)
            e_edge=np.concatenate([se_edge, de_edge])

            e_adj=torch.sparse_coo_tensor([e_edge[:,0], e_edge[:,1]], values=torch.ones(len(e_edge)), size=(self.graph_len, self.graph_len))
            e_adj_norm=self.normalize_sym(e_adj)
            edge_graphs.append(e_adj_norm)

        # self.eadj=torch.stack(edge_graphs)
        self.eadj=edge_graphs
        self.graphs=np.array(adj_list)
        np.save(edata_path,self.eadj)
        np.save(vdata_path, self.graphs)
        return self.graphs, self.eadj
    def gen_edges(self,gps):
        e_edge=[]
        for name, group in gps:
            eid=group['eid'].values
            n=len(eid)
            for i in range(n):
                for j in range(i+1,n):
                    e_edge.append([eid[i], eid[j]])
                    e_edge.append([eid[j], eid[i]])
        e_edge=np.array(e_edge)
        return np.array(e_edge)

class USTC_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000

        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/DyDGI/data/ustc/'
        self.efeat_path=self.data_path+'feats.npy'
    
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
            torch.save(self.nodes_feats, file)
        sc=MinMaxScaler()
        node_feats= sc.fit_transform(self.nodes_feats)
        return torch.from_numpy(node_feats).to(torch.float32)
    
    def gen_edge_feats(self):
        if os.path.exists(self.efeat_path):
            self.edge_feats=np.load(self.efeat_path, allow_pickle=True)
            return self.edge_feats
        return None

    def gen_graphs(self):
        vdata_path=self.data_path+'data.npy'
        edata_path=self.data_path+'eadj.pt'
        if os.path.exists(vdata_path) and os.path.exists(edata_path):
            self.ip_graphs=np.load(vdata_path, allow_pickle=True)
            self.edge_graphs=torch.load(edata_path)
            return self.ip_graphs, self.edge_graphs
        
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

        feats=df.drop(columns=['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'Label'])
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        np.save(self.data_path+'feats.npy',feats.reshape(-1, self.graph_len, feats.shape[1]))

        df = df.loc[:, ['Src IP','Dst IP','Label']]
        df.loc[df['Label']!='BENIGN', 'Label']=1
        df.loc[df['Label']=='BENIGN', 'Label']=0

        graph_labels =df['Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1)
        np.save(self.data_path+'labels.npy', graph_labels.astype(np.int))

        ips=pd.concat([df['Src IP'], df['Dst IP']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df['Src IP'].values), le.transform(df['Dst IP'].values)
        adj_list=[]
        edge_graphs=[]
        for i in tqdm(range(0,len(src), self.graph_len)):      
            sip,dip=src[i:i+self.graph_len], dst[i:i+self.graph_len]
            ip_indices=np.unique(np.concatenate([sip, dip]))  #图中节点在整个数据集中的索引：后续会用以求特征矩阵
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices) #重新编码节点特征：当前时刻的图只是一个小图，所以我们重新定义邻接矩阵
            sip, dip=le.transform(sip), le.transform(dip)
            adj=torch.sparse_coo_tensor([sip, dip], values=torch.ones(self.graph_len), size=(ip_num, ip_num))
            adj_norm=self.normalize_sym(adj) #邻接矩阵归一化

            ip_graph={'adj':adj_norm, 'node_idx':ip_indices}
            adj_list.append(ip_graph)
            #生成线图
            graph=pd.DataFrame({'sip':sip, 'dip':dip, 'eid':[i for i in range(self.graph_len)]})
            sg=graph.groupby('sip')
            dg=graph.groupby('dip')
            se_edge=self.gen_edges(sg)
            de_edge=self.gen_edges(dg)
            if len(de_edge)>0:
                e_edge=np.concatenate([se_edge, de_edge])

            e_adj=torch.sparse_coo_tensor([e_edge[:,0], e_edge[:,1]], values=torch.ones(len(e_edge)), size=(self.graph_len, self.graph_len))
            e_adj_norm=self.normalize_sym(e_adj)
            edge_graphs.append(e_adj_norm)

        self.eadj=torch.stack(edge_graphs)
        self.graphs=np.array(adj_list)
        torch.save(self.eadj, edata_path)
        np.save(vdata_path, self.graphs)
        return self.graphs, self.eadj
    def gen_edges(self,gps):
        e_edge=[]
        for name, group in gps:
            eid=group['eid'].values
            n=len(eid)
            for i in range(n):
                for j in range(i+1,n):
                    e_edge.append([eid[i], eid[j]])
                    e_edge.append([eid[j], eid[i]])
        e_edge=np.array(e_edge)
        return np.array(e_edge)

class Cic2018_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000

        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/DyDGI/data/cic2018/'
        self.efeat_path=self.data_path+'feats.npy'
    
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
            torch.save(self.nodes_feats, file)
        sc=MinMaxScaler()
        node_feats= sc.fit_transform(self.nodes_feats)
        return torch.from_numpy(node_feats).to(torch.float32)
    
    def gen_edge_feats(self):
        if os.path.exists(self.efeat_path):
            self.edge_feats=np.load(self.efeat_path, allow_pickle=True)
            return self.edge_feats
        return None

    def gen_graphs(self):
        vdata_path=self.data_path+'data.npy'
        edata_path=self.data_path+'eadj.pt.npy'
        if os.path.exists(vdata_path) and os.path.exists(edata_path):
            self.ip_graphs=np.load(vdata_path, allow_pickle=True)
            self.edge_graphs=np.load(edata_path, allow_pickle=True)
            return self.ip_graphs, self.edge_graphs
        
        file='/home/xiaoqing/gitpro/GNNPro/DyGCN/data/cicids2018/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'
        df = pd.read_csv(file, encoding='unicode_escape')
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
        # df = df[-(df['Src IP']==df['Dst IP'])] #删除源IP和目的IP相同的流
        k=math.floor(len(df)/self.graph_len)
        df=df.iloc[:k*self.graph_len,:]
        df = df.sort_values(by=['Timestamp'])
        df=df[400000:]

        feats=df.drop(columns=['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'Label'])
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        np.save(self.data_path+'feats.npy',feats.reshape(-1, self.graph_len, feats.shape[1]))

        df = df.loc[:, ['Src IP','Dst IP','Label']]
        df.loc[df['Label']!='Benign', 'Label']=1
        df.loc[df['Label']=='Benign', 'Label']=0

        graph_labels =df['Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1)
        np.save(self.data_path+'labels.npy', graph_labels.astype(np.int))

        ips=pd.concat([df['Src IP'], df['Dst IP']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df['Src IP'].values), le.transform(df['Dst IP'].values)
        adj_list=[]
        edge_graphs=[]
        for i in tqdm(range(0,len(src), self.graph_len)):      
            sip,dip=src[i:i+self.graph_len], dst[i:i+self.graph_len]
            ip_indices=np.unique(np.concatenate([sip, dip]))  #图中节点在整个数据集中的索引：后续会用以求特征矩阵
            ip_num=ip_indices.shape[0]
            le=LabelEncoder().fit(ip_indices) #重新编码节点特征：当前时刻的图只是一个小图，所以我们重新定义邻接矩阵
            sip, dip=le.transform(sip), le.transform(dip)
            adj=torch.sparse_coo_tensor([sip, dip], values=torch.ones(self.graph_len), size=(ip_num, ip_num))
            adj_norm=self.normalize_sym(adj) #邻接矩阵归一化

            ip_graph={'adj':adj_norm, 'node_idx':ip_indices}
            adj_list.append(ip_graph)
            #生成线图
            graph=pd.DataFrame({'sip':sip, 'dip':dip, 'eid':[i for i in range(self.graph_len)]})
            sg=graph.groupby('sip')
            dg=graph.groupby('dip')
            se_edge=self.gen_edges(sg)
            de_edge=self.gen_edges(dg)
            e_edge=np.concatenate([se_edge, de_edge])

            e_adj=torch.sparse_coo_tensor([e_edge[:,0], e_edge[:,1]], values=torch.ones(len(e_edge)), size=(self.graph_len, self.graph_len))
            e_adj_norm=self.normalize_sym(e_adj)
            edge_graphs.append(e_adj_norm)

        # self.eadj=torch.stack(edge_graphs)
        self.eadj=np.array(edge_graphs)
        self.graphs=np.array(adj_list)
        np.save(edata_path, self.eadj)
        np.save(vdata_path, self.graphs)
        return self.graphs, self.eadj
    
    def gen_edges(self,gps):
        e_edge=[]
        for name, group in gps:
            eid=group['eid'].values
            n=len(eid)
            for i in range(n):
                for j in range(i+1,n):
                    e_edge.append([eid[i], eid[j]])
                    e_edge.append([eid[j], eid[i]])
        e_edge=np.array(e_edge)
        return np.array(e_edge)
