import datetime
import glob
import math
import os
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

import utils as u

cur_path=os.path.dirname(__file__)
data_dir=os.path.dirname((cur_path))
os.chdir(cur_path)

class Dataset():
    def __init__(self, dataset='cic'):
        self.graph_len=1000
        self.dataset=dataset
        self.label_types_file=os.path.join('data',self.dataset,'label_types.npy')
        self.label_file=os.path.join('data',self.dataset,'labels.npy')
        self.edge_file=os.path.join('data',self.dataset,'edges.npy')
        self.feat_file=os.path.join('data',self.dataset,'feats.npy')
        self.edge_list, self.feat_list=self.load_data()


    def normalize_random(self,adj):
        """随机游走归一化"""
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(1+adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -1).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        return adj
    
    def normalize_sym(self,adj):
        """原始归一化方式，对称归一化"""
        # adj = sp.coo_matrix(adj)
        rowsum = np.array(1+adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = torch.diag(torch.from_numpy(d_inv_sqrt))
        adj=torch.mm(d_mat_inv_sqrt, adj)
        return torch.mm(adj, d_mat_inv_sqrt)

    def gen_graphs(self):
        le = LabelEncoder()
        ip_list=[]
        Aout_list=[]
        Ain_list=[]
        A_list=[]
        e_list=[]
        for i in tqdm(range(len(self.edge_list))):
            edges = le.fit_transform(self.edge_list[i].reshape(-1))
            ip_list.append(le.classes_.tolist())
            edges=edges.reshape(-1,2)
            n = len(le.classes_)
            A_out = torch.sparse_coo_tensor([edges[:,0],range(self.graph_len)], torch.ones(self.graph_len), size=[n, self.graph_len])
            A_in = torch.sparse_coo_tensor([edges[:,1],range(self.graph_len)], torch.ones(self.graph_len), size=[n, self.graph_len])
            adj = torch.sparse_coo_tensor(edges.T, torch.ones(self.graph_len),size=[n, n])
            
            adj = self.normalize_sym(torch.eye(adj.shape[0])+adj)#无向图进行对称归一化
            A_out=self.normalize_random(A_out.to_dense())#有向图进行随机游走归一化
            A_in=self.normalize_random(A_in.to_dense())#有向图进行随机游走归一化
            Aout_list.append(A_out.to_sparse())#源IP-Flow邻接矩阵
            Ain_list.append(A_in.to_sparse()) #目的IP-Flow邻接矩阵
            A_list.append(adj.to_sparse()) # ip节点邻接矩阵
            e_list.append(edges)
        
        return {
            'Ain_list':Ain_list,
            'Aout_list': Aout_list,
            'A_list':A_list,
            'feat_list':self.feat_list,
            'ip_list': ip_list,
            'edges':e_list
        }
    
    def load_data(self):
        if os.path.exists(self.edge_file):
            edge_list = np.load(self.edge_file, allow_pickle=True)
            feat_list=np.load(self.feat_file, allow_pickle=True)
            return edge_list,feat_list
        if self.dataset=='cic':
            self.cic()
        elif self.dataset=='unsw':
            self.unsw()
        elif self.dataset=='ustc':
            self.ustc()
        elif self.dataset=='cic2018':
            self.cic2018()
    
    def cic(self):
        csv_list=glob.glob(data_dir+'/data/CIC2017/rawdata/'+'*.csv')
        print("共发现%s个csv文件"%len(csv_list))
        dataframe_list=[]
        for file in csv_list:
            print(file)
            df=pd.read_csv(file, index_col=None, encoding='unicode_escape')
            print(len(df))
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
            k=math.floor(len(df)/self.graph_len)
            #按照时间戳进行排序，原始的时间戳格式不统一，需要先统一时间戳格式
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
        self.edges=df.loc[:, [' Source IP', ' Destination IP']].values.reshape(-1,self.graph_len,2)
        # 筛选流特征
        feats=df.drop(columns=['Flow ID', ' Timestamp', ' Fwd Header Length.1', ' Source IP', ' Source Port', ' Destination IP', ' Destination Port', ' Label'])
        # 流特征进行线性归一化
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        self.feat_list=feats.reshape(-1, self.graph_len, feats.shape[1])

        self.mal_types=df[' Label'].values.reshape(-1, self.graph_len)
        np.save(self.label_types_file, self.mal_types)

        # 字符型特征转换为数字，异常检测只区分是否恶意，正常标记为0，异常标记为1
        df.loc[df[' Label']!='BENIGN', ' Label']=1
        df.loc[df[' Label']=='BENIGN', ' Label']=0
        graph_labels =df[' Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1) # 通信图的标签根据图内是否含有异常流判断，包含：1
        
        # 保存预处理后的结果：图标签、有向边、边特征
        np.save(self.label_file, graph_labels)
        np.save(self.edge_file, self.edges)
        np.save(self.feat_file,self.feat_list)

    def cic2018(self):
        file=data_dir+'/data/cicids2018/Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv'
        df = pd.read_csv(file, encoding='unicode_escape')
        df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
        # df = df[-(df['Src IP']==df['Dst IP'])] #删除源IP和目的IP相同的流
        k=math.floor(len(df)/self.graph_len)
        df=df.iloc[:k*self.graph_len,:]
        df = df.sort_values(by=['Timestamp'])
        df=df[400000:] # 忽略前40万条流，因为前40万流中包含异常流，我们基于时序进行异常检测，要求连续时间内的流都是正常的
        
        feats=df.drop(columns=['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'Label'])
        
        self.mal_types=df['Label'].values.reshape(-1, self.graph_len)
        np.save(self.label_types_file, self.mal_types)
        df.loc[df['Label']!='Benign', 'Label']=1
        df.loc[df['Label']=='Benign', 'Label']=0
        np.save('flow_labels.npy', df['Label'].values)
        mm = MinMaxScaler()
        feats=mm.fit_transform(feats)
        graph_labels =df['Label'].values.reshape(-1, self.graph_len)
        graph_labels = np.max(graph_labels, axis=1)

        df = df.loc[:, ['Src IP','Dst IP']]
        self.edges=df.values.reshape(-1,self.graph_len,2)
        self.feat_list=feats.reshape(-1, self.graph_len, feats.shape[1])

        np.save(self.label_file, graph_labels)
        np.save(self.edge_file, self.edges)
        np.save(self.feat_file,self.feat_list)


    # def unsw(self):
    #     # 没有用到这个数据集
    #     csv_list=glob.glob(data_dir+'/data/UNSWNB15/rawdata/'+'*.csv')
    #     cols = pd.read_csv(data_dir+'/data/UNSWNB15/NUSW-NB15_features.csv', index_col=None)['Name'].values.tolist()
    #     print("共发现%s个csv文件"%len(csv_list))
    #     dataframe_list=[]
    #     for file in sorted(csv_list):
    #         print(file)
    #         df = pd.read_csv(file, encoding='unicode_escape')
    #         df.columns = cols
    #         df.loc[df['Label'] == 0, 'Label'] = 'BENIGN'
    #         df['is_ftp_login'] = df['is_ftp_login'].replace(np.nan, 0)
    #         df['ct_ftp_cmd'] = df['ct_ftp_cmd'].replace(np.nan, 0)
    #         df['ct_flw_http_mthd'] = df['ct_flw_http_mthd'].replace(np.nan, 0)
    #         # df = df[-(df['srcip']==df['dstip'])]
    #         k=math.floor(len(df)/self.graph_len)
    #         dataframe_list.append(df.iloc[:k*self.graph_len,:])
    #     df = pd.concat(dataframe_list)
    #     df = pd.get_dummies(data=df, columns=['proto', 'service', 'state'])
    #     feats=df.drop(columns=['srcip', 'sport', 'dstip', 'dsport', 'Stime', 'Ltime', 'Label', 'attack_cat'])
        
    #     df = df.loc[:, ['srcip', 'dstip','Label']]
    #     self.mal_types=df['Label'].values.reshape(-1, self.graph_len)
    #     np.save(self.label_types_file, self.mal_types)
    #     df.loc[df['Label']!='BENIGN', 'Label']=1
    #     df.loc[df['Label']=='BENIGN', 'Label']=0
    #     mm = MinMaxScaler()
    #     feats=mm.fit_transform(feats)
    #     graph_labels =df['Label'].values.reshape(-1, self.graph_len)
    #     graph_labels = np.max(graph_labels, axis=1)

    #     self.edges=df.values.reshape(-1,self.graph_len,3)
    #     self.feat_list=feats.reshape(-1, self.graph_len, feats.shape[1])

    #     np.save(self.label_file, graph_labels)
    #     np.save(self.edge_file, self.edges)
    #     np.save(self.feat_file,self.feat_list)
    #     Counter(graph_labels)

    # def ustc(self):
    #     # 没有用到这个数据集
    #     csv_list=glob.glob(data_dir+'/data/USTCTFC/rawdata/'+'*.csv')
    #     print("共发现%s个csv文件"%len(csv_list))
    #     dataframe_list=[]
    #     for file in sorted(csv_list):
    #         print(file)
    #         df = pd.read_csv(file, encoding='unicode_escape')
    #         df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
    #         df = df[-(df['Src IP']==df['Dst IP'])] #删除源IP和目的IP相同的流
    #         k=math.floor(len(df)/self.graph_len)
    #         dataframe_list.append(df.iloc[:k*self.graph_len,:])
    #     df = pd.concat(dataframe_list)
    #     df = df.sort_values(by=['Timestamp'])
    #     feats=df.drop(columns=['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Timestamp', 'Label'])
    #     df = df.loc[:, ['Src IP','Dst IP','Label']]
    #     self.mal_types=df['Label'].values.reshape(-1, self.graph_len)
    #     np.save(self.label_types_file, self.mal_types)
    #     df.loc[df['Label']!='BENIGN', 'Label']=1
    #     df.loc[df['Label']=='BENIGN', 'Label']=0
    #     mm = MinMaxScaler()
    #     feats=mm.fit_transform(feats)
    #     graph_labels =df['Label'].values.reshape(-1, self.graph_len)
    #     graph_labels = np.max(graph_labels, axis=1)

    #     self.edges=df.values.reshape(-1,self.graph_len,2)
    #     self.feat_list=feats.reshape(-1, self.graph_len, feats.shape[1])

    #     np.save(self.label_file, graph_labels)
    #     np.save(self.edge_file, self.edges)
    #     np.save(self.feat_file,self.feat_list)
    #     Counter(graph_labels)

    