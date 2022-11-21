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
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/CTGCN/data/cic/'

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
    def gen_graphs(self):

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
        j=0
        ip_idx=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            le=LabelEncoder().fit(ip_indices)
            s,d=le.transform(sip), le.transform(dip)
            fd=pd.DataFrame({'from_id':s, 'to_id':d, 'weight':np.ones(len(s), dtype=int)})
            fd.to_csv(self.data_path+'1.format/1-'+str(j)+'.csv', index=False, sep='\t')
            ip_idx.append(ip_indices)
            j+=1
        np.save(self.data_path+'ip_indices.npy', ip_idx)

class UNSW_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000

        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/CTGCN/data/unsw/'
    
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
    
    def gen_graphs(self):

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
        j=0
        ip_idx=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            le=LabelEncoder().fit(ip_indices)
            s,d=le.transform(sip), le.transform(dip)
            fd=pd.DataFrame({'from_id':s, 'to_id':d, 'weight':np.ones(len(s), dtype=int)})
            fd.to_csv(self.data_path+'1.format/1-'+str(j)+'.csv', index=False, sep='\t')
            ip_idx.append(ip_indices)
            j+=1
        np.save(self.data_path+'ip_indices.npy', ip_idx)

class USTC_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000

        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/CTGCN/data/ustc/'
    
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
    
    def gen_graphs(self):

        csv_list=glob.glob('/home/xiaoqing/gitpro/GNNPro/DyGCN/data/USTCTFC/rawdata/'+'*.csv')
        print("共发现%s个csv文件"%len(csv_list))
        i=0
        dataframe_list=[]
        for file in sorted(csv_list):
            print(file)
            df = pd.read_csv(file, encoding='unicode_escape')
            df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
            # df = df[-(df['Src IP']==df['Dst IP'])] #删除源IP和目的IP相同的流
            k=math.floor(len(df)/self.graph_len)
            dataframe_list.append(df.iloc[:k*self.graph_len,:])
        df = pd.concat(dataframe_list)
        df = df.sort_values(by=['Timestamp'])

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
        j=0
        ip_idx=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            le=LabelEncoder().fit(ip_indices)
            s,d=le.transform(sip), le.transform(dip)
            fd=pd.DataFrame({'from_id':s, 'to_id':d, 'weight':np.ones(len(s), dtype=int)})
            fd.to_csv(self.data_path+'1.format/1-'+str(j)+'.csv', index=False, sep='\t')
            ip_idx.append(ip_indices)
            j+=1
        np.save(self.data_path+'ip_indices.npy', ip_idx)

class Cic2018_Dataset():
    def __init__(self):
        #build edge data structure
        self.graph_len=1000

        self.adj_list=[]
        self.label_list=[]
        self.A_list=[]
        self.data_path='/home/xiaoqing/gitpro/GNNPro/DyGCN/compare_models/CTGCN/data/cic2018/'
    
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
    
    def gen_graphs(self):

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

        ips=pd.concat([df['Src IP'], df['Dst IP']]).unique()
        self.ip_num=ips.shape[0]
        self.gen_node_feats()
        le=LabelEncoder().fit(ips)
        
        src, dst=le.transform(df['Src IP'].values), le.transform(df['Dst IP'].values)
        j=0
        ip_idx=[]
        for i in range(0,len(src), 1000):      
            sip,dip=src[i:i+1000], dst[i:i+1000]
            ip_indices=np.unique(np.concatenate([sip, dip]))
            le=LabelEncoder().fit(ip_indices)
            s,d=le.transform(sip), le.transform(dip)
            fd=pd.DataFrame({'from_id':s, 'to_id':d, 'weight':np.ones(len(s), dtype=int)})
            fd.to_csv(self.data_path+'1.format/1-'+str(j)+'.csv', index=False, sep='\t')
            ip_idx.append(ip_indices)
            j+=1
        np.save(self.data_path+'ip_indices.npy', ip_idx)

# Cic_Dataset().gen_graphs()
