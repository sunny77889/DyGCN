import datetime
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

os.chdir(sys.path[0])
TIME_INTERVAL=1000

def gen_A_X(g_data):
    '''生成快照的IP-Flow矩阵和特征矩阵'''
    le =LabelEncoder()
    mms = MinMaxScaler()
    dst_ips = g_data[' Destination IP'].values
    src_ips = g_data[' Source IP'].values
    n = len(g_data)
    labels = g_data[g_data[' Label']!='BENIGN']
    label = 0 # 图的标记：存在异常边就将该图标记为异常
    if len(labels)>0:
        label=1
    ips  = np.concatenate((dst_ips, src_ips), axis=0)
    data = np.ones(n, dtype=int)
    le = le.fit(ips)
    ip_nums = len(le.classes_)
    dst_ips = le.transform(dst_ips)
    src_ips = le.transform(src_ips)
    flow_ids=np.arange(0,n, step=1, dtype=int)
    
    # IP-Flow 关联矩阵, 行表示目的IP， 列表示特征
    dst_ifa = sparse.csr_matrix((data, (dst_ips, flow_ids)),shape=(ip_nums, n))
    # IP-Flow 关联矩阵, 行表示目的IP， 列表示特征
    src_ifa = sparse.csc_matrix((data, (src_ips, flow_ids)), shape=(ip_nums, n))
    # IP-Flow 关联矩阵, 行表示IP， 列表示特征
    ifa = src_ifa+dst_ifa
    df=g_data.drop(columns = ['Flow ID', ' Fwd Header Length.1',' Source IP', ' Source Port', ' Destination IP',' Destination Port', ' Timestamp', ' Label'])
    X = mms.fit_transform(df)#进行归一化
    #有向边，将源IP到目的IP之间的所有边求平均得到均值特征，并把边的数目作为新边的权重
    X = pd.DataFrame(X)
    X['ip_pairs']=[str(src_ips[i])+'-'+str(dst_ips[i]) for i in range(len(src_ips))]
    ip_pairs=X.groupby(X['ip_pairs'])
    ip_pairs_fea = ip_pairs.agg('mean') #均值特征
    weights=[len(d) for d in ip_pairs.indices.values()] #新边权重
    ip_adj = sparse.csr_matrix( (data, (src_ips, dst_ips)) , shape=(ip_nums, ip_nums)) 
    return src_ifa, dst_ifa, ifa, X.drop(columns=['ip_pairs']).values, ip_adj, label, ip_pairs_fea, weights, src_ips, dst_ips
from datetime import timedelta


def graph_split(data_path,typ):
    """
    将csv特征文件按照时间顺序切割成dgl子图：IP作为节点生成同质图，并写入文件
    """
    df  = pd.read_csv(data_path, index_col=None)
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)].dropna()  # 删除异常记录
    #按照时间戳进行排序，原始的时间戳有问题
    td = timedelta(hours=12)
    aa=datetime.datetime.strptime(df[' Timestamp'][0].split(' ')[0]+' 8:00:00',"%d/%m/%Y %H:%M:%S")
    if data_path.split('/')[-1].split('-')[0] != '1Monday':
        df[' Timestamp']=[datetime.datetime.strptime(t, "%d/%m/%Y %H:%M") for t in df[' Timestamp']] #先将字符串类型的时间戳转换为时间戳
    else:
        df[' Timestamp']=[datetime.datetime.strptime(t, "%d/%m/%Y %H:%M:%S") for t in df[' Timestamp']] #先将字符串类型的时间戳转换为时间戳
    
    df[' Timestamp']=df[' Timestamp'].apply(lambda x:x if x>aa else x+td) #把下午的时间改为24小时，1点改为13点
    df = df.sort_values(by=[' Timestamp']) #首先按照时间戳排序
    
    graphs = []
    for i in tqdm(range(len(df))):
        if i==0:
            continue
        data={}
        if (i%TIME_INTERVAL)==0: #每1000条流构建一张图
        # if (df[' Timestamp'].iloc[i]-start_time).seconds>=60: #每分钟构建一张图
            g_data = df.iloc[i-TIME_INTERVAL:i, :]
            # g_data = df.iloc[start_idx:i, :]
            src_ifa, dst_ifa,ifa, X, adj, label, ip_pairs_fea, weights, src, dst =gen_A_X(g_data)
            data['srcIP_Flow_adj']=src_ifa #源节点-Flow ID之间的关联矩阵
            data['dstIP_Flow_adj']=dst_ifa #目的节点-Flow ID之间的关联矩阵
            data['IP_Flow_adj']=ifa
            data['X']=X # Flow的特征矩阵
            data['IP_adj']=adj # 图中所有节点之间的邻接矩阵
            data['label']=label # 图的标签
            data['ip_pairs_fea']=ip_pairs_fea #源IP到目的IP之间的所有边求均值
            data['weights'] = weights #源IP到目的IP之间的所有边的数目
            data['src']=src
            data['dst']=dst
            data['flow_labels']=g_data[' Label'].values
            graphs.append(data)
    print(len(graphs))
    np.save(typ+'_IP_graphs.npy', np.array(graphs))

data_path = "/home/ypd-23-teacher-2/xiaoqing/GNNPro/data/CIC2017/rawdata/8Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

graph_split(data_path, 'test5_3')
