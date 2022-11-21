import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from pyod.models.hbos import HBOS
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.svm import OneClassSVM

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)  
datapath='GCN/data/cic2018/'
def parse():
    parser=argparse.ArgumentParser()
    parser.add_argument("--embs_path", default=datapath+'graph_embs.pt', type=str)
    parser.add_argument('--labels_path', default=datapath+"labels.npy", type=str)
    return parser.parse_args()


def plot_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    print('异常阈值', threshold)
    auc_score = auc(fpr, tpr)
    print('auc值: {:.4f}'.format(auc_score))
    return threshold, auc_score

def eval(labels,pred):
    plot_roc(labels, pred)
    print(confusion_matrix(labels, pred))
    a,b,c,d=accuracy_score(labels, pred),precision_score(labels, pred),recall_score(labels, pred), f1_score(labels, pred)
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a,b,c,d))
    return a,b,c,d

def matrix(true_graph_labels,scores):
    t, auc=plot_roc(true_graph_labels, scores)
    true_graph_labels = np.array(true_graph_labels)
    scores  = np.array(scores)
    pred=np.ones(len(scores))
    pred[scores<t]=0
    print(confusion_matrix(true_graph_labels, pred))
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(accuracy_score(true_graph_labels, pred),precision_score(true_graph_labels, pred),recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)))
    return auc, precision_score(true_graph_labels, pred),recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)

if __name__ =='__main__':
    args=parse()
    seq_len=0
    train_len=[0, 4600] #cic[0,529], unsw[200:600], ustc[10:100], cic2018[0, 4600]
    data_embs = torch.load(args.embs_path).detach().cpu().numpy()
    labels = np.load(args.labels_path, allow_pickle=True)
    labels=labels[seq_len:]
    labels=np.concatenate((labels[:train_len[0]], labels[train_len[1]:]))
    
    # iof = IsolationForest()
    iof = OneClassSVM()
    iof=iof.fit(data_embs[train_len[0]+seq_len:train_len[1]])
    test_embs=np.concatenate([data_embs[:train_len[0]],data_embs[train_len[1]:]])
    scores=iof.decision_function(test_embs) #值越低越不正常
    aucv, pre, rec, f1=matrix(labels.astype(np.long), -scores)
    # np.save('/home/xiaoqing/gitpro/GNNPro/DyGCN/evaluate/cicscores/'+datapath.split('/')[0]+'scores.npy', -scores)
    pred = torch.zeros(len(scores))
    idx=scores.argsort()#从大到小


    vs=[aucv*100, pre*100, rec*100, f1*100]
    for k in range(500,2400,500):
        print('============ k=',k)
        pred[idx[:k]]=1
        a,b,c,d=eval(labels.astype(np.long), pred)
        # print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a,b,c,d))
        vs+=[b*100,c*100]
        # df.append([b,c])
    print(vs)
    df = pd.DataFrame([vs])
    df.to_csv('rgcn-o-osvm.csv')
