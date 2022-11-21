import os
import sys
from statistics import mode
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from DyGCN.utils import get_edge_embs
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.svm import OneClassSVM
from torch import optim
from torch.nn.functional import mse_loss
from torch.utils.data import dataset
from tqdm import tqdm

cur_path=os.path.dirname(__file__)
os.chdir(cur_path)

class AutoEncoder(torch.nn.Module):
    def __init__(self,in_dim):
        super(AutoEncoder,self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.Tanh()
        )
        self.decoder=nn.Sequential(
            nn.Linear(32, in_dim),
        )
    def forward(self,x):
        z=self.encoder(x)
        x_hat=self.decoder(z)
        return x_hat


class FlowDataset(dataset.Dataset):
    def __init__(self, data):
        self.data=data
    def __getitem__(self, index):
        return self.data[index], self.data[index]
    def __len__(self):
        return len(self.data)

def train_ae(train_flow, device, epochs=50):  
    fds = FlowDataset(train_flow)
    loader =dataset.DataLoader(dataset=fds, batch_size=128, shuffle=True)
    ae = AutoEncoder(train_flow.shape[1])
    ae.to(device)
    optimizer = optim.Adam(ae.parameters(), lr=1e-4)
    loss_ae = nn.MSELoss()
    ls=[]
    for epoch in range(epochs):
        for train_data, _ in tqdm(loader):
            train_data=train_data.to(device)
            optimizer.zero_grad()
            y_pred = ae(train_data)
            le = loss_ae(train_data, y_pred[1])
            le.backward()
            optimizer.step()
        print('epoch:{:.3f} loss:{:.4f}'.format(epoch,le))
        ls.append(le)
        
    x = np.arange(0,epochs)
    plt.plot(x, ls)
    plt.show()    
    plt.close()
    return ae


def plot_roc(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    maxindex = (tpr-fpr).tolist().index(max(tpr-fpr))
    threshold = thresholds[maxindex]
    print('异常阈值', threshold)
    auc_score = auc(fpr, tpr)
    print('auc值: {:.4f}'.format(auc_score))
    return threshold

def matrix(true_graph_labels,scores):
    t=plot_roc(true_graph_labels, scores)
    true_graph_labels = np.array(true_graph_labels)
    scores  = np.array(scores)
    pred=np.ones(len(scores))
    pred[scores<t]=0
    print(confusion_matrix(true_graph_labels, pred))
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(accuracy_score(true_graph_labels, pred),precision_score(true_graph_labels, pred),recall_score(true_graph_labels, pred), f1_score(true_graph_labels, pred)))

def eval(labels,pred):
    plot_roc(labels, pred)
    print(confusion_matrix(labels, pred))
    a,b,c,d=accuracy_score(labels, pred),precision_score(labels, pred),recall_score(labels, pred), f1_score(labels, pred)
    print("acc:{:.4f},pre{:.4f},rec:{:.4f}, f1:{:.4f}".format(a,b,c,d))
    return a,b,c,d
def dataprocess(datapath):
    data = np.load(datapath, allow_pickle=True)
    return np.mean(data, 1)

def trainOsvm(datapath, labelpath):
    train_len=[0, 529]
    data = np.load(datapath, allow_pickle=True)
    labels=np.load(labelpath, allow_pickle=True)
    x= np.mean(data, 1)
    torch.save(torch.from_numpy(x), 'graph_embs.pt')
    f_dim=x.shape[1]
    x_train, x_test=x[train_len[0]:train_len[1]].reshape(-1, f_dim), x[train_len[1]:].reshape(-1, f_dim)
    y_test=labels[train_len[1]:]
    model=OneClassSVM()
    model=model.fit(x_train)
    scores=model.decision_function(x_test) #值越低越不正常
    matrix(y_test.astype(np.long), -scores)

def train_ae():
    device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    train_len=[0, 529]
    x = dataprocess('/home/xiaoqing/gitpro/GNNPro/DyGCN/DyGCN/data/cic2018/feats.npy')
    f_dim=x.shape[-1]
    x=torch.FloatTensor(x).to(device)
    x_train, x_test=x[train_len[0]:train_len[1]].reshape(-1, f_dim), x[train_len[1]:].reshape(-1, f_dim)
    t=time()

    model = train_ae(x_train, device, epochs=50)
    print('train ae time', time()-t)

    t3=time()
    x_hat=model(x_test)
    mse=F.mse_loss(x_test, x_hat, reduce=False).mean(dim=1)
    mse=mse.cpu().detach().numpy()
    print('test ae time', time()-t3)
    np.save('ae_scores.npy', mse)

if __name__ ==  '__main__':

    trainOsvm(datapath='/home/xiaoqing/gitpro/GNNPro/DyGCN/DyGCN/data/cic2018/feats.npy', labelpath='/home/xiaoqing/gitpro/GNNPro/DyGCN/DyGCN/data/cic2018/labels.npy')


