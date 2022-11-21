from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import softmax
from torch.nn.functional import mse_loss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from DyGCN.models import DGCN, DGCN2, DGCN3, LSTM_AE, AutoEncoder, Classifier
from DyGCN.utils import FlowDataset, eval, get_edge_embs


# LSTM 学习节点的时序性，若干个时刻的节点集作为一个序列
def train_gcn_lstm5(args, data, epochs, train_len):
    device=args.device
    model = DGCN3(args).to(device)
    cls = Classifier(indim=args.out_dim*2).to(device)
    opt_gcn = optim.Adam(model.parameters(),lr=args.learning_rate)
    opt_cls=  optim.Adam(cls.parameters(), lr=args.learning_rate)
    loss_gcn = nn.CrossEntropyLoss()
    seq_len=args.seq_len
    losses=[]
    model.train()
    cls.train()
    for epoch in range(epochs):
        gcn_loss=0
        for i in tqdm(range(train_len[0], train_len[1]-seq_len)):
            opt_gcn.zero_grad()
            opt_cls.zero_grad()
            x = data.feat_list[i: i+seq_len]
            x = th.FloatTensor(x).to(device)
            Ain = data.Ain_list[i: i+seq_len] # 目的IP-边: 入边有向图
            Aout = data.Aout_list[i: i+seq_len] # 源IP-边: 出边有向图
            adj = data.adj_list[i: i+seq_len] # 源IP-目的IP: 无向图
            ips = data.ip_list[i: i+seq_len] # 节点集

            cur_ips = data.ip_list[i+seq_len-1] # 当前时刻的图
            cur_A=data.A_list[i+seq_len-1] # 归一化前的IP邻接矩阵（当前时刻）

            node_feats = model(x, Ain, Aout, adj, ips, cur_ips)
            edge_embs, labels, _ = get_edge_embs(node_feats, cur_A, abalation=True) # 边的正负采样
            pred= cls(edge_embs) # 基于分类器计算边的异常分数
            loss = loss_gcn(pred, torch.LongTensor(labels).to(device=device))
            loss.backward()
            gcn_loss+=loss.item()
            opt_gcn.step()
            opt_cls.step()
        losses.append(gcn_loss/i)
        print('epoch:{:.4f}, gcn_loss:{:.4f}'.format(epoch, gcn_loss/i))
        torch.save({
            'gcn_model':model.state_dict(),
            'cls':cls.state_dict()
        }, args.ck_path)

    return losses

def predict5(args,data):

    device = args.device
    model = DGCN3(args).to(device)
    cls = Classifier(indim=args.out_dim*2).to(device)

    ck=torch.load(args.ck_path)
    model.load_state_dict(ck['gcn_model'])
    cls.load_state_dict(ck['cls'])
    
    model.eval()
    cls.eval()
    seq_len=args.seq_len
    graph_embs=[]
    data.feat_list=torch.FloatTensor(data.feat_list).to(device)
    for i in tqdm(range(len(data.feat_list)-seq_len)):
        x = data.feat_list[i: i+seq_len]
        A_in = data.Ain_list[i: i+seq_len]
        A_out=data.Aout_list[i: i+seq_len]
        adj = data.adj_list[i: i+seq_len]
        ips = data.ip_list[i: i+seq_len]
        cur_ips = data.ip_list[i]
        cur_A=data.A_list[i]
        output = model(x, A_in,A_out, adj, ips,cur_ips)
        edge_embs, labels, pos_edges = get_edge_embs(output, cur_A)
        pred=cls(pos_edges)
        pred = softmax(pred,0)
        pos_edges=pos_edges*pred[:,0].unsqueeze(1)
        graph_embs.append(pos_edges.sum(0)) # 图的嵌入向量=边嵌入向量*边异常概率
        # graph_embs.append(torch.cat((data.X_list[i], pos_edges),1).mean(0)) #边的特征=[原始特征，源节点特征，目的节点特征]
        # graph_embs.append(data.X_list[i].mean(0))
    torch.save(torch.stack(graph_embs),args.embs_path)

# LSTM学习节点的时序性，每个节点序列都是LSTM的一个隐层状态
def train_gcn_lstm4(data, device, epochs):
    model = DGCN(nfeat=77, nhid=64, outd=32,  dropout=0.5).to(device)
    opt_gcn = optim.Adam(model.parameters(),lr=0.0001)

    cls = Classifier(in_features=32).to(device)
    opt_cls=  optim.Adam(cls.parameters(), lr=0.0001)
    loss_gcn = nn.CrossEntropyLoss()


    lstm = nn.LSTM(input_size=32, hidden_size=16).to(device)
    opt_lstm = optim.Adam(lstm.parameters(),lr=0.0001)
    

    model.train()
    for epoch in range(epochs):
        gcn_loss=0
        seq_len=3
        for i in tqdm(range(len(data.A_list))):
            if i>=seq_len:
                opt_lstm.zero_grad()
                opt_gcn.zero_grad()
                opt_cls.zero_grad()
                edge_embs, labels, pos_edges=inference(data,i,seq_len, device, model, lstm)
                pred= cls(edge_embs)
                loss = loss_gcn(pred, torch.LongTensor(labels).to(device))
                loss.backward()
                gcn_loss+=loss.item()
                opt_lstm.step() 
                opt_gcn.step()
                opt_cls.step()
            
        print('epoch:{:.4f}, gcn_loss:{:.4f}'.format(epoch,gcn_loss/i))
        
    return model, cls, lstm

def predict4(data, gl, device, name):
    gcn_model, cls, lstm=gl
    gcn_model.eval()
    seq_len=5
    graph_embs=[]
    
    for i in tqdm(range(len(data.A_list))):
        if i>=seq_len:
            _, _, pos_edges=inference(data,i,seq_len, device, gcn_model, lstm)
            graph_embs.append(pos_edges.mean(dim=0))
    torch.save(torch.stack(graph_embs), name+"_embs.pt")


def inference(data,i,seq_len,device,gcn_model, lstm):
    x = data.X_list[i: i+seq_len]
    x = th.FloatTensor(x).to(device)
    ifa = data.A_list[i: i+seq_len]
    adj = data.adj_list[i: i+seq_len]
    ips = data.ip_list[i: i+seq_len]
    cur_ips = data.ip_list[i]
    cur_adj=data.adj_list[i]
    seqs=[]
    for j in range(seq_len):
        output = gcn_model(x[j], ifa[j].to(device), adj[j].to(device))
        idx1 = np.where(np.in1d(ips[j], cur_ips))[0]
        idx2 = np.where(np.in1d(cur_ips, ips[j]))[0]
        aa = torch.zeros(len(cur_ips),32)
        aa[idx2]=output[idx1]
        seqs.append(aa.unsqueeze(0))
    seqs=torch.vstack(seqs)
    node_embs,_ = lstm(seqs)
    edge_embs, labels, pos_edges = get_edge_embs(node_embs[-1],cur_adj.to(device))
    return edge_embs, labels, pos_edges


# 考虑特征向量中心性
def train_gcn_lstm3(data, device, epochs):
    model = DGCN(nfeat=77, nhid=32, outd=16,  dropout=0.5).to(device)
    opt_gcn = optim.Adam(model.parameters(),lr=0.0001)

    lstm_ae = LSTM_AE(32).to(device)
    opt_lstm = optim.Adam(lstm_ae.parameters(),lr=0.0001)
    loss_lstm_ae=nn.MSELoss()

    train_flow=[]
    losses=[]
    model.train()
    seq_len=5
    for epoch in range(epochs):
        for i in tqdm(range(len(data.A_list))):
            if i>=seq_len:
                opt_lstm.zero_grad()
                opt_gcn.zero_grad()
                x = data.X_list[i: i+seq_len]
                train_flow.append(x)
                x = th.FloatTensor(x).to(device)
                ifa = data.A_list[i: i+seq_len]
                adj = data.adj_list[i: i+seq_len]
                graph_embs = []
                for i in range(seq_len):
                    output = model(x[i], ifa[i].to(device), adj[i].to(device))
                    edge_embs, labels, pos_edges = get_edge_embs(output,adj[i].to(device))
                    graph_embs.append(pos_edges.mean(dim=0))
                graph_embs=torch.stack(graph_embs).unsqueeze(0)
                z,x_hat=lstm_ae(graph_embs)
                loss = loss_lstm_ae(graph_embs, x_hat)
                loss.backward()
                losses.append(loss.item())
            opt_gcn.step()
            opt_lstm.step()
        
        print('epoch {:.4f} gcn_loss {:.4f}'.format(epoch, np.average(losses)))
    return model, lstm_ae
def predict3(data, gl, device):
    model, lstm_ae = gl

    train_flow=[]
    losses=[]
    model.eval()
    seq_len=6
    for i in tqdm(range(len(data.A_list))):
        if i>=seq_len:
            x = data.X_list[i: i+seq_len]
            train_flow.append(x)
            x = th.FloatTensor(x).to(device)
            ifa = data.A_list[i: i+seq_len]
            adj = data.adj_list[i: i+seq_len]
            graph_embs = []
            for i in range(seq_len):
                output = model(x[i], ifa[i].to(device), adj[i].to(device))
                edge_embs, labels, pos_edges = get_edge_embs(output,adj[i].to(device))
                graph_embs.append(pos_edges.mean(dim=0))
            graph_embs=torch.stack(graph_embs).unsqueeze(0)
            
            z,x_hat=lstm_ae(graph_embs)
            loss = F.mse_loss(graph_embs, x_hat)
            losses.append(loss.item())
    return np.array(losses)


# GCN+LSTMAE联合训练
def train_gcn_lstm2(data, device, epochs):
    model = DGCN(nfeat=77, nhid=32, outd=16,  dropout=0.5).to(device)
    opt_gcn = optim.Adam(model.parameters(),lr=0.0001)

    lstm_ae = LSTM_AE(32).to(device)
    opt_lstm = optim.Adam(lstm_ae.parameters(),lr=0.0001)
    loss_lstm_ae=nn.MSELoss()

    train_flow=[]
    losses=[]
    model.train()
    seq_len=6
    for epoch in range(epochs):
        t0 = time()
        for i in tqdm(range(len(data.A_list))):
            if i>=seq_len:
                opt_lstm.zero_grad()
                opt_gcn.zero_grad()
                x = data.X_list[i: i+seq_len]
                train_flow.append(x)
                x = th.FloatTensor(x).to(device)
                ifa = data.A_list[i: i+seq_len]
                adj = data.adj_list[i: i+seq_len]
                graph_embs = []
                for i in range(seq_len):
                    output = model(x[i], ifa[i].to(device), adj[i].to(device))
                    edge_embs, labels, pos_edges = get_edge_embs(output,adj[i])
                    graph_embs.append(pos_edges.mean(dim=0))
                graph_embs=torch.stack(graph_embs).unsqueeze(0)
                z,x_hat=lstm_ae(graph_embs)
                loss = loss_lstm_ae(graph_embs, x_hat)
                loss.backward()
                losses.append(loss.item())
            opt_gcn.step()
            opt_lstm.step()
        
        print('epoch {:.4f} gcn_loss {:.4f}'.format(epoch, np.average(losses)))
        print('epoch time', time()-t0)
    return model, lstm_ae
def predict2(data, gl, device):
    model, lstm_ae = gl

    losses=[]
    model.eval()
    seq_len=6
    for i in tqdm(range(len(data.A_list))):
        if i>=seq_len:
            x = data.X_list[i: i+seq_len]
            x = th.FloatTensor(x).to(device)
            ifa = data.A_list[i: i+seq_len]
            adj = data.adj_list[i: i+seq_len]
            graph_embs = []
            for i in range(seq_len):
                output = model(x[i], ifa[i].to(device), adj[i].to(device))
                edge_embs, labels, pos_edges = get_edge_embs(output,adj[i])
                graph_embs.append(pos_edges.mean(dim=0))
            graph_embs=torch.stack(graph_embs).unsqueeze(0)
            
            z,x_hat=lstm_ae(graph_embs)
            loss = F.mse_loss(graph_embs, x_hat)
            losses.append(loss.item())
    return np.array(losses)

# LSTM学习图的时序性
def train_gcn_lstm(data, device, epochs):
    model = DGCN(nfeat=77, nhid=32, outd=16,  dropout=0.5).to(device)
    opt_gcn = optim.Adam(model.parameters(),lr=0.0001)

    cls = Classifier(in_features=32).to(device)
    opt_cls=  optim.Adam(cls.parameters(), lr=0.0001)
    loss_gcn = nn.CrossEntropyLoss()


    lstm = nn.LSTMCell(input_size=32, hidden_size=16).to(device)
    opt_lstm = optim.Adam(lstm.parameters(),lr=0.0001)
    
    # #每一时刻的图都是一个状态，将该时刻的图的全局表示，上一时刻的隐藏层状态，记忆单元作为当前时刻LSTM的输入
    model.train()
    for epoch in range(epochs):
        h,c = th.zeros(1,16).to(device),th.zeros(1,16).to(device) #初始化0时刻的隐藏层状态和记忆单元
        his_graph = th.zeros(1,16).to(device)
        train_graph_embs = []
        gcn_loss=0
        lstm_loss=0
        for i in tqdm(range(len(data.A_list))):
            opt_gcn.zero_grad()
            opt_cls.zero_grad()
            x = data.X_list[i]
            x = th.FloatTensor(x).to(device)
            ifa = data.A_list[i].to(device)
            adj = data.adj_list[i].to(device)
            output = model(x, ifa, adj)
            edge_embs, labels, pos_edges = get_edge_embs(output,adj)

            # train_graph_embs.append(pos_edges.mean(dim=0).detach())
            graph_emb = pos_edges.mean(dim=0).detach()
            pred= cls(edge_embs)
            loss = loss_gcn(pred, torch.LongTensor(labels).to(device))
            loss.backward(retain_graph=True)
            
            gcn_loss+=loss.item()
            if epoch ==epochs-1:
                opt_lstm.zero_grad()
                # graph_emb = train_graph_embs[i]
                h,c = lstm(graph_emb.unsqueeze(0),(h.detach(),c.detach()))   
                mean_his = his_graph/(i+1)
                his_graph=his_graph+h
                loss_lstm = th.sum(th.abs(h-mean_his.detach())) #求损失值
                loss_lstm.backward()
                opt_lstm.step()
            opt_gcn.step()
            opt_cls.step()
            
            # lstm_loss+=loss_lstm.item()
        # print('lstm_loss:{:.4f}'.format(loss_lstm.item()))
        # train_graph_embs=torch.stack(train_graph_embs)
        print('epoch:{:.4f}, gcn_loss:{:.4f}'.format(epoch,gcn_loss/i))
        
    return model, cls, lstm, mean_his,h,c

def predict(test, gl, device):
    gcn_model, cls, lstm, mean_his,h,c=gl
    scores = []
    gcn_model.eval()
    for i in tqdm(range(len(test.A_list))):
        x = test.X_list[i]
        x = th.FloatTensor(x).to(device)
        ifa = test.A_list[i].to(device)
        adj = test.adj_list[i].to(device)
        output=gcn_model(x, ifa, adj)
        # 通过拼接源节点的入边特征, 目的节点的出边特征得到边的特征
        edge_indices = adj._indices()
        edge_embs = torch.cat([output[edge_indices[0]], output[edge_indices[1]]], dim=1)

        #对所有边的特征求mean的得到图的嵌入特征
        graph_emb = th.mean(edge_embs, dim=0).detach().type_as(h)
        h, c = lstm(graph_emb.unsqueeze(0),(h.detach(), c.detach()))
        loss_lstm = th.sum(th.abs(h-mean_his.detach()))
        scores.append(loss_lstm.item())
    return np.array(scores)


def train_ae(train_flow, device, epochs=50):  
    fds = FlowDataset(train_flow)
    loader =DataLoader(dataset=fds, batch_size=128, shuffle=True)
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
