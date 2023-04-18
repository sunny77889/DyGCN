# 基于动态图神经网络的异常流量检测方法DyGCN
## 实验数据集
CIC-IDS 2017：https://www.unb.ca/cic/datasets/ids-2017.html 

CSE-CIC-IDS 2018: https://www.unb.ca/cic/datasets/ids-2018.html

将下载的数据集放在 `/data` 目录下。对于CSE-CIC-IDS 2018数据集，只使用“Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv”这一天的数据，因为其他数据文件不包含IP，无法建图。

动态图模型的作用在于进行子图嵌入，子图嵌入的结果可以用异常检测模型进行异常判断。

其他对比模型的代码在`/compare_models`目录下

## 模型训练
基于cic2017数据集进行模型训练

    python DyGCN/main.py --mode train --ck_path DyGCN/savedmodel/model.pt --embs_path DyGCN/data/graph_embs.pt --dataset data/cic2017

## 模型测试
基于cic2017数据集进行模型测试

    python DyGCN/main.py --mode test --ck_path DyGCN/savedmodel/model.pt --embs_path DyGCN/data/graph_embs.pt --dataset data/cic2017


## 异常检测
基于模型图嵌入结果进行异常检测

    python DyGCN/intrusion_detection.py --dataset cic2017
