# Mathematical-modeling-contest-for-Postgraduates
模型代码
3层LSTM(每层units128,256,128，每层dropout0.3,0.5,0.3)和一层全连接Dense。  
调参后，最优参数为：  
不进行shuffle（shuffle模型拟合能力显著下降，侧面说明了序列的时序相关性）  
输入序列长度（数据帧）设置为3  
优化器为带动量的SGD(lr=0.015，学习率衰减=1e-6)  
batch_size=64  
验证集loss随epoch先降低后升高，故在序列长度为3情况下最优epoch为10左右。  
未调参的DLSTM模型预测MAPE大致保持在20%左右（最高为25%），调参后MAPE稳定在13％以下，\.  
最后模型平均绝对百分比误差MAPE达到了12％(预测精度接近90%)，好过回归和ANN。
