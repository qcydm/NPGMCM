from keras.layers import LSTM, Dropout, Dense, Embedding
from keras.models import Sequential
import keras
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_name, sequence_length, split):
    # 读取数据，并把数据转换为数组
    df = pd.read_csv(file_name, sep=',')
    data_all = np.array(df).astype(float)

    # 将数据缩放至给定的最小值与最大值之间，这里是０与１之间，数据预处理
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)

    # 构造送入lstm的数据帧：(322, 3, 26)
    data = []
    for i in range(len(data_all) - sequence_length ):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')

    # 时序问题不进行shuffle
    #np.random.shuffle(reshaped_data)

    # 向下取整，得到训练集和测试集
    train=[]
    test=[]
    train_num=int(len(reshaped_data)*split)
    test_num=len(reshaped_data)-train_num
    for i in range(train_num):
        train.append(reshaped_data[i])
    for j in range(train_num,len(reshaped_data)):
        test.append(reshaped_data[j])

    #划分特征和待预测目标
    x_train=[]
    y_train=[]
    for frame in train:
        x_train.append(frame[:sequence_length])
        y_train.append(frame[sequence_length][-1])

    x_test=[]
    y_test=[]
    for frame in test:
        x_test.append(frame[:sequence_length])
        y_test.append(frame[sequence_length][-1])

    return x_train,x_test,y_train,y_test,scaler

#DLST模型搭建
model = Sequential()
model.add(LSTM(128,return_sequences=True,input_shape=(3,26),dropout=0.3))
model.add(LSTM(256,return_sequences=True,dropout=0.5))
model.add(LSTM(128,return_sequences=False,dropout=0.3))
model.add(Dense(1,activation='linear'))
sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse',optimizer='sgd')
#网络层参数打印
model.summary()



#模型训练（参数保存代码另示）
filename='./data.csv'
x_train,x_test,y_train,y_test,scaler=load_data(filename,sequence_length=3,split=0.8)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_test=np.asarray(x_test)
y_test=np.asarray(y_test)
model.fit(x_train, y_train,
          batch_size=64, epochs=12,
          validation_data=(x_test, y_test))
#keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0002, patience=1, verbose=0, mode='min', baseline=0.01, restore_best_weights=False)

#模型测试
mape=0
rmse=0
predict=model.predict(x_test)
for i in range(len(predict)):
    mape+=(abs(predict[i][0]-y_test[i])/y_test[i])
    rmse+=((predict[i][0]-y_test[i])**2)
print("MAPE=",float(mape/len(y_test)))
print("RMSE=",float((rmse/len(y_test)))**0.5)
#0.1521390735140718

#第一层dropout=0.1，第二层dropout=0.5：0.13691493415083336  val_loss=0.0123
#0.3 0.5 0.3 0.12927837044776957
#不shuffle，epoch11左右，#不shuffle，0.12337597135456382   0.09009866600314395
#序列长度3 epoch12 0.12007058569474864   0.08986360055470237  128 256 128

