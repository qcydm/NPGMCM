# Mathematical-modeling-contest-for-Postgraduates
Model code 3-layer LSTM (units 128, 256, 128 per layer, dropout 0.3, 0.5, 0.3 per layer) and one layer of fully connected Dense.
After tuning, the optimal parameters are:
No shuffle (the shuffle model's fitting ability is significantly reduced, which shows the time series correlation of the sequence)
Input sequence length (dataframe) set to 3
The optimizer is SGD with momentum (lr=0.015, learning rate decay=1e-6)
batch_size=64
The validation set loss first decreases and then increases with the epoch, so the optimal epoch is about 10 when the sequence length is 3.
The DLSTM model without parameter adjustment predicts that MAPE is roughly maintained at around 20% (the highest is 25%), and MAPE stabilizes below 13% after parameter adjustment.\.  
Finally, the average absolute percentage error MAPE of the model reaches 12% (prediction accuracy is close to 90%), which is better than regression and ANN.
![PANN 1YX_G ` S4`BD}4 MU](https://user-images.githubusercontent.com/42266769/113497877-7fd15880-953a-11eb-8dd2-ee74a5954e78.png)
