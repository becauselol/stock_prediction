import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as metrics

from LSTM_model import lstmModel
from helper_processing_functions import min_max_scaler, min_max_inverse_scaler, create_dataset

# params
learning_rate = 0.01
n_epochs = 1000
train_test_split_ratio = 0.8
lookback = 4
batch_size = 16

device = "cuda" if torch.cuda.is_available() else "cpu"

spy_df = pd.read_csv("../data/spy_data.csv")

timeseries = spy_df[["Close"]].values.astype("float32")

timeseries = [i[0] for i in timeseries]

# convert to 1 for increase and 0 for decrease
labels = [int(i - j > 0) for i, j in zip(timeseries[1:], timeseries[:-1])]

# remove the first item in time series to have same length input of labels and timeseries
timeseries.pop(0)

# train-test split for time series
train_size = int(len(timeseries) * train_test_split_ratio)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]
train_label, test_label = labels[:train_size], labels[train_size:]

train = np.array(train)

maximum = train.max()
minimum = train.min()

train = min_max_scaler(train, minimum, maximum)
test = min_max_scaler(test, minimum, maximum)

X_train, y_train = create_dataset(train, train_label, lookback=lookback)
X_test, y_test = create_dataset(test, test_label, lookback=lookback)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

model =lstmModel()
model.to(device)

optimizer=optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = nn.BCELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size)


model.train()
for epoch in range(n_epochs):
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 
        

torch.save(model.state_dict(), f"LSTM_{ticker}_nodes{nodes}_epochs{n_epochs}_lookback{lookback}_lr{learning_rate}_batch{batch_size}_.pth")