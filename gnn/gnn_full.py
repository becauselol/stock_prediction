import json
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

from gnn_torch_components import gcnlstmDataset, LSTMGCN

import wandb
wandb.init(# set the wandb project where this run will be logged
    project="gcn-madness",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.01,
    "architecture": "GCN",
    "dataset": "AAPL",
    "epochs": 1000,
    }
)


# params
learning_rate = 0.01
n_epochs = 1000
train_test_split_ratio = 0.8
split_date = "2022-01-01"
lookback = 60
batch_size = 16

ticker = "AAPL"
start = "2017-01-01"
end = "2022-12-31"

feature_columns = [
    "close_",
    "volume_"
]

start = "2017-01-01"
end = "2022-12-31"

device = "cuda" if torch.cuda.is_available() else "cpu"

file_path = f'../{ticker}_new_company_test.json'

data = json.load(open(file_path))

mega_df = pd.read_csv(f"../{ticker}_company_only_close_price.csv", index_col=0)

# map mega_df to get the new indexes based on the df
# only need to have it loading from csv
# get ticker to old index
ticker_to_old_index = {}
for k, company_dict in data["nodes"].items():
    if company_dict["ticker"] == "":
        continue
    ticker_to_old_index[company_dict["ticker"]] = k

# map old index to new_index
reverse_map = {}
# map new_index to old index
new_map = {}
last = ""
counter = 0
for name in mega_df.columns:
    header, ticker = name.split("_")
    if ticker == last:
        continue
    
    last = ticker
    
    old_index = ticker_to_old_index[ticker]
    reverse_map[old_index] = counter
    new_map[counter] = old_index
    counter += 1

# construct the new edge list
new_edgelist = []
for u, t, v in data["relationships"]:
    if str(u) in reverse_map and str(v) in reverse_map:
        new_edgelist.append([reverse_map[str(u)], reverse_map[str(v)]])

# only take those keys and make the edge list
edge_index = torch.tensor(new_edgelist)
edge_index = edge_index.t().contiguous()


# create the labels

y_series = []
# for every 2 columns, we take that DF then create a new column

for i in range(0, len(mega_df.columns), 2):
    temp_df = mega_df.iloc[:,i:i+2]
    ticker = temp_df.columns[0].split("_")[1]
    temp_df.loc[:,f"next_close_{ticker}"] = temp_df[f"close_{ticker}"].shift(-1)
    temp_df.loc[:,f"label_{ticker}"] = temp_df[f"next_close_{ticker}"] - temp_df[f"close_{ticker}"] >= 0
    temp_df.loc[:,f"label_{ticker}"] = temp_df[f"label_{ticker}"].apply(int)
    y_series.append(temp_df[f"label_{ticker}"])


target_df = pd.concat(y_series, axis=1)


# remove last row cos you cant predict for something you don't have
target_df = target_df.iloc[:-1,:]
mega_df = mega_df.iloc[:-1,:]


# construct all the features from 0 to n - 1
    
target_df.reset_index(inplace=True)
mega_df.reset_index(inplace=True)


# train-test split for time series
date_change_index =  mega_df.loc[mega_df["Date"] >= split_date,:].index[0]

train_features_df = mega_df.iloc[:date_change_index,:]
test_features_df = mega_df.iloc[date_change_index - lookback:,:]

train_labels_df = target_df.iloc[:date_change_index,:]
test_labels_df = target_df.iloc[date_change_index - lookback:,:]


# drop the date columns and scale
train_features_df = train_features_df.drop("Date", axis=1)
test_features_df = test_features_df.drop("Date", axis=1)

train_labels_df = train_labels_df.drop("Date", axis=1)
test_labels_df = test_labels_df.drop("Date", axis=1)

scaler = MinMaxScaler()
scaler.fit(train_features_df)


train_dataset = gcnlstmDataset(train_features_df, train_labels_df, scaler, 2, 1040, 60)
test_dataset = gcnlstmDataset(test_features_df, test_labels_df, scaler, 2, 1040, 60)


# MODEL TIME
model = LSTMGCN(0)


edge_index = edge_index.to(device)
model.to(device)

optimizer=optim.SGD(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.BCELoss()
loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)


# Magic
wandb.watch(model, log_freq=100)


for epoch in tqdm(range(1, n_epochs + 1)):
    model.train()
    for X_batch, y_batch in loader:
        
        y_batch = y_batch.to(device)
        X_batch = [i.to(device) for i in X_batch]
        
        y_pred = None
        
        for x in X_batch:

            y = model(x, edge_index)

            if y_pred is None:
                y_pred = y.t()
            else:
                y_pred = torch.cat((y_pred, y.t()), axis=0)
                
        loss = loss_fn(y_pred, y_batch)
        
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
    
    model.eval()

    y_pred = None
    y_actual = None
    for i in range(len(test_dataset)):
        x, y = test_dataset[i]

        x = x.to("cuda")
        y = y.unsqueeze(0)

        y_hat = model(x, edge_index)

        if y_pred is None:
            y_pred = y_hat.t()
            y_actual = y
        else:
            y_pred = torch.cat((y_pred, y_hat.t()), axis=0)
            y_actual = torch.cat((y_actual, y), axis=0)

    test_loss = loss_fn(y_pred, y_actual.to(device))

    wandb.log({
        "train/loss": loss.item(),
        "test/loss": test_loss.item(),
    })


    if epoch % 100 == 0:
        torch.save(model.state_dict(),  f"LSTMGCN_{ticker}_epoch{epoch}_lookback{lookback}_lr{learning_rate}_batch{batch_size}.pth")

torch.save(model.state_dict(),  f"LSTMGCN_{ticker}_epoch{n_epoch}_lookback{lookback}_lr{learning_rate}_batch{batch_size}.pth")
