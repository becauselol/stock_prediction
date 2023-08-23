import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

import torch.optim as optim
import torch.utils.data as data
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm


# construct all the features from 0 to n - 1


class gcnlstmDataset(torch.utils.data.Dataset):
    def __init__(self, features_dataframe, labels_dataframe, scaler, num_features, num_companies, sequence_length=60):
        self.num_features = num_features
        self.num_companies = num_companies
#         self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(labels_dataframe.values).float()
        self.scaler = scaler
        # transpose to make it horizontal
#         self.X = torch.tensor(scaler.transform(features_dataframe)).t().float()

        # SCALEEE
        self.X = torch.tensor(self.scaler.transform(features_dataframe)).float()

    def __len__(self):
        return self.X.shape[0] - self.sequence_length + 1

    def __getitem__(self, i): 
        
        # splits it up into close + volume
        # we will need to transpose this later when using the model
        
#         print(x)
        # raise error if too long
#         if i > len(self) - 1:
#             raise IndexError("out of bounds")
#         print(i + self.sequence_length)
        return torch.reshape(self.X[i:i + self.sequence_length,:], (self.sequence_length, self.num_companies, self.num_features)), self.y[i + self.sequence_length - 1]
    
    
# self.X = torch.tensor(scaler.transform(dataframe[features])).float()


# create the model
# We take LSTM to create sequence embedding
# feed that as features
# Put into regular GCN
# predict outcome

class LSTMGCN(torch.nn.Module):
    def __init__(self, target_node):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=2, hidden_size = 50, num_layers=1)
        self.conv1 = GCNConv(50, 16)
        self.relu = torch.nn.ReLU()
        self.conv2 = GCNConv(16, 1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.target_node = target_node
        

    def forward(self, data, edge_index):
        # for each stock price and volume
        x, (embedded_input, c) = self.lstm(data)
        
                
#         print(embedded_input.shape)
        # I add it to the macro features (not done yet)
        x = self.conv1(embedded_input.squeeze(dim=0), edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.softmax(x)
        return x
