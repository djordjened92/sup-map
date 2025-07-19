import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.prelu1 = nn.SiLU()
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.prelu2 = nn.SiLU()
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.prelu3 = nn.SiLU()
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x_init = x
        x = F.normalize(x, dim=-1)
        x = self.conv1(x, edge_index)
        x = self.prelu1(x)
        x += x_init
        
        x_init = x
        x = F.normalize(x, dim=-1)
        x = self.conv2(x, edge_index)
        x = self.prelu2(x)
        x += x_init

        x_init = x
        x = F.normalize(x, dim=-1)
        x = self.fc(x)
        x = self.prelu3(x)
        x += x_init

        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.normalize(x, dim=-1)
        return x