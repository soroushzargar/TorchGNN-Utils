import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, APPNP, SAGEConv
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Torch Graph Models are running on {device}")


class GCN(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_classes, p_dropout):
        super().__init__()
        self.conv1 = GCNConv(n_features, n_hidden)
        self.conv2 = GCNConv(n_hidden, n_classes)
        self.p_dropout = p_dropout

    def forward(self, X, edge_weight=None):
        x, edge_index = X.x, X.edge_index
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x