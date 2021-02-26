from inits import glorot, zeros
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.data import Data

import math
import time
from datetime import datetime
import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import (negative_sampling, remove_self_loops)


'''
test
'''
print(torch.cuda.is_available())

'''
input data(test)
'''
edge_index = torch.tensor([[0, 1], [0, 2], [0, 3], [1, 2], [3, 2]],dtype = torch.long)
edge_index = edge_index.t()
edge_attr = torch.tensor([5, 3, 4, 2, 1], dtype = torch.long)
x = torch.tensor([[35, 21, 42, 52, 3],[12, 32, 32, 12, 43],[1, 3, 4, 3, 5], [4, 5, 3, 19, 5]], dtype = torch.float)
data = Data(x = x, edge_index = edge_index, edge_attr = edge_attr)
adj_mat = np.mat([[0, 5, 3, 2],
                                [0, 0, 2, 0],
                                [0, 0, 0, 0],
                                [0, 0, 1, 0]])

'''
models
'''
class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels, data, adj_mat):
        super(CustomConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att = adj_mat
        self.denom =self.att.sum(axis = 0)
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight, 'weight')
        zeros(self.bias, 'bias')

    def forward(self, x, edge_index,):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes = x.size(self.node_dim))
        x = torch.matmul(x, self.weight)
        return self.propagate(edge_index, x = x)

    def message(self, edge_index_i, edge_index_j, x_i, x_j,):
        x_j = x_j.view(-1, self.out_channels) #reshape to (?, out_channels)
        denom =  self.denom[0, edge_index_i]
        alpha = np.exp(self.att[edge_index_j, edge_index_i])/ np.exp(denom)
        return x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out.add(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channels,
                                   self.out_channels)


class CustomEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomEncoder, self).__init__()
        self.conv1 = CustomConv(in_channels, 2*out_channels, data, adj_mat)
        self.conv2 = CustomConv(2*out_channels, out_channels, data, adj_mat)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class CustomDecoder(nn.Module):
    def forward(self, z, edge_index, sigmoid = True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z, sigmoid = True):
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class CustomGAE(torch.nn.Module):
    def __init__(self):
        super(CustomGAE, self).__init__()
        self.encoder = CustomEncoder(5, 5) #in_channels, out_channels
        self.decoder = CustomDecoder()
        CustomGAE.reset_parameters(self)

    def reset_parameters(self):
        #reset(self.encoder)
        #reset(self.decoder)
        pass

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, edge_index):
        L = (1/2.0)*(x-self.decoder(z, edge_index, sigmoid = True))#correct?
        return L

    def forward(self):
        pass

    def test(self, z, edge_index):
        y = z.new_ones(edge_index.size(1))
        pred = self.decoder(z, edge_index, sigmoid=True)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        return roc_auc_score(y, pred), average_precision_score(y, pred)
    
'''
test & output
'''
device = torch.device('cuda')
model = CustomGAE()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.005, weight_decay = 1e-3)
data= data.to(device)
#adj_mat = adj_mat.to(device)

#train
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model()
    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, edge_index)
    loss.backward()
    optimizer.step()

    #test
    if (epoch+1) % 5 == 0:
        model.eval()
        with torch.no_grad():
            z = model.encode(x, train_pos_edge_index)
        print(model.test(z, pos_edge_index, neg_edge_index))
