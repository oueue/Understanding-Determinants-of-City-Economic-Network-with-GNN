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
a = torch.tensor(
    [[0, 5, 3, 4],
        [0, 0, 2, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 0]])
'''
models
'''
class CustomConv(MessagePassing):
    def __init__(self, in_channels, out_channels, data):
        super(CustomConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att = edge_attr
        self.weight = Parameter(torch.empty(in_channels, out_channels, dtype = torch.float, requires_grad = True))
        self.bias = Parameter(torch.empty(out_channels, dtype = torch.float, requires_grad = True))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight, 'weight')
        zeros(self.bias, 'bias')

    def forward(self, x, edge_index,):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x = torch.mm(x, self.weight).view(-1, self.out_channels)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes = x.size(0))
        return self.propagate(edge_index, x = x)

    def message(self, edge_index_i, edge_index_j, x_i, x_j,):
        x_j = x_j.view(-1, self.out_channels) #reshape to (?, out_channels)
        alpha = np.exp(self.att)
        return x_j * alpha

    def update(self, aggr_out):
        aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channels,
                                   self.out_channels)


class CustomEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomEncoder, self).__init__()
        self.conv = CustomConv(in_channels, out_channels, data)

    def forward(self, x, edge_index):
        x = F.relu(self.conv(x, edge_index))
        return x


class CustomDecoder(nn.Module):
    def forward(self, z, sigmoid = True):
        #print('z')
        #print(z)
        adj = torch.matmul(z, z.t())
        #print('z*z.t = ')
        #print(adj)
        return torch.sigmoid(adj) if sigmoid else adj


class CustomGAE(torch.nn.Module):
    def __init__(self):
        super(CustomGAE, self).__init__()
        self.encoder = CustomEncoder(5, 1) #in_channels, out_channels
        self.decoder = CustomDecoder()
        CustomGAE.reset_parameters(self)

    def reset_parameters(self):
        glorot(self.encoder.conv.weight)
        #reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z, a):
        a_prime = self.decoder(z, sigmoid = False)
        sub = torch.sub(a, a_prime)
        L = (1/2.0)*(sub)
        l = torch.mean(L)
        l.requires_grad_(True)
        return l

    def forward(self):
        pass
    
'''
test & output
'''
device = torch.device('cuda')
model = CustomGAE()
optimizer = torch.optim.Adam((model.encoder.conv.weight, model.encoder.conv.bias), lr = 0.01, weight_decay = 0)
data= data.to(device)

#train
for epoch in range(500):
    model.eval()
    optimizer.zero_grad()
    out = model()
    z = model.encode(x, edge_index)
    loss = model.recon_loss(z, a)
    loss.backward()
    optimizer.step()

    #test
    if (epoch+1) % 50 == 0:
        print('loss')
        print(loss.item())
        print('weight')
        print(model.encoder.conv.weight)
        print('bias')
        print(model.encoder.conv.bias)
