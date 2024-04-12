import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from termcolor import cprint
from models.Kmeans import Kmeans
from torch_geometric.nn import GCNConv, ChebConv
from torch_geometric.nn import global_mean_pool, EdgePooling


class WeightedGCN(torch.nn.Module):

    def __init__(self, params):
        super(WeightedGCN, self).__init__()
        self.embed = nn.Embedding(params.N, 10, scale_grad_by_freq=False)

        self.convs = nn.ModuleList([GCNConv(i, o, improved=True) for i, o in [(10, params.z_dim)]])
        # self.convs = nn.ModuleList([ChebConv(i, o, K=2) for i, o in [(10, params.z_dim)]])

        # self.convs = nn.ModuleList([GCNConv(i, o, improved=True) for i, o in [(10, 10), (10, 10), (10, params.z_dim)]])
        # self.convs = nn.ModuleList([ChebConv(i, o, K=2) for i, o in [(10, 10), (10, 10), (10, params.z_dim)]])

        # self.activate = nn.ELU()
        self.activate = nn.Identity()
        self.drop = nn.Dropout(p=0.0)
        self.readout = nn.Sequential(
            # nn.Linear(params.z_dim, params.z_dim, bias=True),
            # nn.ELU(),
            nn.Linear(params.z_dim, params.K, bias=False),)

        a = 0
        for p in self.parameters():
            a += np.prod(p.shape)
        print(f"Number of parameters in model: {a}")
        print(self)

    def forward(self, x, edge_index, edge_weight, batch, deterministic=False):
        x = self.embed(x).squeeze()

        if deterministic:
            torch.use_deterministic_algorithms(False)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = self.activate(x)
            x = self.drop(x)
        x = self.activate(global_mean_pool(x, batch))
        # x = F.normalize(x, p=2, dim=1)

        if deterministic:
            torch.use_deterministic_algorithms(True)

        logits = self.readout(x)

        return x, logits