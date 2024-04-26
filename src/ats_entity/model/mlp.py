from __future__ import unicode_literals, print_function, division
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.datautil import DropoutDim


class MLP(nn.Module):
    def __init__(self, cfg: List[int], activation = torch.tanh,
                dropout: float = 0.0, bias: bool = True,
                batch_norm: bool = False):
        super().__init__()
        layers = [nn.Linear(idim, odim, bias=bias) for odim, idim in zip(cfg[1:], cfg[:-1])]
        self.mlp = nn.ModuleList(layers)
        self.batch_norms = None
        if batch_norm:
            batch_norms = [nn.BatchNorm1d(dim) for dim in cfg[1:]]
            self.batch_norms = nn.ModuleList(batch_norms)
        self.activation = activation
        self.dropout = dropout

    def forward(self, x, training=False):
        for i in range(len(self.mlp)):
            x = self.mlp[i](x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x.transpose(1,2))
                x = x.transpose(1,2)
            if self.activation is not None:
                x = self.activation(x)
            if training and self.dropout > 0.:
                if len(x.shape) == 3:
                    # x: [nB, nL, ndim]
                    dropout_dim = DropoutDim()
                    x = dropout_dim(x, 1, self.dropout)
                else:
                    # x: [nB, nL/ndim]
                    x = F.dropout(x, p=self.dropout, training=training)
        return x
