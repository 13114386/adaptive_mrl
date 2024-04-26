from __future__ import unicode_literals, print_function, division
'''
    Adopt the article "Deep Biaffine Attention for Neural Dependency Parsing"
'''
import torch
import torch.nn as nn


class BiAffineTransform(nn.Module):
    def __init__(self, h_dim: int, s_dim: int, n_label: int,
                  h_bias=False, s_bias=False,
                  activation=None, init_value=0.):
        super().__init__()
        h_dim += s_bias
        s_dim += h_bias
        self.h_dim, self.s_dim, self.n_label = h_dim, s_dim, n_label
        self.h_bias, self.s_bias = h_bias, s_bias
        self.activation = activation
        self.U = nn.Parameter(torch.FloatTensor(n_label, h_dim, s_dim))
        nn.init.constant_(self.U, init_value)

    def forward(self, h, s):
        '''
            h: [nB, nL, h_dim]
            s: [nB, nL, s_dim]
        '''
        if len(h.shape) == 3:
            if self.h_bias:
                s = torch.cat([s, s.new_ones(s.shape[:-1]).unsqueeze(-1)], dim=-1)
            if self.s_bias:
                h = torch.cat([h, h.new_ones(h.shape[:-1]).unsqueeze(-1)], dim=-1)
        else:
            assert True, "BiAffineTransform need to implement the case"
        # [nB, 1, nL, nDim]
        h = h.unsqueeze(1)
        s = s.unsqueeze(1)
        # Using left matmul @ operator
        x = h @ self.U.to(device=s.device)
        x = x @ s.transpose(-1,-2) # [nB, n_label, nL, nL]
        x = x.permute(1,-2,-1,0) # transpose -> [n_label, nL, nL, nB]
        x = x.squeeze(0) # for n_label == 1
        if len(x.shape) == 4:
            x = x.squeeze(1)
        if self.activation is not None:
            x = self.activation(x)
        return x
