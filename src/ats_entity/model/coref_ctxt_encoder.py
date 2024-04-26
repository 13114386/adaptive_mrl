from __future__ import unicode_literals, print_function, division
import math
import torch
import torch.nn as nn
from model.mlp import MLP

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        output_size,
        apply_layernorm
    ):
        super().__init__()
        self.convd1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding="same")
        if apply_layernorm:
            self.layer_norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(output_size)
        self.apply_layernorm = apply_layernorm

    def forward(self, inputs):
        x = inputs.transpose(1,2) # [nb, nl, nd] -> [nb, nd, nl]
        x = self.convd1(x)
        if self.apply_layernorm:
            x = x.transpose(1,2) # [nb, nd, nl] -> [nb, nl, nd]
            x = self.layer_norm(x)
            x = x.transpose(1,2) # [nb, nl, nd] -> [nb, nd, nl]
        x = self.activation(x)
        x = self.pool(x)
        return x.transpose(1,2)

class CnnModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels=config.in_channels
        filter_sizes=config.filter_sizes
        num_filters=config.num_filters
        output_size_factor=config.output_size_factor
        apply_layernorm=config.layernorm
        dropout=config.dropout
        max_length=config.max_length

        if isinstance(in_channels, (list, tuple)):
            assert len(in_channels) == 2, "Channel list/tuple should be size of 2."
            self.mlp = MLP(in_channels, activation=nn.ReLU())
            in_channels = in_channels[-1]
        else:
            self.mlp = None
        self.filter_layers = nn.ModuleList()
        n_blocks = math.floor(math.log(max_length, output_size_factor))
        for sfilter, nfilter in zip(filter_sizes, num_filters):
            layers = nn.ModuleList()
            in_chs = in_channels
            for i in range(n_blocks):
                output_size = max_length//output_size_factor**(i+1)
                block = ConvBlock(in_channels=in_chs,
                                  out_channels=nfilter,
                                  kernel_size=sfilter,
                                  output_size=output_size,
                                  apply_layernorm=apply_layernorm)
                in_chs = nfilter
                layers.append(block)
            self.filter_layers.append(layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(sum(num_filters), sum(num_filters))
        # self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(
        self,
        input_states,
        attention_mask
    ):
        outputs = []
        if self.mlp is None:
            inputs = input_states
        else:
            inputs = self.mlp(input_states)
        for i, filter_layers in enumerate(self.filter_layers):
            x = inputs
            for i, layer in enumerate(filter_layers):
                x = layer(x)
            outputs.append(x)

        x = torch.cat([x.squeeze() for x in outputs], dim=-1)
        x = self.dropout(x)
        x = self.fc(x)
        # x = x.transpose(1,2)
        # x = self.avg_pool(x)
        # x = x.transpose(1,2)
        return x
