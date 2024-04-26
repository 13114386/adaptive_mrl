from __future__ import unicode_literals, print_function, division
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GeneralConv
from model.datautil import (flat_gather)
from model.gnn_datatransform import GNNDataTransform, DependencyIndexer

class SubwordNet(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.subword_edge_feature_emb = nn.Embedding(options["depth_size"],
                                                    options["depth_dim"],
                                                    padding_idx=options["depth_emb_pad"])
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(options["n_layers"]):
            conv = GeneralConv(in_channels=options["dims"][0],
                                    out_channels=options["dims"][1],
                                    in_edge_channels=options["depth_dim"],
                                    aggr=options["aggregation"],
                                    skip_linear=options["linearity"],
                                    directed_msg=True,
                                    attention=False,
                                    l2_normalize=False,
                                    bias=True,
                                    flow="target_to_source")
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(options["dims"][1]))

        self.transform = GNNDataTransform()
        self.dep_indexer = DependencyIndexer()

    def forward(self, inputs):
        '''
            Compute graph representation of sequence dependency structure.

            transform:              Data transformation object.
            dep_indexer:            Dependency data indexing object.
            data:                   Latent data.
            token_arc_head:         Sparsified dependency head of syntactic arc.
            token_arc_type:         Sparsified dependency type of syntactic arc.
            attention_mask:         Inlucde special tokens such as BOS and EOS.
                                    It is created by a tokenizer.
            token_dense_mask:       Mask out the special tokens from attention_mask.
            token_sparse_mask:      Sparse mask that mask the first token of word segmentation.
                                    It may include the special tokens.
        '''
        data = inputs["data"]
        attention_mask = inputs["attention_mask"]
        token_head = inputs["token_head"]
        token_depth = inputs["token_depth"]
        token_dense_mask = inputs["token_dense_mask"]
        token_sparse_mask = inputs["token_sparse_mask"]

        sparse_mask = token_sparse_mask*token_dense_mask
        results = self.dep_indexer(mask=attention_mask,
                                sparse_mask=sparse_mask)
        indexer = results["indexer"]
        cumsum = results["cumsum"]
        edges = self.transform.make_edges(token_head, indexer, cumsum)
        edge_features = self.subword_edge_feature_emb(token_depth)
        edge_features = flat_gather(edge_features, sparse_mask)
        data_g = self.transform.debatch(data, indexer=None)
        repr = self.subword_net(x=data_g,
                                edge_index=edges,
                                edge_attr=edge_features)

        repr = repr.reshape(data.shape)
        return repr

    def subword_net(self, x, edge_index, edge_attr):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        return x
