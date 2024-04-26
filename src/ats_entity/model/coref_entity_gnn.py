from __future__ import unicode_literals, print_function, division
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from torch_geometric.nn import GeneralConv
from model.gnn_datatransform import GNNDataTransform

class EntityAggrNet(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.edge_feature_emb = nn.Embedding(options["depth_size"],
                                            options["depth_dim"],
                                            padding_idx=options["depth_emb_pad"]) \
                                    if options.edge_feature else None
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
                                l2_normalize=options["l2_normalize"],
                                bias=True,
                                flow="target_to_source")
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(options["dims"][1]))

        self.transform = GNNDataTransform()

    def forward(self, inputs):
        '''
            Compute graph representation of sequence dependency structure.

            transform:              Data transformation object.
            data:                   Latent data.
            edge:               Each edge is between head word and subsequent words of an entity.
            edge_feature:       Edge features.
        '''
        data = inputs["data"]
        edges = inputs["edge"]
        edge_feature_id = inputs.get("edge_feature", None)
        apply_edge_feature = edge_feature_id is not None and self.edge_feature_emb is not None
        edge_features = self.edge_feature_emb(edge_feature_id) if apply_edge_feature else None
        data_g = self.transform.debatch(data, indexer=None)
        repr = self._net(x=data_g,
                        edge_index=edges,
                        edge_attr=edge_features)
        repr = repr.reshape(data.shape)
        return repr

    def _net(self, x, edge_index, edge_attr):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index, edge_attr)))
        return x
