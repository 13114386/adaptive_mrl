from __future__ import unicode_literals, print_function, division
"""
    Basic custom representation models.
"""
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F # called by eval()
from model.mlp import MLP

class RepresentationModel(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(
        self,
        indices: Optional[torch.LongTensor],
        initial_states: torch.FloatTensor = None,
        **kwargs
    ) -> torch.FloatTensor:
        raise NotImplementedError("Representation base class forward method is not implemented.")


class IndexedStateRepresentationModel(RepresentationModel):
    def forward(
        self,
        indices: Optional[torch.LongTensor],
        initial_states: torch.FloatTensor = None,
        **kwargs
    ) -> torch.FloatTensor:
        nb, nl, _ = initial_states.size()
        flat_latent_states = initial_states.view(nb*nl, -1)
        r = flat_latent_states[indices]
        return r


class AttributedStateRepresentationModel(IndexedStateRepresentationModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation=None,
        batch_norm=True,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        act_f = eval(f"F.{activation}") if activation is not None else None
        self.mlp = MLP([in_channels, out_channels],
                       activation=act_f,
                       batch_norm=batch_norm,
                       dropout=dropout)

    def forward(
        self,
        indices: Optional[torch.LongTensor],
        initial_states: torch.FloatTensor = None,
        **kwargs
    ) -> torch.FloatTensor:
        r = super().forward(indices, initial_states)
        attrib_states = kwargs.pop("ar", None)
        assert attrib_states is not None, f"{self.__class__} got None attributes."
        r = torch.cat((r, attrib_states), dim=-1)
        if len(r.shape) < 3:
            r = r[None,...]  # Forge a batch dimension
        r = self.mlp(r)
        return r.squeeze(0)


class EmbeddingRepresentationModel(RepresentationModel):
    def __init__(
        self,
        n_embeddings: int = 1024,
        n_dims: int = 512,
        padding_idx: int = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=n_embeddings,
                                        embedding_dim=n_dims,
                                        padding_idx=padding_idx)

    def forward(
        self,
        indices: Optional[torch.LongTensor],
        initial_states: torch.FloatTensor = None,
        **kwargs
    ) -> torch.FloatTensor:
        d_offset = self.embedding.padding_idx+1 \
                    if self.embedding.padding_idx is not None \
                    else 0
        indices = indices + d_offset
        rr = self.embedding(indices)
        return rr


class PairedRelationRepresentationModel(RepresentationModel):
    def __init__(
        self,
        in_channels,
        out_channels,
        activation=None,
        batch_norm=True,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        act_f = eval(f"F.{activation}") if activation is not None else None
        self.mlp = MLP([in_channels, out_channels],
                       activation=act_f,
                       batch_norm=batch_norm,
                       dropout=dropout)

    def forward(
        self,
        indices: Optional[torch.LongTensor],
        initial_states: torch.FloatTensor = None,
        **kwargs
    ) -> torch.FloatTensor:
        attrib_states = kwargs.pop("ar", None)
        assert attrib_states is not None, f"{self.__class__} got None attributes."
        r = torch.cat(attrib_states, dim=-1)
        if len(r.shape) < 3:
            r = r[None,...]  # Forge a batch dimension
        r = self.mlp(r)
        return r.squeeze(0)
