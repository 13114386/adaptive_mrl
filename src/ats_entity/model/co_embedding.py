
from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from model.modelutil import create_embedding

class CoEmbedding(nn.Module):
    def __init__(
        self,
        attr_struct,
        type_vocab,
        animacy_vocab,
        number_vocab,
        gender_vocab,
        key_order,
        logger
    ):
        '''
            type_vocab, animacy_vocab, gender_vocab and number_vocab are called by eval().
        '''
        super().__init__()
        pad_id = attr_struct['pad']
        attr_dim = attr_struct['dim']
        self.order_embeddings = nn.ModuleList()
        for k in key_order:
            emb = create_embedding("coref_"+k, eval(k+"_vocab"), pad_id, attr_dim, logger)
            self.order_embeddings.append(emb)

    def forward(self, inputs):
        emb_states = []
        for i, embedding in enumerate(self.order_embeddings):
            e = embedding(inputs[:,i])
            emb_states.append(e)
        state = torch.cat(emb_states, dim=-1)
        return state
