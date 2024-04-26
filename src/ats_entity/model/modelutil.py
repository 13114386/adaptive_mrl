from __future__ import unicode_literals, print_function, division
import torch.nn as nn


def create_embedding(vocab_name, vocab, pad_id, dim, logger):
    '''
        pad_id: padding id (string form)
        dim:    embedding dimension
    '''
    padding_idx = vocab[pad_id] if pad_id in vocab.get_stoi() else None
    embedding = nn.Embedding(num_embeddings=len(vocab),
                            embedding_dim=dim,
                            padding_idx=padding_idx)
    if padding_idx is None:
        logger.warning(f"Embedding vocab ({vocab_name}) is missing specified padding value {pad_id}.")
    return embedding


def query_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size = (param_size + buffer_size) / 1024**2 # in MB unit
    return size
