from __future__ import unicode_literals, print_function, division
import numpy as np
import torch

srng = np.random.RandomState(seed=20210919)

class DropoutDim():
    '''
        Dropout along a specified dimension
    '''
    def __call__(self, x, dim, p):
        ndim = len(x.shape)
        if dim < 0:
            dim = ndim + dim
        size = x.shape[:dim+1]
        mask = self.dropout_calc(size, p)
        mask = torch.tensor(mask, dtype=torch.float32, device=x.device)
        dims = list(range(ndim-dim-1))
        for _ in dims:
            mask = mask.unsqueeze(-1)
        x = x*mask
        return x

    def dropout_calc(self, size, rate = 0.0):
        mask = srng.binomial(n=1, p=(1-rate), size=list(size))
        return mask

def cumulative(values):
    '''
        Compute a list of integer values into cumulative sums.
    '''
    length = len(values)
    values = [sum(values[0:x:1]) for x in range(0, length+1)]
    return values

def fill_ignore_value(x, mask, ignore_value=-100):
    ignores = (1. - mask) * ignore_value
    x_ = x*mask + ignores.long()
    x_ = x_.contiguous()
    return x_

def get_cumsum(mask, use_mask, prepad=None):
    '''
        x: [nb, nd, nl]
    '''
    if use_mask:
        lens_per_sample = torch.sum(mask, dim=1)
    else:
        lens_per_sample = [mask.shape[-1]]*mask.shape[0]
        lens_per_sample = torch.tensor(lens_per_sample, device=mask.device)
    cumsum = torch.cumsum(lens_per_sample, dim=0)
    if prepad is not None:
        cumsum = torch.cat((prepad,cumsum), dim=0)
    return cumsum

def index_to_gather(mask, sparse_mask=None, flat=True, as_tuple=False):
    '''
        mask: The dense mask corresponding to actual length.
        sparse_mask: The mask of some positions in the dense mask is of interest.
    '''
    which = mask if sparse_mask is None else sparse_mask
    if flat:
        index = torch.nonzero(which.reshape(-1), as_tuple=as_tuple)
        index = index[0] if as_tuple else index.reshape(-1)
    else:
        index = torch.nonzero(which, as_tuple=as_tuple)
    return index

def flat_gather(source, mask):
    '''
        Gather those masked values in a flattened dense form.
    '''
    src_index_flat = torch.nonzero(mask.reshape(-1), as_tuple=False).reshape(-1)
    if len(source.shape) <= 2:
        src_gathered = torch.gather(source.reshape(-1), -1, src_index_flat)
    elif len(source.shape) == 3:
        nb,nl,nd = source.size()
        source = source.reshape(nb*nl, -1)
        src_gathered = source[src_index_flat]
    return src_gathered

def covert_word_to_token_indices(
    token_mask,
    token_mask_mask,
    indices,
    indices_mask,
    indices_mask_mask,
    token_span_count, # subword_span
    token_span_count_mask,
    flat_batch=True,
    indices_offset_n=0,
    batch_offset_required=False,
    ignore_value=-100):
    '''
        Map the word-level indices to the corresponding token-level first token indices of
        words tokenized by the word-segmentation token encoding methods e.g. byte-pair encoding.

        Parameters:
            token_mask:
                The first token mask of each word in the model tokenized sequence.
            token_mask_mask:
                Token batch mask.
            indices:
                out of order word indices and may be a subset of word indices.
                Note that it differs from sparsify_indices.
            indices_mask:
                The first word mask of each entity (by indices).
                An entity is formed of an one or more words.
            indices_mask_mask:
                Indices batch mask.
            token_span_count:
                Token (model tokenized tokens) span of each word.
            token_span_count_mask:
                Token sequence batch mask.
            indices_offset_n:
                Mapping to tokens starts after the model takenizer's prepended tokens (e.g. BOS) 
                assuming that the prepended tokens are single-token-per-word.
                So, right-shift word-level indices by indices_offset_n.
            batch_offset_required:
                Offset indices by batch-wide sequence length if True.
                Has an effect only when flat_batch is True.

        Return:
            A tensor of token indices, and a mask to identify indices' batch dimensions.
    '''
    indices = indices + indices_offset_n

    # Each entry of the token span cumsums corresponds to the first token index (of a word)
    # while the indexing position to the token_span_count_cumsums 'list' itself has one-to-one 
    # mapping to input word-level indices.
    # Thus, acquiring token indices from token_span_count_cumsums can be done
    # simply by using the input word-level indices to slice it.
    # For example, use 2 in the input word-level indices to get the first token index
    # by token_span_count_cumsums[2].
    token_span_count_cumsums = torch.cumsum(token_span_count*token_span_count_mask, dim=-1)
    # Indexing starts from zero.
    pre_pad = torch.zeros((token_span_count_cumsums.shape[0],1),
                          dtype=token_span_count_cumsums.dtype,
                          device=token_span_count_cumsums.device)
    token_span_count_cumsums = torch.cat((pre_pad, token_span_count_cumsums), dim=-1)
    # Ensure not use batch padding values as valid indices.
    valid_size = torch.sum(indices_mask_mask, dim=-1, keepdim=True)

    # Slice indices within the range bound by valid size (i.e. excluding padding at the end).
    nb, _ = token_mask.size()
    token_indices = [token_span_count_cumsums[ib,indices[ib,:valid_size[ib]]] for ib in range(nb)]

    indices_sizes = [indices.shape[0] for indices in token_indices]

    if flat_batch:
        # Create mask to match indices to their batch dimensions, similar to torch.nonzero.
        mask = torch.tensor(list(range(nb)))
        mask = mask.repeat_interleave(torch.tensor(indices_sizes), dim=0)
        if batch_offset_required:
            pre_pad = torch.zeros((1,), dtype=token_span_count_cumsums.dtype, device=token_span_count_cumsums.device)
            batch_offset = get_cumsum(token_mask_mask, use_mask=False, prepad=pre_pad)[:,None]
            token_indices = [token_indices[ib] + batch_offset[ib] for ib in range(nb)]
        token_indices = torch.cat(token_indices, dim=0)
    else:
        max_size = token_mask.shape[-1]
        mask = torch.ones_like(token_mask)
        padded = []
        for ib in range(nb):
            mask[ib, indices_sizes[ib]:] = 0
            pad = torch.ones(max_size-indices_sizes[ib],
                             dtype=token_indices[ib].dtype,
                             device=token_indices[ib].device)*ignore_value
            padded.append(torch.cat((token_indices[ib], pad), dim=-1))
        token_indices = torch.stack(padded, dim=0)

    return token_indices, mask


def get_unique_index(values, indices):
    '''
        Input values should be aleady sorted.
    '''
    _, inv_idx, counts = torch.unique(values, dim=0, sorted=True, return_inverse=True, return_counts=True)
    _, ind_sorted = torch.sort(inv_idx, stable=True)
    cumsum = torch.cat((torch.tensor([0], device=counts.device), counts.cumsum(0)[:-1]))
    # First occurring duplicate location.
    unique_indicies = ind_sorted[cumsum]
    indices = indices.view(-1)[unique_indicies]
    return indices, inv_idx


def get_keys_by_prefix(kwargs, prefix, pop=True):
    keys = [k for k in kwargs.keys() if prefix in k]
    kwargs = {k: kwargs.pop(k) if pop else kwargs.get(k) for k in keys}
    return kwargs
