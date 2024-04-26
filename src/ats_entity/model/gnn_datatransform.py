from __future__ import unicode_literals, print_function, division
import torch
from model.datautil import get_cumsum, index_to_gather

class DependencyIndexer():
    def __call__(self, mask, sparse_mask=None):
        '''
            Make batch-offsetted and flatten edge pairs.
            In case of exclude_ending_stopdot True, data must be dependency head.
        '''
        actual_lens = torch.sum(mask, dim=1, keepdim=True, dtype=torch.int64)
        cumsum = get_cumsum(mask, use_mask=False,
                            prepad=torch.zeros((1), dtype=torch.long, device=mask.device))

        # Gather indexer by mask
        indexer = index_to_gather(mask=mask, sparse_mask=sparse_mask)

        return {"indexer": indexer,
                "cumsum": cumsum,
                "actual_lens": actual_lens}


class LastStopDotExclusionDependencyIndexer(DependencyIndexer):
    def __call__(self, data, mask, sparse_mask=None):
        index_info = super().__call__(data, mask, sparse_mask)
        cumsum = index_info["cumsum"]
        actual_lens = index_info["actual_lens"]
        indexes_head_to_stopdot = self.get_index_head_to_stopdot(data, cumsum,
                                                                 actual_lens)
        stopdot_indexes = self.get_index_to_stopdot(cumsum, actual_lens)
        # Concat all indices to be excluded
        indexes_excl = torch.cat((indexes_head_to_stopdot, stopdot_indexes), dim=0)
        indexes_excl = indexes_excl.sort().values
        # Exclude from heads
        indexer = index_info["indexer"]
        gathered = list(set(indexer.tolist()) - set(indexes_excl.tolist()))
        indexer = torch.tensor(gathered, dtype=indexer.dtype, device=data.device)
        return {"indexer": indexer,
                "cumsum": cumsum,
                "actual_lens": actual_lens-1}

    def get_index_to_stopdot(self, cumsum, lens_per_samples):
        '''
            Get index of ending stopdot.
        '''
        last_indexes = lens_per_samples - 1
        last_indexes_offsetted = last_indexes+cumsum[:,None][:-1]
        last_indexes_offsetted = last_indexes_offsetted.flatten()
        return last_indexes_offsetted

    def get_index_head_to_stopdot(self, head, cumsum, lens_per_samples):
        '''
            Get index of head that points to ending stopdot.
        '''
        last_indexes = lens_per_samples - 1
        bool_m = head==last_indexes
        row_indicators, indexes_excl = bool_m.nonzero(as_tuple=True)
        for i, offset in enumerate(cumsum[:-1]):
            row_indicators[row_indicators==i] = offset # Replace index by offset
        indexes_excl_offsetted = row_indicators+indexes_excl
        return indexes_excl_offsetted


class GNNDataTransform():
    def debatch(self, x, indexer):
        '''
            x:          [nb, nl, nd]
            indexer:    1D tensor, by which x elements are selected.
        '''
        nb, nl, nd = x.size()
        x = x.reshape(nb*nl,-1)
        if indexer is not None:
            x = torch.index_select(x, dim=0, index=indexer)
        return x

    def assign_rebatch(self, x, indexer, val, inplace=False):
        '''
            x:          Original data to GNN.
            indexer:    1D tensor, by which x elements are selected.
            val:        GNN representation learning output.
        '''
        nb, nl, nd = x.size()
        x = x.reshape(nb*nl,-1)
        if inplace:
            x.index_copy_(0, indexer, val)
        else:
            x = x.index_copy(0, indexer, val)
        x = x.reshape(nb, nl, -1)
        return x

    def make_edges(self, head, head_indexer, cumsum,
                    order=0, initial_start=0):
        '''
            Make batch-offsetted and flatten edge pairs
        '''
        # Offset head indices
        hd = head + cumsum[:-1][:,None]
        hd = hd.reshape(-1) # Flatten
        # Dependents
        # Gather actual head index values pointed/indexed to by the head_indexer
        hd_gathered = torch.index_select(hd, dim=0, index=head_indexer)
        dep_gathered = head_indexer
        # Pair up
        dep_pair = (hd_gathered, dep_gathered) if order == 0 \
                                                else (dep_gathered, hd_gathered)
        edges = torch.stack(dep_pair, dim=0) + initial_start
        return edges
