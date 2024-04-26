from __future__ import unicode_literals, print_function, division
from copy import deepcopy
import itertools
import numpy as np
import torch
from model.datautil import (
    index_to_gather,
    cumulative,
    get_cumsum,
    get_unique_index
)

sanity_check=False

def find_indices_of_value(data, value, cumsum=False):
    array = np.array(data)
    indices = np.where(array == value)[0]
    indices = list(indices)
    if cumsum:
        indices += [array.shape[-1]]
    return indices


def get_sparse_token_index_by_span(
    token_mask,
    token_mask_mask,
    span_indices
):
    '''
        Use word level span indices to select token level indices.
    '''
    # 1. Get the sparse token leading subword token locations, and the cumsum of actual words along batch.
    token_loc = index_to_gather(token_mask*token_mask_mask)

    # 2. Slice
    selected_indices = token_loc[span_indices]
    return selected_indices


def make_mention_ctxt_mask(
    coref_index,
    coref_sent_num,
    coref_repr_mask,
    coref_repr_mask_mask,
    coref_sent_num_mask,
    sentence_cumsum,
    token_mask,
    token_mask_mask,
    span_last=True,
    include_ante=False
):
    '''
        Create token level span mask from word level spans.
    '''
    # Acquire max length among batched sequences.
    batch_ctxt_masks = []
    batch_mentions = []
    nbatch, max_seq_len = token_mask.size()
    for i in range(nbatch):
        sentence_span = list(zip(sentence_cumsum[i][0:-1], sentence_cumsum[i][1:]))
        # Work out coref mention contextul spans.
        assert coref_sent_num_mask is not None
        sent_num_splits = find_indices_of_value(coref_sent_num_mask[i], 1, cumsum=True)
        coref_repr_mask_sums = torch.sum(coref_repr_mask_mask[i], dim=-1, keepdim=True)
        repr_splits = find_indices_of_value(coref_repr_mask[i,:coref_repr_mask_sums].cpu(), 1, cumsum=True)

        mention_pairs = []
        mention_ctxt_mask = torch.zeros((max_seq_len, max_seq_len),
                                        dtype=token_mask.dtype,
                                        device=token_mask.device)
        for (ss, se), (rs, re) in zip(zip(sent_num_splits[:-1], sent_num_splits[1:]), \
                                        zip(repr_splits[:-1], repr_splits[1:])):
            # Skip antecedent not having mentions.
            men_start = 0 if include_ante else 1
            mention_index = coref_index[i][rs:re]
            if len(mention_index[men_start:]) == 0:
                continue

            ctxt_mask = torch.zeros(max_seq_len, dtype=token_mask.dtype, device=token_mask.device)
            mention_mask = torch.zeros(max_seq_len, dtype=token_mask.dtype, device=token_mask.device)
            # Unravel span range to indices.
            sent_num = coref_sent_num[i][ss:se]
            coref_span = [sentence_span[n-1] for n in sent_num] # Sentence span boundry indices
            coref_span_indices = [list(range(*span)) for span in coref_span] # Unravel indices within the boundry.
            coref_span_indices = list(itertools.chain(*coref_span_indices))

            # Mask for spans
            sparse_span_indices = get_sparse_token_index_by_span(
                                        token_mask[i,:][:,None],
                                        token_mask_mask[i,:][:,None],
                                        torch.tensor(coref_span_indices, device=token_mask.device)
                                    )
            ctxt_mask[sparse_span_indices] = 1

            # Mask for mentions.
            sparse_mention_indices = get_sparse_token_index_by_span(
                                        token_mask[i,:][:,None],
                                        token_mask_mask[i,:][:,None],
                                        mention_index[men_start:])
            mention_mask[sparse_mention_indices] = 1

            if span_last:
                ctxt_mask = ctxt_mask[:,None].expand(-1, max_seq_len).transpose(0,1)
                mention_mask = mention_mask[:,None].expand(-1, max_seq_len)
            else:
                ctxt_mask = ctxt_mask[:,None].expand(-1, max_seq_len)
                mention_mask = mention_mask[:,None].expand(-1, max_seq_len).transpose(0,1)

            # Create mention's span contextual mask.
            mention_ctxt_mask += ctxt_mask*mention_mask

            # We also collect antecedent and mention pairs.
            sparse_antecedent_index = get_sparse_token_index_by_span(
                                        token_mask[i,:][:,None],
                                        token_mask_mask[i,:][:,None],
                                        mention_index[0])
            mention_pairs.append((sparse_antecedent_index, sparse_mention_indices))

        batch_ctxt_masks.append(mention_ctxt_mask)
        batch_mentions.append(mention_pairs)

    # Batch them into tensor
    batch_ctxt_mask = torch.stack(batch_ctxt_masks, dim=0)
    return batch_ctxt_mask, batch_mentions


def calc_entity_span_cumsums(coref_entity_span, coref_entity_span_mask):
    coref_entity_span_cumsums = torch.cumsum(coref_entity_span*coref_entity_span_mask, dim=-1)
    pre_pad = torch.zeros((coref_entity_span_cumsums.shape[0],),
                          dtype=coref_entity_span_cumsums.dtype,
                          device=coref_entity_span_cumsums.device)
    coref_entity_span_cumsums = torch.cat((pre_pad[:,None], coref_entity_span_cumsums), dim=-1)
    batch_cumsums = get_cumsum(mask=coref_entity_span_mask,
                               use_mask=False,
                               prepad=torch.zeros(1, dtype=coref_entity_span_mask.dtype, device=coref_entity_span_mask.device))
    coref_entity_span_cumsums = coref_entity_span_cumsums + batch_cumsums[:,None][:-1]

    # Ensure only use valid span values.
    coref_entity_span_count = torch.sum(coref_entity_span_mask, dim=-1, keepdim=True)
    mask = torch.arange(coref_entity_span_count.shape[0], device=coref_entity_span_count.device)
    mask = torch.repeat_interleave(input=mask, repeats=coref_entity_span_count.view(-1), dim=0)
    # Gather valid entity heads
    nb, _ = coref_entity_span.size()
    coref_entity_span_cumsums = torch.cat([coref_entity_span_cumsums[ib,:coref_entity_span_count[ib]] for ib in range(nb)], dim=0)
    return coref_entity_span_cumsums, mask


def offset_key_to_head(
    coref_index,
    coref_index_mask,
    coref_entity_span,
    coref_entity_span_mask,
    coref_key_index,
    coref_key_index_mask,
):
    assert torch.all(torch.sum(coref_entity_span_mask, dim=-1).eq(torch.sum(coref_key_index_mask, dim=-1)))
    # Each entry of the span cumsums is the entity leading word position index into coref_index.
    coref_entity_span_cumsums = torch.cumsum(coref_entity_span*coref_entity_span_mask, dim=-1)
    pre_pad = torch.zeros((coref_entity_span_cumsums.shape[0],),
                          dtype=coref_entity_span_cumsums.dtype,
                          device=coref_entity_span_cumsums.device)
    coref_entity_span_cumsums = torch.cat((pre_pad[:,None], coref_entity_span_cumsums), dim=-1)

    # Ensure only use valid span values.
    coref_entity_span_count = torch.sum(coref_entity_span_mask, dim=-1, keepdim=True)

    # Compute key index offset to the leading index.
    nb, _ = coref_index.size()
    key2head_offset = [coref_key_index[ib,:coref_entity_span_count[ib]] - \
                       coref_index[ib,coref_entity_span_cumsums[ib,:coref_entity_span_count[ib]]] \
                          for ib in range(nb)]
    key2head_offset = torch.cat(key2head_offset, dim=0)
    # Mask batch
    bcounts = torch.sum(coref_entity_span_mask, dim=-1)
    mask = torch.tensor(list(range(bcounts.shape[0])), device=bcounts.device)
    mask = mask.repeat_interleave(bcounts, dim=0)
    return key2head_offset, mask


def get_graph_entity_key_info(
    entity_index,
    entity_mask,
    entity_head_offset
):
    '''
        Parameters:
            entity_index:
                Entity index array.
            entity_mask:
                Entity first word mask.
            entity_head_offset:
                Head word offset relative to the first word of each entity.

        Return:
            entity_head:
                Head word index.
            entity_head_loc:
                Head word location w.r.t. entity_index array.
            head_loc_offset:
                Head word offset w.r.t. the first word of each entity.

        Note:
            The entity_index may not be contiguous for each entity if they are the token indices instead of word indices.
    '''
    # First word as default keyword. 
    entity_head_loc = torch.nonzero(entity_mask.view(-1), as_tuple=True)[0]

    if entity_head_offset is not None:
        # Use keyword as head.
        entity_head_loc = entity_head_loc + entity_head_offset
        entity_head = entity_index.view(-1)[entity_head_loc]
        head_loc_offset = entity_head_offset
    else:
        # Use first word as head.
        entity_head = entity_index.view(-1)[entity_head_loc]
        head_loc_offset = torch.zeros_like(entity_head)

    return (entity_head, entity_head_loc, head_loc_offset)

class CorefGraphBuilder():
    def __init__(
        self,
        ngram_bound=None,
        ignore_value=-100,
        depth_padding_id=0
    ):
        self.ngram_bound = ngram_bound
        self.ignore_value = ignore_value
        self.depth_padding_id = depth_padding_id

    @staticmethod
    def build_mention_edge(
        batch_mentions,
        token_mask_mask,
        ignore_value=-100,
        depth_padding_id=0,
        return_tensor=True
    ):
        edge_batch = []
        depth_batch = []
        mask_batch = []
        nB, nLens = token_mask_mask.size()
        for iB, (mentions, nL) in enumerate(zip(batch_mentions, [nLens]*nB)):
            edges = torch.tensor([ignore_value] * nL, device=token_mask_mask.device)
            depths = torch.tensor([depth_padding_id] * nL, device=token_mask_mask.device)
            for ante, coref in mentions:
                edges[coref] = ante
                depths[ante] = depth_padding_id
                depths[coref] = coref - ante
            edge_batch.append(edges.tolist())
            depth_batch.append(depths.tolist())
            mask_batch.append((edges > ignore_value).long().tolist())

        if return_tensor:
            edge_batch = torch.tensor(edge_batch, dtype=token_mask_mask.dtype, device=token_mask_mask.device)
            depth_batch = torch.tensor(depth_batch, dtype=token_mask_mask.dtype, device=token_mask_mask.device)
            mask_batch = torch.tensor(mask_batch, dtype=token_mask_mask.dtype, device=token_mask_mask.device)
        return edge_batch, depth_batch, mask_batch

    def __call__(
        self,
        coref_index,
        coref_sent_num,
        coref_repr_mask,
        coref_repr_mask_mask,
        coref_entity_mask,
        coref_entity_span,
        coref_sent_num_mask,
        token_sentence_sizes,
        token_mask,
        token_mask_mask,
        coref_key_offset=None,
        coref_key_offset_mask=None,
        include_ante=False,
        edge_builder=None,
        return_tensor=True
    ):
        '''
            Create coreference pairing (referent, references) and their sentence-level context spans.

            Note that the caller has to make sure coref indices mapped to 
            either word-level or token-level accordingly before or after the call.

            Parameters:
                coref_index:
                    Identifies coreference (referent/reference) entity word (by indices) in a document.
                coref_repr_mask:
                    Identifies which indices are referent entities and the references otherwise.
                coref_repr_mask_mask:
                    Batch mask.
                coref_entity_mask:
                    Masks the first word of each (single-/multi-word) entity indices.
                coref_entity_span:
                    The number of words per entity.
                coref_sent_num_mask:
                    Masks which sentence numbers associate with each referent's coreference context.
                token_sentence_sizes:
                    Sentence size in terms of tokens instead of words.
                token_mask:
                    Mask first token of each word.
                token_mask_mask:
                    Batch mask.
                coref_key_offset:
                    Identifies each entity keyword's position relative to the first word of the entity.
                coref_key_offset_mask:
                    Mask coref_key_offset by mini-batch indices.
                include_ante:
                    A flag of whether or not include referent in its references.

            Note:
                Each coreference boundary can be identified by
                coref_repr_mask*coref_entity_mask and coref_sent_num_mask.
        '''
        batch_context = []
        batch_mentions = []
        batch_entity_spans = []
        nbatch, _ = token_mask.size()

        token_sent_sizes = deepcopy(token_sentence_sizes) \
                            if token_sentence_sizes is not None else None

        # For each coreference entities, only mask the first word of referent entity to
        # set the boundary of each referent's coreference context.
        coref_repr_1_mask = coref_repr_mask*coref_entity_mask

        for i in range(nbatch):
            # Build sentence start and end index pair list.
            token_sentence_cumsum = cumulative(token_sent_sizes[i])
            token_sentence_span = list(zip(token_sentence_cumsum[0:-1], token_sentence_cumsum[1:]))

            assert coref_sent_num_mask is not None
            # For each referent instance, mark the range in the list of the coreference sentence numbers.
            sent_num_splits = find_indices_of_value(coref_sent_num_mask[i], 1, cumsum=True)
            # For each referent instance, mark the referent's first word/token index positions by the coreference mask.
            coref_repr_mask_sums = torch.sum(coref_repr_mask_mask[i], dim=-1, keepdim=True)
            repr_splits = find_indices_of_value(coref_repr_1_mask[i,:coref_repr_mask_sums].cpu(), 1, cumsum=True)

            context_spans = {}
            mention_pairs = []
            entity_spans = {}
            mention_n = 0
            key_offsets = coref_key_offset[coref_key_offset_mask==i] \
                            if coref_key_offset is not None and coref_key_offset_mask is not None else None

            # For each referent, we have the sentence numbers of its coreference context
            # and its coreference entities.
            for (ss, se), (rs, re) in zip(zip(sent_num_splits[:-1], sent_num_splits[1:]), \
                                            zip(repr_splits[:-1], repr_splits[1:])):
                ### Acquire the pairings of referent-references (mentions).
                # Get all mention entity indices of the coreference (referent-references) context.
                mention_index = coref_index[i,rs:re]

                # Get the starting locations of the coreference entities.
                coref_start = 0 if include_ante else torch.sum(coref_repr_mask[i,rs:re]).item()
                assert len(mention_index[coref_start:]) > 0 or include_ante, \
                        "No references for the referent."

                # Gather keyword index offsets for the coreference.
                n_mention = torch.sum(coref_entity_mask[i,rs:re])
                k_offset = key_offsets[mention_n:n_mention+mention_n] if key_offsets is not None else None
                entity_span = coref_entity_span[i][mention_n:n_mention+mention_n]
                mention_n += n_mention

                # Get each entity's head information.
                (mention_head_index, mention_head_loc, head_loc_offset) = \
                    get_graph_entity_key_info(
                        mention_index,
                        coref_entity_mask[i,rs:re],
                        k_offset
                    )

                # Make referent-references pairing.
                sparse_referent_index = mention_head_index[0]
                sparse_reference_indices = mention_head_index if include_ante else mention_head_index[1:]

                mention_pairs.append((sparse_referent_index, sparse_reference_indices))

                # Get sparse_referent_index's n_gram neighbouring word indices.
                e_spans = self.acquire_entity_span(
                    entity_index=mention_index,
                    entity_head=mention_head_index,
                    entity_head_loc=mention_head_loc,
                    head_loc_offset=head_loc_offset,
                    entity_span=entity_span,
                    ngram_bound=self.ngram_bound,
                    return_map=True
                )
                entity_spans.update(e_spans)

                ### Acquire sentence-level context spans.
                # Note that sentence number is one-based index and 
                # should be converted to zero-based index for indexing.
                sent_num = coref_sent_num[i][ss:se]
                sent_num = sorted(list(set(sent_num)))
                token_mention_span = [token_sentence_span[n-1] for n in sent_num] # A list of sentence spans.
                context_spans[sparse_referent_index.item()] = token_mention_span # The referent-context map.

            batch_mentions.append(mention_pairs)
            batch_context.append(context_spans)
            batch_entity_spans.append(entity_spans)


        if edge_builder is not None:
            mention_edges, mention_depths, mention_edge_mask = \
                    edge_builder(batch_mentions,
                                token_mask_mask=token_mask_mask,
                                ignore_value=self.ignore_value,
                                depth_padding_id=self.depth_padding_id,
                                return_tensor=return_tensor)
            return {"context": batch_context,
                    "edge": mention_edges,
                    "depth": mention_depths,
                    "edge_mask": mention_edge_mask,
                    "entity_span": batch_entity_spans}
        else:
            return {"context": batch_context,
                    "mention": batch_mentions,
                    "entity_span": batch_entity_spans}

    def acquire_entity_span(
        self,
        entity_index,
        entity_head,
        entity_head_loc,
        head_loc_offset,
        entity_span,
        ngram_bound=None,
        return_map=True
    ):
        '''
            Parameters:
                entity_index:
                    The word indices of entities.
                entity_head:
                    Head word index.
                entity_head_loc:
                    Head word location w.r.t. entity_index array.
                head_loc_offset:
                    Head word offset w.r.t. the first word of each entity.
                entity_span:
                    The number of words every entity.
                entity_head_offset:
                    The offsets of the keyword indices relative to the first word indices.
                ngram_bound:
                    The number of neighbouring words at each side of the head word if applicable.
        '''
        dep_count = entity_span.view(-1)

        # Make the head word as the center.
        dep_offset_list = [torch.arange(start=0, end=count.item(), device=entity_span.device) - loc_offset.item() \
                            for count, loc_offset in zip(dep_count, head_loc_offset)]

        if ngram_bound is not None:
            if isinstance(ngram_bound, (tuple, list)):
                lf_ngram, rt_ngram = ngram_bound[0], ngram_bound[1]
            elif isinstance(ngram_bound, int):
                lf_ngram = rt_ngram = ngram_bound
            else:
                raise ValueError("ngram_bound must be one of None, tuple, list or int.")
            # Update by ngram bound
            dep_offset_list = [offsets[(-lf_ngram<=offsets).logical_and(offsets<=rt_ngram)] \
                                for offsets in dep_offset_list]

        # Dependant word locations w.r.t. the head location.
        dependant_locs = [loc+offsets for loc, offsets in zip(entity_head_loc, dep_offset_list)]

        # Gather dependant indices.
        if return_map:
            entity_spans = {head.item(): entity_index.view(-1)[locs].tolist() for head, locs in zip(entity_head[:,None], dependant_locs)}
        else:
            entity_spans = [(head.item(), entity_index.view(-1)[locs].tolist()) for head, locs in zip(entity_head[:,None], dependant_locs)]
        return entity_spans


class EntityGraphBuilder():
    def __init__(self, ngram_bound=None):
        self.ngram_bound = ngram_bound

    def __call__(
        self,
        coref_index,
        coref_index_mask,
        coref_entity_mask,
        coref_entity_mask_mask,
        coref_entity_span,
        coref_entity_span_mask,
        coref_key_offset=None,
        coref_key_offset_mask=None,
        batch_offset_required=True,
        lone_entity_included=False,
        entity_head_self_loop=False,
        dep_to_head_direction=True,
    ):
        '''
            Build antecedent-coreferent pairs for a graph based representation learning
            to create a single representation of the multi-word entity.

            Output:
                antecedent-coreferent edge pairs.
                batch mask for the edges.
        '''
        # Gather the head word location/index of single- or multiple-word entity.
        bcounts = torch.sum(coref_entity_mask*coref_entity_mask_mask, dim=-1)
        mask = torch.tensor(list(range(bcounts.shape[0])), device=bcounts.device)
        bcounts = mask.repeat_interleave(bcounts, dim=0)

        # Compute the offset between the key word index of entity and the first word of the entity.
        assert coref_key_offset_mask is None or torch.all(coref_key_offset_mask.eq(bcounts))

        if sanity_check:
            # Note that entity_head_loc is batch-flattened locations offsetted by batch-wide sequence length.
            entity_head_loc = torch.nonzero((coref_entity_mask*coref_entity_mask_mask).view(-1), as_tuple=True)[0]
            coref_entity_span_cumsums, _ = calc_entity_span_cumsums(coref_entity_span, coref_entity_span_mask)
            assert torch.all(entity_head_loc.eq(coref_entity_span_cumsums))

        if batch_offset_required:
            cumsums = get_cumsum(mask=coref_entity_mask,
                               use_mask=False,
                               prepad=torch.zeros(1, dtype=coref_entity_mask.dtype, device=coref_entity_mask.device))
            # Offset indices before flattening mini-batch.
            coref_index = coref_index + cumsums[:,None][:-1]

        # counts of dependants including head.
        dep_count = coref_entity_span.view(-1)[(coref_entity_span_mask==1).view(-1)]

        (entity_head, entity_head_loc, head_loc_offset) = \
            get_graph_entity_key_info(
                entity_index=coref_index,
                entity_mask=coref_entity_mask*coref_entity_mask_mask,
                entity_head_offset=coref_key_offset
            )

        edges, edge_mask = self._make_edge(coref_index=coref_index,
                                mask=bcounts,
                                entity_head=entity_head,
                                entity_head_loc=entity_head_loc,
                                head_loc_offset=head_loc_offset,
                                dep_count=dep_count,
                                ngram_bound=self.ngram_bound,
                                lone_entity_included=lone_entity_included,
                                entity_head_self_loop=entity_head_self_loop,
                                dep_to_head_direction=dep_to_head_direction
                            )
        return edges, edge_mask, \
                (coref_index, entity_head, entity_head_loc, head_loc_offset, dep_count, bcounts)

    def _make_edge(
        self,
        coref_index,
        mask,
        entity_head,
        entity_head_loc,
        head_loc_offset,
        dep_count,
        ngram_bound,
        lone_entity_included,
        entity_head_self_loop,
        dep_to_head_direction
    ):
        '''
            coref_index: coref entity index w.r.t. doc.
            entity_head: entity's first word or key/head word index gathered from coref_index.
            entity_head_loc: the location indexing to entity_head.
            head_loc_offset: entity_head_loc offset w.r.t. the first word index of the entity.
            ngram_bound: Only take ngram centered around head/key word of entity.
        '''
        dep_range = torch.stack((mask, entity_head, entity_head_loc, head_loc_offset, dep_count), dim=1)
        if not lone_entity_included:
            # Skip single word entity.
            dep_mask = (dep_range[...,-1] > 1)
            dep_range = dep_range[dep_mask]

        dep_offset_list = [torch.arange(start=0, end=count.item(), device=dep_range.device) - loc_offset.item() \
                            for count, loc_offset in zip(dep_range[...,-1], dep_range[...,-2])]
        if ngram_bound is not None:
            if isinstance(ngram_bound, (tuple, list)):
                lf_ngram, rt_ngram = ngram_bound[0], ngram_bound[1]
            elif isinstance(ngram_bound, int):
                lf_ngram = rt_ngram = ngram_bound
            else:
                raise ValueError("ngram_bound must be one of None, tuple, list or int.")
            dep_offset_list = [offsets[(-lf_ngram<=offsets).logical_and(offsets<=rt_ngram)] \
                                for offsets in dep_offset_list]

        # Dependant locations w.r.t. the head location.
        dep_count = torch.tensor([item.shape[-1] for item in dep_offset_list], device=entity_head.device)
        dependant_locs = torch.repeat_interleave(input=dep_range[...,2], repeats=dep_count, dim=0)
        dep_offsets = torch.cat(dep_offset_list, dim=0)
        dependant_locs = dependant_locs + dep_offsets

        if sanity_check:
            # Actual dependant locations to entity head index.
            dependant_locs2 = [loc + offset for loc, offset in zip(dep_range[...,2], dep_offset_list)]
            dependant_locs2 = torch.cat(dependant_locs2, dim=-1)
            assert torch.all(dependant_locs2.eq(dependant_locs)).item(), \
                            "dependant_locs2 != dependant_locs"

        # Gather dependant indices.
        entity_dependants = coref_index.view(-1)[dependant_locs]

        # Expand head to match the number of its dependants.
        mask = torch.repeat_interleave(input=dep_range[...,0], repeats=dep_count, dim=0)
        entity_head = torch.repeat_interleave(input=dep_range[...,1], repeats=dep_count, dim=0)

        if not entity_head_self_loop:
            dep_pos_mask = (dep_offsets != 0)
            entity_dependants = entity_dependants[dep_pos_mask]
            entity_head = entity_head[dep_pos_mask] # Also exclude single word entity.
            mask = mask[dep_pos_mask]

        edge_pair = (entity_head, entity_dependants) if dep_to_head_direction \
                    else (entity_dependants, entity_head)
        edges = torch.stack(edge_pair, dim=0)
        return edges, mask


def uniquify_attributes(attrbitues, batch_counts):
    '''
        Remove duplication, i.e. attributes of repeated index.
    '''
    sizes = torch.bincount(batch_counts)
    cumsum = torch.cat((torch.tensor([0], device=sizes.device), sizes.cumsum(0))).tolist()

    uniq_attributes = []
    uniq_batch_counts = []
    for start, end in zip(cumsum[:-1], cumsum[1:]):
        # Sort by entity indices.
        values, indices = torch.sort(attrbitues[start:end][:,0][:,None], dim=0, descending=False, stable=True)
        # Get the locations of first occurring duplicate indices.
        uniq_loc, _ = get_unique_index(values, indices)
        # Slice them.
        uniq_attr = attrbitues[start:end][uniq_loc]
        uniq_attributes.append(uniq_attr)
        uniq_batch_counts.append(batch_counts[start:start+uniq_attr.shape[0]])

    # Tensors
    attrbitues = torch.cat(uniq_attributes, dim=0)
    batch_counts = torch.cat(uniq_batch_counts, dim=0)
    return attrbitues, batch_counts


def get_indexed_attributes(
    attributes,
    entity_head,
    entity_span_mask,
    entity_index=None,
    entity_mask=None,
    entity_mask_mask=None,
    entity_span=None,
    key_offset=None,
    key_offset_mask=None,
    batch_offset_required=False
):
    '''
        Bundle the attributes with their corresponding entity indices in form of
        (index, attr1, attr2, ...).

        Note:
            If entity_head is not None, we use it directly.
    '''
    if entity_head is None:
        # Offset coref index before flattening the batch.
        if batch_offset_required:
            batch_cumsums = get_cumsum(mask=entity_mask,
                                       use_mask=False,
                                       prepad=torch.zeros(1, dtype=entity_mask.dtype, device=entity_mask.device))
            entity_index = entity_index + batch_cumsums[:,None][:-1]

        # Gather valid entity heads
        (entity_head, _, _) = get_graph_entity_key_info(
                                entity_index=entity_index,
                                entity_mask=entity_mask*entity_mask_mask,
                                entity_head_offset=key_offset
                            )
    # Match up entity with attributes within masked range.
    indexed_attrbitues = torch.stack([entity_head]+[v[entity_span_mask==1].reshape(-1) for v in attributes], dim=-1)
    return indexed_attrbitues


def query_indexed_attributes(query, attributes):
    '''
        query must contain unique indices.
    '''
    queried_mask = attributes[:,0][:,None] == query
    queried_mask = torch.sum(queried_mask.long(), dim=-1, keepdim=False)
    query_loc = torch.nonzero(queried_mask).squeeze()
    result = attributes[query_loc]
    # Ensure the dimension consistency for both single and multiple query items.
    if len(result.shape) == 1:
        result = result[None,...]
    return result
