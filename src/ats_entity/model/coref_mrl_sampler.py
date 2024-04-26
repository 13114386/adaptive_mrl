from __future__ import unicode_literals, print_function, division
import random
import torch

sanity_check=False


def offset_sample(samples, offset, locs=None):
    offset_samples = samples.new_zeros(samples.shape)
    offset_samples.copy_(samples)
    if locs is None:
        offset_samples = samples + offset
    else:
        for loc in locs:
            offset_samples[:,loc] = samples[:,loc] + offset
    return offset_samples

class NegativeSampler():
    def build_negative_samples(
        self,
        referents,
        positives,
        relations=None,
        attributes=None
    ):
        '''
            referents:  The referents.
            positives:  The references.
            relations:  The relations between referents and references.
            attributes: The attribute tuples of referents and references.
        '''
        # Initialize negative samples as placeholder.
        negatives = positives.repeat_interleave(1, dim=0)
        negative_rels = relations.repeat_interleave(1, dim=0) if relations is not None else None
        negative_attrs = attributes.repeat_interleave(1, dim=0) if attributes is not None else None
        # List the number of samples of each referents.
        _, counts = torch.unique_consecutive(referents, dim=0, return_counts=True)
        # Turn the count list into accumulative count list.
        cumsums = torch.cat((torch.zeros(1, dtype=counts.dtype, device=counts.device),
                            torch.cumsum(counts, dim=0)))
        # Create index ranges for indexing.
        ranges = torch.arange(referents.shape[0], device=counts.device)

        for start, end in zip(cumsums[0:-1], cumsums[1:]):
            # The mask of query context to be replaced by negative context samples.
            query_mask_base = torch.zeros(ranges.shape,
                                        dtype=torch.bool,
                                        device=ranges.device)
            query_mask_base[start:end] = True
            # The mask of negative contexts to be sampled.
            neg_mask_base = torch.ones(ranges.shape,
                                        dtype=torch.bool,
                                        device=ranges.device)
            neg_mask_base[start:end] = False
            # Query context and negative context ranges.
            query_range, neg_range = ranges[query_mask_base], ranges[neg_mask_base]
            # Draw negative sample indices.
            query_range_size = query_range.shape[0]
            neg_range_size = neg_range.shape[0]
            if neg_range_size == 0: # Only exists a single entity referent without any references.
                # Positive samples as negative samples so that no cost.
                # Ideally should be excluded altogether.
                neg_indices = query_range
            elif neg_range_size >= query_range_size:
                # Enough negative contexts to draw from.
                neg_indices = sorted(random.sample(neg_range.tolist(), k=query_range_size))
            else:
                # Sample with replacement if there are not enough negative contexts to draw from.
                neg_indices = sorted(random.choices(neg_range.tolist(), k=query_range_size))

            # Replace initial values with negative samples.
            negatives[query_range] = positives[neg_indices]
            if relations is not None:
                negative_rels[query_range] = relations[neg_indices]
            if attributes is not None:
                negative_attrs[query_range] = attributes[neg_indices]

        return (negatives, negative_attrs, negative_rels)


class CorefCtxtNegativeSampler(NegativeSampler):
    def __call__(
        self,
        batch_triples,
        batch_offsets=None,
        batch_contexts=None,
        batch_attribs=None,
        **kwargs
    ):
        batch_referents = []
        batch_attributes = [] if batch_attribs is not None else None
        batch_positives = []
        batch_negatives = []
        batch_counts = []
        for ib, contexts in enumerate(batch_contexts):
            samples = [(key, v) for key, val in contexts.items() for v in val]
            samples = list(zip(*samples))
            # Referents/anchors
            referents = torch.tensor(samples[0], device=batch_triples[ib].device)[:,None]
            # Positive samples
            positives = torch.tensor(samples[1], device=referents.device)
            # Referent attributes
            attributes = None
            if batch_attribs is not None:
                attributes = batch_attribs[ib]
                referent_attribs, _ = attributes
                keys = torch.tensor([key for key, _ in contexts.items()],
                                    device=referent_attribs.device)
                attributes = self.get_key_attributes(keys, referent_attribs)
                if sanity_check:
                    assert torch.all(keys.eq(attributes[:,0]))
                _, counts = torch.unique_consecutive(referents[:,0], dim=0, return_counts=True)
                attributes = attributes.repeat_interleave(counts, dim=0)
            # Draw negative samples
            (negatives, _, _) = \
                self.build_negative_samples(
                    referents=referents,
                    positives=positives,
                )
            if batch_offsets is not None:
                # Batch offset samples.
                referents = offset_sample(referents, batch_offsets[ib])
                positives = offset_sample(positives, batch_offsets[ib])
                negatives = offset_sample(negatives, batch_offsets[ib])
            batch_counts.append(referents.shape[0])
            batch_referents.append(referents)
            batch_positives.append(positives)
            batch_negatives.append(negatives)
            if batch_attributes is not None:
                batch_attributes.append(attributes)

        # Make sample tensors
        batch_counts = torch.tensor(batch_counts, device=batch_referents[0].device)
        batch_referents = torch.cat(batch_referents, dim=0)
        batch_positives = torch.cat(batch_positives, dim=0)
        batch_negatives = torch.cat(batch_negatives, dim=0)

        # Transpose to column view of positive entity attributes
        if batch_attributes is not None:
            batch_attributes = torch.cat(batch_attributes, dim=0)
        return {"batch_counts": batch_counts,
                "referents": batch_referents,
                "ref_attribs": batch_attributes,
                "positive_samples": batch_positives,
                "negative_samples": batch_negatives}

    def get_key_attributes(self, keys, attributes):
        '''
            keys: indices.
            attrbitues: first column of last dimension is the key (i.e. index).
        '''
        delta = keys[:,None]-attributes[:,0][None,:]
        mask = delta == 0 # Mask of matched index
        # Expand to match mask dimensionality.
        a = attributes[None,:].repeat_interleave(len(keys), dim=0)
        a = a[mask]
        _, counts = torch.unique_consecutive(a, dim=0, return_counts=True)
        # Turn the count list into accumulative count list.
        index = torch.cat((torch.zeros(1, dtype=counts.dtype, device=counts.device),
                            torch.cumsum(counts, dim=0)))
        index = index[:-1][:,None]
        a = torch.take_along_dim(a, indices=index, dim=0)
        return a


class CorefNegativeSampler(NegativeSampler):
    def __call__(
        self,
        batch_triples,
        batch_offsets=None,
        batch_contexts=None,
        batch_attribs=None,
        **kwargs
    ):
        '''
            batch_triples will be offsetted in place if batch_offsets is not None
        '''
        batch_referents = []
        batch_positive_samples = []
        batch_negative_samples = []
        batch_positive_relations = []
        batch_negative_relations = []
        batch_referent_attribs = [] if batch_attribs is not None else None
        batch_positive_attribs = [] if batch_attribs is not None else None
        batch_negative_attribs = [] if batch_attribs is not None else None
        batch_counts = []
        for ib, triples in enumerate(batch_triples):
            positive_samples = triples
            positive_sample_attribs = batch_attribs[ib] if batch_attribs is not None else None

            referents = positive_samples[:,0]
            positives = positive_samples[:,-1]
            relations = positive_samples[:,1]
            referent_attribs, reference_attribs = None, None
            if positive_sample_attribs is not None:
                referent_attribs, reference_attribs = positive_sample_attribs
            (negatives, negative_attribs, negative_relations) = \
                self.build_negative_samples(
                    referents=referents,
                    positives=positives,
                    relations=relations,
                    attributes=reference_attribs
                )

            # Offset samples
            if batch_offsets is not None:
                # Batch offset positive samples.
                referents = offset_sample(referents, batch_offsets[ib])
                positives = offset_sample(positives, batch_offsets[ib])
                # Batch offset negative samples.
                negatives = offset_sample(negatives, batch_offsets[ib])
            batch_counts.append(referents.shape[0])
            batch_referents.append(referents)
            batch_positive_samples.append(positives)
            batch_negative_samples.append(negatives)
            batch_positive_relations.append(relations)
            batch_negative_relations.append(negative_relations)
            # Offset indexed attributes
            if batch_offsets is not None:
                # Batch offset indexed attributes of referent samples.
                if referent_attribs is not None:
                    referent_attribs = offset_sample(
                                        referent_attribs,
                                        batch_offsets[ib],
                                        locs=[0]) # index column
                # Batch offset indexed attributes of positive samples.
                if reference_attribs is not None:
                    reference_attribs = offset_sample(
                                        reference_attribs,
                                        batch_offsets[ib],
                                        locs=[0]) # index column
                # Batch offset indexed attributes of negative samples.
                if negative_attribs is not None:
                    negative_attribs = offset_sample(
                                        negative_attribs,
                                        batch_offsets[ib],
                                        locs=[0])
            if batch_referent_attribs is not None:
                batch_referent_attribs.append(referent_attribs)
            if batch_positive_attribs is not None:
                batch_positive_attribs.append(reference_attribs)
            if batch_negative_attribs is not None:
                batch_negative_attribs.append(negative_attribs)

        # Make sample tensors
        batch_counts = torch.tensor(batch_counts, device=batch_referents[0].device)
        batch_referents = torch.cat(batch_referents, dim=0)
        batch_positive_samples = torch.cat(batch_positive_samples, dim=0)
        batch_negative_samples = torch.cat(batch_negative_samples, dim=0)
        batch_positive_relations = torch.cat(batch_positive_relations, dim=0)
        batch_negative_relations = torch.cat(batch_negative_relations, dim=0)
        if batch_referent_attribs is not None:
            batch_referent_attribs = torch.cat(batch_referent_attribs, dim=0)
        if batch_positive_attribs is not None:
            batch_positive_attribs = torch.cat(batch_positive_attribs, dim=0)
        if batch_negative_attribs is not None:
            batch_negative_attribs = torch.cat(batch_negative_attribs, dim=0)

        if sanity_check:
            # Sanity check referent index samples.
            ok = torch.all(batch_referents.eq(batch_referent_attribs[:,0])).item()
            assert ok
            # Sanity check reference index match of positive samples.
            ok = torch.all(batch_positive_samples.eq(batch_positive_attribs[:,0])).item()
            assert ok
            # Sanity check reference index match of negative samples.
            ok = torch.all(batch_negative_samples.eq(batch_negative_attribs[:,0])).item()
            assert ok

        return {"batch_counts": batch_counts,
                "referents": batch_referents,
                "positive_samples": batch_positive_samples,
                "negative_samples": batch_negative_samples,
                "referent_attribs": batch_referent_attribs,
                "positive_attribs": batch_positive_attribs,
                "negative_attribs": batch_negative_attribs,
                "positve_relations": batch_positive_relations,
                "negative_relations": batch_negative_relations}
