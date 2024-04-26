from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from model.coref_ctxt_repr import MentionContextRepresentationModel
from model.mlp import MLP
from model.cost import span_iou

class WeightModelBase(nn.Module):
    def __init__(
        self,
        config,
        linear_in,
        squash_weighting,
        embeddings,
        mwc_repr,
        logger,
        **kwargs
    ):
        super().__init__()
        if mwc_repr is not None:
            self.cr_m = mwc_repr
        else:
            self.cr_m = self._prepare_representation_models(
                            config,
                            embeddings,
                            logger,
                            **kwargs
                        )
        if config.ctxt_repr.type == "conv":
            self.mlp = MLP([config.ctxt_repr.conv.in_channels[0],
                            config.ctxt_repr.conv.feature_dims],
                            activation=nn.ReLU())
        elif config.ctxt_repr.type == "bert":
            self.mlp = MLP([config.ctxt_repr.bert.output_dims,
                            config.ctxt_repr.bert.feature_dims],
                            activation=nn.ReLU())
        self.span_head = nn.Linear(linear_in,
                                2*config.ctxt_repr.num_labels,
                                bias=False)
        self.max_length = config.ctxt_repr.max_length
        self.num_labels = config.ctxt_repr.num_labels
        self.squash_weighting = squash_weighting

    def _prepare_representation_models(
        self,
        config,
        embeddings,
        logger,
        **kwargs
    ):
        cr_m = MentionContextRepresentationModel(
                    config=config.weight_func,
                    embeddings=embeddings,
                    logger=logger,
                    **kwargs
                )
        return cr_m

    def _offset_context(
        self,
        batch_offsets,
        batch_contexts,
    ):
        # Offset context
        offset_contexts = {}
        for offsets, contexts in zip(batch_offsets, batch_contexts):
            for key, values in contexts.items():
                offset_contexts[key+offsets] = \
                    [(v[0]+offsets, v[1]+offsets) for v in values]
        return offset_contexts

    def compute_weights(self, targets, logits, ordered):
        if isinstance(targets, (tuple, list)):
            spans = torch.cat(targets, dim=-1)
        else:
            spans = targets

        total_loss = None
        ignored_index = self.max_length
        if self.num_labels == 1:
            clamp_logits = logits.clamp(0, ignored_index)
            if not ordered:
                # Ensure small index first.
                spans, order = torch.sort(spans, dim=-1, descending=False, stable=True)
                clamp_logits = torch.gather(clamp_logits, 1, order)
            iou = span_iou(spans, clamp_logits, DIoU=True)
            w = (3.-iou) # 2+(1-iou)=>2+[0,2]=>[2,4]
            if self.squash_weighting:
                w = w/2. # squash into [1,2]
            else:
                w = w-1. # [2,4]=>[1,3]
            return w
        else:
            start_logits, end_logits = logits.split(self.num_labels, dim=-1)
            start_logits = start_logits.squeeze(-1).contiguous()
            end_logits = end_logits.squeeze(-1).contiguous()

            start_positions = spans[:,0]
            end_positions = spans[:,1]

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            # start_positions = start_positions.clamp(0, ignored_index)
            start_loss = loss_fct(start_logits, start_positions)

            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # end_positions = end_positions.clamp(0, ignored_index)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

            w = (1.0 + total_loss / float(ignored_index))
            return w


class DistWeightModel(WeightModelBase):
    '''
        Margin weighting model based on distance for entity-entity relation.
    '''
    def forward(self, inputs):
        h_x = inputs["h_x"]
        referents = inputs["referents"]
        references = inputs["references"]
        batch_offsets = inputs["offsets"]
        batch_contexts = inputs["contexts"]
        offset_context = inputs["offset_context"]
        batch_counts = inputs["batch_counts"]
        if offset_context:
            contexts = self._offset_context(batch_offsets, batch_contexts)
        else:
            contexts = batch_contexts

        kwargs = {"contexts": contexts}
        x = self.cr_m(indices=referents, initial_states=h_x, **kwargs)
        h_x_flat = h_x.reshape(-1, h_x.shape[-1])
        h_referents = h_x_flat[referents]
        h_references = h_x_flat[references]
        if self.mlp is not None: # Reduce dimensionality.
            x = self.mlp(x)
            h_referents = self.mlp(h_referents)
            h_references = self.mlp(h_references)
        x = torch.cat([x,h_referents,h_references], dim=-1)
        logits = self.span_head(x)
        # Remove offset from index for span head comparison with logits.
        offsets = torch.tensor(batch_offsets, device=referents.device)
        offsets = offsets.repeat_interleave(batch_counts, dim=0)
        distance_spans = torch.stack((referents-offsets, references-offsets), dim=-1)
        w = self.compute_weights(distance_spans, logits, ordered=False)
        return w


class EntityWeightModel(WeightModelBase):
    def forward(self, inputs):
        h_x = inputs["h_x"]
        referents = inputs["referents"]
        references = inputs["references"]
        batch_offsets = inputs["offsets"]
        batch_contexts = inputs["contexts"]
        batch_entity_spans = inputs["entity_spans"]
        offset_context = inputs["offset_context"]
        offset_entity_span = inputs["offset_entity_span"]
        if offset_context:
            contexts = self._offset_context(batch_offsets, batch_contexts)
        else:
            contexts = batch_contexts
        if offset_entity_span:
            entity_spans = self._offset_entity_span(batch_offsets, batch_offsets, batch_entity_spans)
        elif offset_context:
            offsets = [0]*len(batch_entity_spans)
            entity_spans = self._offset_entity_span(batch_offsets, offsets, batch_entity_spans)
        kwargs = {"contexts": contexts}
        x = self.cr_m(indices=referents, initial_states=h_x, **kwargs)
        h_x_flat = h_x.reshape(-1, h_x.shape[-1])
        h_referents = h_x_flat[referents]
        h_references = None
        if references is not None:
            h_references = h_x_flat[references]
        if self.mlp is not None: # Reduce dimensionality.
            x = self.mlp(x)
            h_referents = self.mlp(h_referents)
            if h_references is not None:
                h_references = self.mlp(h_references)
        if h_references is not None:
            x = torch.cat([x,h_referents,h_references], dim=-1)
        else:
            x = torch.cat([x,h_referents], dim=-1)
        logits = self.span_head(x)

        entity_spans = self.get_entity_span(
                            referents=referents,
                            entity_spans=entity_spans
                        )
        # entity_spans indices are not batch offset.
        # Or, must roll back just as DistWeightModel does.
        entity_spans = torch.tensor(entity_spans, device=logits.device)
        w = self.compute_weights(entity_spans, logits, ordered=True)
        return w

    def get_entity_span(self, referents, entity_spans):
        '''
            batch_triples will be offsetted in place if batch_offsets is not None
        '''
        ref_entity_spans = []
        for referent in referents.tolist():
            entity_span = entity_spans[referent]
            # Only need the bound indices.
            entity_span = [entity_span[0], entity_span[-1]]
            ref_entity_spans.append(entity_span)
        return ref_entity_spans

    def _offset_entity_span(
        self,
        referent_offsets,
        reference_offsets,
        batch_entity_spans,
    ):
        # Offset context
        offset_spans = {}
        for rr_offset, rc_offset, spans in \
            zip(referent_offsets, reference_offsets, batch_entity_spans):
            for key, values in spans.items():
                offset_spans[key+rr_offset] = [v+rc_offset for v in values]
        return offset_spans
