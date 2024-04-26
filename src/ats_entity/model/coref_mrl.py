from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from model.er_modeling.representation_models import (
    IndexedStateRepresentationModel,
    AttributedStateRepresentationModel,
    EmbeddingRepresentationModel,
    PairedRelationRepresentationModel
)
from model.datautil import (
    get_unique_index,
    covert_word_to_token_indices
)
from model.coref_data_utils import (
    CorefGraphBuilder,
    EntityGraphBuilder,
    offset_key_to_head,
    get_indexed_attributes,
    uniquify_attributes,
    query_indexed_attributes
)
from model.coref_ctxt_repr import (
    ContextRepresentationModel,
    MentionContextRepresentationModel
)
from model.coref_cls import (EntityWeightModel, DistWeightModel)
from model.co_embedding import CoEmbedding
from model.coref_entity_gnn import EntityAggrNet
from model.coref_mrl_modules import (CorefCtxtMRL, CorefMRL)


class CorefNet(nn.Module):
    def __init__(
        self,
        config,
        attr_map,
        type_vocab,
        animacy_vocab,
        number_vocab,
        gender_vocab,
        embeddings,
        logger,
        data_monitor=None,
        **kwargs
    ):
        super().__init__()
        self.coref_entity_builder = EntityGraphBuilder(config.ngram_bound) if config.aggregate_entity else None
        self.coref_entity_net = EntityAggrNet(config.entity_aggr) if config.aggregate_entity else None
        self.coref_graph_builder = CorefGraphBuilder(config.ngram_bound)
        mwc_repr = None
        if config.share_weight_ctxt_repr:
            mwc_repr = self._prepare_margin_weight_ctxt_repr_model(
                            config,
                            embeddings,
                            logger,
                            **kwargs
                        )
        (er_m, rr_m, ar_m, w_m) = \
            self._prepare_coref_mrl_repr_models(
                                    config=config,
                                    attr_map=attr_map,
                                    type_vocab=type_vocab,
                                    animacy_vocab=animacy_vocab,
                                    number_vocab=number_vocab,
                                    gender_vocab=gender_vocab,
                                    embeddings=embeddings,
                                    mwc_repr=mwc_repr,
                                    logger=logger,
                                    **kwargs
                                )

        self.ref_mrl = None
        if config.coref_mrl.on:
            mrl_kwargs = {"attributed": config.coref_mrl.relational=="attributed",
                        "score_function": config.coref_mrl.score_function}
            self.ref_mrl = CorefMRL(er_model=er_m,
                                    rr_model=rr_m,
                                    cr_model=er_m,
                                    ar_model=ar_m,
                                    w_model=w_m,
                                    loss_margin=config.coref_mrl.loss_margin,
                                    lambda_w=config.coref_mrl.lambda_w,
                                    ffn_config=config.mlp,
                                    logger=logger,
                                    data_monitor=data_monitor,
                                    **mrl_kwargs
                            )

        self.ctxt_mrl = None
        if config.ctxt_mrl.on:
            (er_m, cr_m, w_m) = self._prepare_ctxt_mrl_repr_models(
                                    config=config,
                                    embeddings=embeddings,
                                    mwc_repr=mwc_repr,
                                    logger=logger,
                                    **kwargs
                                )
            mrl_kwargs = {"score_function": config.ctxt_mrl.score_function}
            self.ctxt_mrl = CorefCtxtMRL(er_model=er_m,
                                        rr_model=None,
                                        cr_model=cr_m,
                                        ar_model=None,
                                        w_model=w_m,
                                        loss_margin=config.ctxt_mrl.loss_margin,
                                        lambda_w=config.ctxt_mrl.lambda_w,
                                        logger=logger,
                                        data_monitor=data_monitor,
                                        **mrl_kwargs
                            )

        self.aggregate_entity = config.aggregate_entity
        self.attributed = config.attributed
        self.contextual_weight = config.coref_mrl.contextual_weight
        self.mention_keyed = config.mention_keyed
        self.attribute_order = config.attribute_order

    def forward(self, inputs):
        '''
            Note:
                Negative sampler returns batch flattened samples
                if batch_offsets is not None.
                It makes computational efficiency of scoring.
        '''
        (h_x, struct_inputs) = inputs
        data = self.prepare_data(h_x, struct_inputs)
        h_x = data.pop("h_x")
        nb, nl, _ = h_x.size()
        batch_offsets = [nl*b for b in range(nb)]
        batch_triples, batch_contexts, batch_attribs, batch_entity_spans = self.get_triples(data)

        loss = 0.0
        if self.ref_mrl is not None:
            kwargs = {"h_x": h_x,
                    "triples": batch_triples,
                    "offsets": batch_offsets,
                    "contexts": batch_contexts,
                    "attributes": batch_attribs,
                    "entity_spans": batch_entity_spans}
            loss, _ = self.ref_mrl.forward(kwargs)

        if self.ctxt_mrl is not None:
            kwargs = {"h_x": h_x,
                    "triples": batch_triples,
                    "offsets": batch_offsets,
                    "contexts": batch_contexts,
                    "attributes": None,
                    "entity_spans": batch_entity_spans}
            ctxt_loss, _ = self.ctxt_mrl.forward(kwargs)
            loss = loss + ctxt_loss
        return (loss, h_x)

    def get_triples(self, inputs):
        batch_triples = []
        # Entity attributes
        batch_entity_attribs = [] if self.attributed else None
        # Contexts
        use_context = self.ctxt_mrl is not None or self.contextual_weight
        batch_contexts = inputs.get("context", None) if use_context else None
        batch_entity_spans = inputs.get("entity_spans", None) if batch_contexts is not None else None
        for ib, mentions in enumerate(inputs["mentions"]):
            # Sort by antecedent index first
            mentions = sorted(mentions, key=lambda x: x[0].item())
            triples = []
            mantec_attribs = [] if self.attributed else None
            mcoref_attribs = [] if self.attributed else None
            for (antecedent, corefs) in mentions:
                # Remove index duplicated corefs.
                values, indices = torch.sort(corefs[:,None], dim=0, descending=False, stable=True)
                uniq_loc, _ = get_unique_index(values, indices)
                corefs = corefs[uniq_loc]

                relations = torch.abs(corefs - antecedent) # Placeholder.
                n_coref = corefs.shape[0]
                antecedent = antecedent.expand(n_coref)
                triple = (antecedent, relations, corefs) # Pair up.
                triples.append(triple)
                # Gather sparse coref_type, coref_animacy, coref_gender, coref_number by mentions
                if self.attributed:
                    entity_attrbitues = inputs["attrbitues"]
                    batch_counts = inputs["batch_counts"]
                    antecedent_attribs = query_indexed_attributes(antecedent, entity_attrbitues[batch_counts==ib])
                    antecedent_attribs = antecedent_attribs.expand(n_coref, -1)
                    corefs_attribs = query_indexed_attributes(corefs, entity_attrbitues[batch_counts==ib])
                    mantec_attribs.append(antecedent_attribs)
                    mcoref_attribs.append(corefs_attribs)
            triples = torch.cat([torch.stack(t, dim=0).T for t in triples], dim=0)
            # Add to batch
            batch_triples.append(triples)
            if self.attributed:
                mantec_attribs = torch.cat(mantec_attribs, dim=0)
                mcoref_attribs = torch.cat(mcoref_attribs, dim=0)
                batch_entity_attribs.append((mantec_attribs, mcoref_attribs))
        return (batch_triples, batch_contexts, batch_entity_attribs, batch_entity_spans)

    def _prepare_margin_weight_ctxt_repr_model(
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

    def _prepare_coref_mrl_repr_models(
        self,
        config,
        attr_map,
        type_vocab,
        animacy_vocab,
        number_vocab,
        gender_vocab,
        embeddings,
        mwc_repr,
        logger,
        **kwargs
    ):
        if config.attributed and config.coref_mrl.relational != "attributed":
            assert config.entity_repr.in_channels == \
                config.mlp.out_channels + attr_map.dim*len(config.attribute_order)
            er_m = AttributedStateRepresentationModel(
                        in_channels=config.entity_repr.in_channels,
                        out_channels=config.entity_repr.out_channels,
                        activation=config.entity_repr.activation,
                        batch_norm=config.entity_repr.batch_norm,
                        dropout=config.entity_repr.dropout)
        else:
            er_m = IndexedStateRepresentationModel()

        rr_m = None
        if config.coref_mrl.relational == "attributed":
            rr_m = PairedRelationRepresentationModel(
                        in_channels=config.attributed_rel.in_channels,
                        out_channels=config.attributed_rel.out_channels,
                        activation=config.attributed_rel.activation,
                        batch_norm=config.attributed_rel.batch_norm,
                        dropout=config.attributed_rel.dropout)
        elif config.coref_mrl.relational == "distance":
            rr_m = EmbeddingRepresentationModel(
                        n_embeddings=config.distance_rel.n_embeddings,
                        n_dims=config.distance_rel.n_dims,
                        padding_idx=config.distance_rel.padding_idx
                    )

        ar_m = CoEmbedding(
                    attr_struct=attr_map,
                    type_vocab=type_vocab,
                    animacy_vocab=animacy_vocab,
                    number_vocab=number_vocab,
                    gender_vocab=gender_vocab,
                    key_order=config.attribute_order,
                    logger=logger
                ) if config.attributed else None

        w_m = None
        if config.coref_mrl.contextual_weight:
            if config.coref_mrl.weight_func == "dspan": # Distance span based
                w_m = DistWeightModel(config,
                                      config.coref_mrl.weight_linear_in,
                                      squash_weighting=config.coref_mrl.squash_weighting,
                                      embeddings=embeddings,
                                      mwc_repr=mwc_repr,
                                      logger=logger,
                                      **kwargs)
            else: # "espan"
                w_m = EntityWeightModel(config,
                                      config.coref_mrl.weight_linear_in,
                                      squash_weighting=config.coref_mrl.squash_weighting,
                                      embeddings=embeddings,
                                      mwc_repr=mwc_repr,
                                      logger=logger,
                                      **kwargs)
        return (er_m, rr_m, ar_m, w_m)

    def _prepare_ctxt_mrl_repr_models(
        self,
        config,
        embeddings,
        mwc_repr,
        logger,
        **kwargs
    ):
        er_m = IndexedStateRepresentationModel()

        cr_m = ContextRepresentationModel(
                    config,
                    embeddings,
                    logger,
                    **kwargs
                )

        w_m = None
        if config.ctxt_mrl.contextual_weight:
            if config.ctxt_mrl.weight_func == "dspan": # Distance span based
                w_m = DistWeightModel(config,
                                      config.ctxt_mrl.weight_linear_in,
                                      squash_weighting=config.ctxt_mrl.squash_weighting,
                                      embeddings=embeddings,
                                      mwc_repr=mwc_repr,
                                      logger=logger,
                                      **kwargs)
            else: # "espan"
                w_m = EntityWeightModel(config,
                                      config.ctxt_mrl.weight_linear_in,
                                      squash_weighting=config.ctxt_mrl.squash_weighting,
                                      embeddings=embeddings,
                                      mwc_repr=mwc_repr,
                                      logger=logger,
                                      **kwargs)

        return (er_m, cr_m, w_m)

    def prepare_data(self, h_x, struct_inputs):
        # Get an entity key/head word index offset w.r.t. the first word of the entity.
        keyhead_offset, keyhead_offset_mask = None, None
        if self.mention_keyed:
            keyhead_offset, keyhead_offset_mask = \
                offset_key_to_head(
                    coref_index=struct_inputs["coref_index"],
                    coref_index_mask=struct_inputs["coref_index_mask"],
                    coref_entity_span=struct_inputs["coref_entity_span"],
                    coref_entity_span_mask=struct_inputs["coref_entity_span_mask"],
                    coref_key_index=struct_inputs["coref_head_index"],
                    coref_key_index_mask=struct_inputs["coref_head_index_mask"],
                )

        # Map word level indices onto token level indices
        ignore_value=-100
        entity_token_indices, entity_token_mask = \
            covert_word_to_token_indices(token_mask=struct_inputs["token_mask"],
                                        token_mask_mask=struct_inputs["token_mask_mask"],
                                        indices=struct_inputs["coref_index"],
                                        indices_mask=struct_inputs["coref_entity_mask"],
                                        indices_mask_mask=struct_inputs["coref_entity_mask_mask"],
                                        token_span_count=struct_inputs["subword_span"],
                                        token_span_count_mask=struct_inputs["subword_span_mask"],
                                        indices_offset_n=1, # offset from BOS
                                        flat_batch=False,
                                        ignore_value=ignore_value)
        # Build edges for aggregating multiple word entities by GNN.
        if self.aggregate_entity:
            _, nl = entity_token_indices.size()
            assert nl == struct_inputs["coref_index"].shape[-1], \
                    "covert_word_to_token_indices has changed length dimension."
            entity_edges, _, (_, _, _, _, _, batch_counts) = \
                self.coref_entity_builder(
                    coref_index=entity_token_indices, #struct_inputs["coref_index"],
                    coref_index_mask=struct_inputs["coref_index_mask"],
                    coref_entity_mask=struct_inputs["coref_entity_mask"],
                    coref_entity_mask_mask=entity_token_mask, #struct_inputs["coref_entity_mask_mask"],
                    coref_entity_span=struct_inputs["coref_entity_span"],
                    coref_entity_span_mask=struct_inputs["coref_entity_span_mask"],
                    coref_key_offset=keyhead_offset,
                    coref_key_offset_mask=keyhead_offset_mask,
                    batch_offset_required=True,
                    lone_entity_included=False,
                    entity_head_self_loop=False,
                    dep_to_head_direction=True)
  
            # Aggregate multiple word entity to a single word representation by GNN.
            entity_distances = torch.abs(entity_edges[0] - entity_edges[1])
            h_x = self.coref_entity_net({"data": h_x,
                                        "edge": entity_edges,
                                        "edge_feature": entity_distances,
                                        })

        # Bundle entity indices and their attributes.
        attrbitues = None
        if self.attributed:
            attributes = [struct_inputs["coref_" + k] for k in self.attribute_order]
            # Create an indexed attributes as a map (sliceable tensor by matching index).
            indexed_attrbitues = \
                get_indexed_attributes(
                        attributes=attributes,
                        entity_head=None,
                        entity_index=entity_token_indices,
                        entity_mask=struct_inputs["coref_entity_mask"],
                        entity_mask_mask=struct_inputs["coref_entity_mask_mask"],
                        entity_span=struct_inputs["coref_entity_span"],
                        entity_span_mask=struct_inputs["coref_entity_span_mask"],
                        key_offset=keyhead_offset,
                        key_offset_mask=keyhead_offset_mask,
                        batch_offset_required=False
                    )
            attrbitues, batch_counts = uniquify_attributes(indexed_attrbitues, batch_counts)

        # Gather antecedent-coreferent pairs.
        g_data = self.coref_graph_builder(
                    coref_index=entity_token_indices,
                    coref_sent_num=struct_inputs["coref_sent_num"],
                    coref_repr_mask=struct_inputs["coref_repr_mask"],
                    coref_repr_mask_mask=struct_inputs["coref_repr_mask_mask"],
                    coref_entity_mask=struct_inputs["coref_entity_mask"],
                    coref_entity_span=struct_inputs["coref_entity_span"],
                    coref_sent_num_mask=struct_inputs["coref_sent_num_mask"],
                    token_sentence_sizes=struct_inputs["tokenized_sent_sizes"],
                    token_mask=struct_inputs["token_mask"],
                    token_mask_mask=struct_inputs["token_mask_mask"],
                    coref_key_offset=keyhead_offset,
                    coref_key_offset_mask=keyhead_offset_mask,
                    include_ante=True,
                    edge_builder=None,
                    return_tensor=False
                )
        return {"h_x": h_x,
                "attrbitues": attrbitues,
                "batch_counts": batch_counts,
                "context": g_data["context"],
                "mentions": g_data["mention"],
                "entity_spans": g_data["entity_span"]}
