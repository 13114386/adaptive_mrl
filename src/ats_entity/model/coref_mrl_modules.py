from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
# from pykeen.losses import MarginRankingLoss, SoftMarginRankingLoss
from model.er_modeling.losses import MarginRankingLoss
from model.coref_mrl_sampler import (
    CorefNegativeSampler,
    CorefCtxtNegativeSampler,
)
from model.er_modeling.complex import ComplEx
from model.er_modeling.similarity import SimilarityER
from model.mlp import MLP

class CorefCtxtMRL(nn.Module):
    def __init__(
        self,
        er_model,
        rr_model,
        cr_model,
        ar_model,
        w_model,
        loss_margin,
        lambda_w,
        logger,
        data_monitor=None,
        **kwargs
    ):
        '''
            Referent-context(spans) MRL.
        '''
        super().__init__()
        self.negative_sampler = CorefCtxtNegativeSampler()
        score_function = kwargs.pop("score_function")
        if score_function == "complex":
            self.model = ComplEx(
                            referent_repr_model=er_model,
                            relation_repr_model=rr_model,
                            reference_repr_model=cr_model,
                            attrib_repr_model=ar_model,
                            **kwargs
                        )
        else:
            self.model = SimilarityER(
                            referent_repr_model=er_model,
                            relation_repr_model=rr_model,
                            reference_repr_model=cr_model,
                            attrib_repr_model=ar_model,
                            **kwargs
                        )
        self.w_model = w_model
        self.loss = MarginRankingLoss(margin=loss_margin)
        self.lambda_w = lambda_w
        self.data_monitor = data_monitor
        self.data_monitor.register(self.__class__.__name__)

    def forward(self, inputs):
        '''
            Note:
                Negative sampler returns batch flattened samples
                if batch_offsets is not None.
                It makes computational efficiency of scoring.
        '''
        h_x = inputs["h_x"]
        batch_triples = inputs["triples"]
        batch_offsets = inputs["offsets"]
        batch_contexts = inputs["contexts"]
        batch_entity_attribs = inputs["attributes"]
        mr_samples = self.negative_sampler(
                    batch_triples,
                    batch_offsets,
                    batch_contexts,
                    batch_entity_attribs
                )

        referents = mr_samples["referents"].squeeze(dim=-1)
        ws = None
        if self.w_model is not None:
            batch_contexts = inputs["contexts"]
            batch_entity_spans = inputs["entity_spans"]
            w_inputs = {"h_x": h_x,
                        "batch_counts": mr_samples["batch_counts"],
                        "referents": referents,
                        "references": None,
                        "contexts": batch_contexts,
                        "entity_spans": batch_entity_spans,
                        "offsets": batch_offsets,
                        "offset_context": True,
                        "offset_entity_span": False}
            ws = self.w_model.forward(w_inputs)
            if self.data_monitor is not None:
                min_w = torch.min(ws).item()
                max_w = torch.max(ws).item()
                self.data_monitor.update_data(owner=self.__class__.__name__,
                                              min_w=min_w, max_w=max_w)

        positive_scores = self.model.score_hrt(
                                h_x,
                                hs=referents,
                                rs=None,
                                ts=mr_samples["positive_samples"],
                                attribs=(mr_samples["ref_attribs"], None),
                                mode=None
                            )
        negative_scores = self.model.score_hrt(
                                h_x,
                                hs=referents,
                                rs=None,
                                ts=mr_samples["negative_samples"],
                                attribs=(mr_samples["ref_attribs"], None),
                                mode=None
                            )

        kwargs = {"margin_weights": ws}
        loss = self.loss.process_slcwa_scores(
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    label_smoothing=None,
                    **kwargs
                )
        loss = loss*self.lambda_w
        return (loss, h_x)


class CorefMRL(nn.Module):
    def __init__(
        self,
        er_model,
        rr_model,
        cr_model,
        ar_model,
        w_model,
        loss_margin,
        lambda_w,
        ffn_config,
        logger,
        data_monitor=None,
        **kwargs
    ):
        '''
            
            Referent-Reference MRL.
        '''
        super().__init__()
        self.negative_sampler = CorefNegativeSampler()
        self.mlp = MLP([ffn_config.in_channels, ffn_config.out_channels],
                        activation=nn.ReLU())
        score_function = kwargs.pop("score_function")
        if score_function == "complex":
            self.model = ComplEx(
                            referent_repr_model=er_model,
                            relation_repr_model=rr_model,
                            reference_repr_model=cr_model,
                            attrib_repr_model=ar_model,
                            **kwargs
                        )
        else:
            self.model = SimilarityER(
                            referent_repr_model=er_model,
                            relation_repr_model=rr_model,
                            reference_repr_model=cr_model,
                            attrib_repr_model=ar_model,
                            **kwargs
                        )
        self.w_model = w_model
        self.loss = MarginRankingLoss(margin=loss_margin)
        self.lambda_w = lambda_w
        self.data_monitor = data_monitor
        self.data_monitor.register(self.__class__.__name__)

    def forward(self, inputs):
        '''
            Note:
                Negative sampler returns batch flattened samples
                if batch_offsets is not None.
                It makes computational efficiency of scoring.
        '''
        h_x = inputs["h_x"]
        batch_triples = inputs["triples"]
        batch_offsets = inputs["offsets"]
        batch_attribs = inputs["attributes"]
        mr_samples = self.negative_sampler(
                        batch_triples,
                        batch_offsets,
                        None,
                        batch_attribs
                    )

        ws = None
        if self.w_model is not None:
            batch_contexts = inputs["contexts"]
            batch_entity_spans = inputs["entity_spans"]
            w_inputs = {"h_x": h_x,
                        "batch_counts": mr_samples["batch_counts"],
                        "referents": mr_samples["referents"],
                        "references": mr_samples["positive_samples"],
                        "contexts": batch_contexts,
                        "entity_spans": batch_entity_spans,
                        "offsets": batch_offsets,
                        "offset_context": True,
                        "offset_entity_span": False}
            ws = self.w_model.forward(w_inputs)
            if self.data_monitor is not None:
                min_w = torch.min(ws).item()
                max_w = torch.max(ws).item()
                self.data_monitor.update_data(owner=self.__class__.__name__,
                                              min_w=min_w, max_w=max_w)

        h_x = self.mlp(h_x)

        positive_scores = self.model.score_hrt(
                                h_x,
                                hs=mr_samples["referents"],
                                rs=mr_samples["positve_relations"],
                                ts=mr_samples["positive_samples"],
                                attribs=(mr_samples["referent_attribs"],
                                         mr_samples["positive_attribs"]),
                                mode=None
                            )

        negative_scores = self.model.score_hrt(
                                h_x,
                                hs=mr_samples["referents"],
                                rs=mr_samples["negative_relations"],
                                ts=mr_samples["negative_samples"],
                                attribs=(mr_samples["referent_attribs"],
                                         mr_samples["negative_attribs"]),
                                mode=None
                            )

        kwargs = {"margin_weights": ws}
        loss = self.loss.process_slcwa_scores(
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    label_smoothing=None,
                    **kwargs
                )
        loss = loss*self.lambda_w
        return (loss, h_x)
