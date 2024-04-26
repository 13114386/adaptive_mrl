from __future__ import unicode_literals, print_function, division
"""
    Tailored ERModel model based on pykeen's implementation.
    (Refer to doc site https://pykeen.readthedocs.io/)
"""
from typing import Generic, Optional, Tuple, Dict, List, cast
import torch
from class_resolver import HintOrType, OptionalKwargs
from torch import nn
# from pykeen.nn.modules import Interaction, interaction_resolver
from model.er_modeling.interactions import (
    Interaction,
    interaction_resolver,
    AttributeRepresentation,
)
from pykeen.typing import (
    HeadRepresentation,
    InductiveMode,
    RelationRepresentation,
    TailRepresentation,
)


class ERModel(
    Generic[HeadRepresentation,
            RelationRepresentation,
            TailRepresentation,
            AttributeRepresentation],
    nn.Module,
):
    interaction: Interaction

    def __init__(
        self,
        *,
        referent_repr_model,
        relation_repr_model,
        reference_repr_model,
        attrib_repr_model,
        interaction: HintOrType[Interaction[HeadRepresentation,
                                            RelationRepresentation,
                                            TailRepresentation]],
        interaction_kwargs: OptionalKwargs = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.interaction = interaction_resolver.make(interaction, pos_kwargs=interaction_kwargs)
        self.referent_repr_model = referent_repr_model
        self.relation_repr_model = relation_repr_model
        self.reference_repr_model = reference_repr_model
        self.attrib_repr_model = attrib_repr_model
        self.attributed_relation = kwargs.pop("attributed", False)

    def forward(
        self,
        latent_states: torch.Tensor,
        hs: torch.LongTensor,
        rs: torch.LongTensor,
        ts: torch.LongTensor,
        slice_size: Optional[int] = None,
        slice_dim: int = 0,
        *,
        attribs: Optional[torch.LongTensor] = None,
        mode: Optional[InductiveMode],
    ) -> torch.FloatTensor:
        hr, rr, tr = self._get_representations(
                                latent_states=latent_states,
                                h=hs,
                                r=rs,
                                t=ts,
                                attribs=attribs,
                                mode=mode
                            )
        return self.interaction.score(h=hr, r=rr, t=tr,
                                    slice_size=slice_size,
                                    slice_dim=slice_dim)

    def score_hrt(
        self,
        latent_states: torch.Tensor,
        hs: torch.LongTensor,
        rs: torch.LongTensor,
        ts: torch.LongTensor,
        attribs: Optional[torch.LongTensor] = None,
        mode: Optional[InductiveMode] = None
    ) -> torch.FloatTensor:
        hr, rr, tr = self._get_representations(
                                    latent_states,
                                    h=hs,
                                    r=rs,
                                    t=ts,
                                    attribs=attribs,
                                    mode=mode
                                )
        return self.interaction.score_hrt(h=hr, r=rr, t=tr)

    def _get_representations(
        self,
        latent_states: torch.FloatTensor,
        h: Optional[torch.LongTensor],
        r: Optional[torch.LongTensor],
        t: Optional[torch.LongTensor],
        attribs: Optional[torch.LongTensor] = None,
        mode: Optional[InductiveMode] = None,
    ) -> Tuple[HeadRepresentation,
               RelationRepresentation,
               TailRepresentation,
               AttributeRepresentation]:
        """Get representations for head, relation and tails."""
        has_attrib = self.attrib_repr_model is not None and \
                        attribs is not None
        h_a, t_a = attribs if has_attrib else (None, None)
        h_ar = self.attrib_repr_model(h_a[:,1:]) if h_a is not None else None
        t_ar = self.attrib_repr_model(t_a[:,1:]) if t_a is not None else None

        kwargs = {"ar": h_ar} if h_ar is not None and not self.attributed_relation else {}
        hr = self.referent_repr_model(h, initial_states=latent_states, **kwargs)
        hr = hr.contiguous()

        kwargs = {"ar": t_ar} if t_ar is not None and not self.attributed_relation else {}
        tr = self.reference_repr_model(t, initial_states=latent_states, **kwargs)
        tr = tr.contiguous()

        rr = None
        if self.relation_repr_model is not None:
            if self.attributed_relation:
                kwargs = {"ar": (h_ar, t_ar)}
                rr = self.relation_repr_model(indices=None, **kwargs).contiguous()
            elif r is not None:
                rr = self.relation_repr_model(r).contiguous()

        return cast(
            Tuple[HeadRepresentation, RelationRepresentation, TailRepresentation],
            tuple(x[0] if x is not None and len(x) == 1 else x for x in (hr, rr, tr)),
        )
