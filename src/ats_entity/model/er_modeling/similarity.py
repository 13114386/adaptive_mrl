from __future__ import unicode_literals, print_function, division
"""
    Tailored ComplEx model based on pykeen's implementation.
    (Refer to doc site https://pykeen.readthedocs.io/)
"""
from model.er_modeling.er_model import ERModel
from model.er_modeling.interactions import SimilarityInteraction, Interaction

class SimilarityER(ERModel):
    interaction: Interaction

    def __init__(
        self,
        *,
        referent_repr_model,
        relation_repr_model,
        reference_repr_model,
        attrib_repr_model,
        **kwargs,
    ) -> None:
        super().__init__(
            interaction=SimilarityInteraction,
            referent_repr_model=referent_repr_model,
            relation_repr_model=relation_repr_model,
            reference_repr_model=reference_repr_model,
            attrib_repr_model=attrib_repr_model,
            **kwargs,
        )
