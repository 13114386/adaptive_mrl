from __future__ import unicode_literals, print_function, division
"""
    Tailored ComplEx model based on pykeen's implementation.
    (Refer to doc site https://pykeen.readthedocs.io/)
"""
from model.er_modeling.er_model import ERModel
# from pykeen.nn.modules import ComplExInteraction, Interaction
from model.er_modeling.interactions import ComplExInteraction, Interaction

class ComplEx(ERModel):
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
            interaction=ComplExInteraction,
            referent_repr_model=referent_repr_model,
            relation_repr_model=relation_repr_model,
            reference_repr_model=reference_repr_model,
            attrib_repr_model=attrib_repr_model,
            **kwargs,
        )

