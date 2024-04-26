from __future__ import unicode_literals, print_function, division
"""
    Tailored complex interaction based on pykeen's implementation.
    (Refer to doc site https://pykeen.readthedocs.io/)
"""

import logging
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    MutableMapping,
    Optional,
    TypeVar
)

import torch
from class_resolver import ClassResolver
from torch import FloatTensor, nn
from pykeen.typing import (
    HeadRepresentation,
    RelationRepresentation,
    Representation,
    TailRepresentation,
    OneOrSequence,
)
from pykeen.nn.modules import parallel_slice_batches
from model.er_modeling import functional as pkf

AttributeRepresentation = TypeVar("AttributeRepresentation", bound=OneOrSequence[torch.FloatTensor])

logger = logging.getLogger(__name__)


class Interaction(
    nn.Module,
    Generic[HeadRepresentation,
            RelationRepresentation,
            TailRepresentation],
    ABC
):
    """Base class for interaction functions."""

    @abstractmethod
    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        """Compute broadcasted triple scores given broadcasted representations for head, relation and tails.

        :param h: shape: (`*batch_dims`, `*dims`)
            The head representations.
        :param r: shape: (`*batch_dims`, `*dims`)
            The relation representations.
        :param t: shape: (`*batch_dims`, `*dims`)
            The tail representations.

        :return: shape: batch_dims
            The scores.
        """

    def score(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
        slice_size: Optional[int] = None,
        slice_dim: int = 1,
    ) -> torch.FloatTensor:
        if slice_size is None:
            return self(h=h, r=r, t=t)

        return torch.cat(
            [
                self(h=h_batch, r=r_batch, t=t_batch)
                for h_batch, r_batch, t_batch in \
                    parallel_slice_batches(h, r, t, split_size=slice_size, dim=slice_dim)
            ],
            dim=slice_dim,
        )

    def score_hrt(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        return self.score(h=h, r=r, t=t).unsqueeze(dim=-1)


class FunctionalInteraction(
    Interaction,
    Generic[HeadRepresentation,
            RelationRepresentation,
            TailRepresentation]
):
    """Base class for interaction functions."""

    #: The functional interaction form
    func: Callable[..., torch.FloatTensor]

    def forward(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> torch.FloatTensor:
        return self.__class__.func(**self._prepare_for_functional(h=h, r=r, t=t))

    def _prepare_for_functional(
        self,
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> Mapping[str, torch.FloatTensor]:
        """Conversion utility to prepare the arguments for the functional form."""
        kwargs = self._prepare_hrt_for_functional(h=h, r=r, t=t)
        kwargs.update(self._prepare_state_for_functional())
        return kwargs

    # docstr-coverage: inherited
    @staticmethod
    def _prepare_hrt_for_functional(
        h: HeadRepresentation,
        r: RelationRepresentation,
        t: TailRepresentation,
    ) -> MutableMapping[str, torch.FloatTensor]:  # noqa: D102
        """Conversion utility to prepare the h/r/t representations for the functional form."""
        assert all(x is None or torch.is_tensor(x) for x in (h, r, t))
        return dict(h=h, r=r, t=t)

    def _prepare_state_for_functional(self) -> MutableMapping[str, Any]:
        """Conversion utility to prepare the state to be passed to the functional form."""
        return dict()


class ComplExInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    """A module wrapper for the stateless ComplEx interaction function.

    .. seealso:: :func:`pykeen.nn.functional.complex_interaction`
    """
    func = pkf.complex_interaction


class SimilarityInteraction(FunctionalInteraction[FloatTensor, FloatTensor, FloatTensor]):
    func = pkf.similarity_interaction



interaction_resolver: ClassResolver[Interaction] = ClassResolver.from_subclasses(
    Interaction,
)
