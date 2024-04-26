from __future__ import unicode_literals, print_function, division
'''
    Fix the margin ranking loss formulation bug in the original implementation.
    (Refer to doc site https://pykeen.readthedocs.io/)
'''
from typing import Any, ClassVar, Mapping, Optional
import torch
from torch import nn
from pykeen.losses import (
    Loss,
    UnsupportedLabelSmoothingError,
    DEFAULT_MARGIN_HPO_STRATEGY
)
from class_resolver import Hint
from class_resolver.contrib.torch import margin_activation_resolver
from docdata import parse_docdata


class PairwiseLoss(Loss):
    """Pairwise loss functions compare the scores of a positive triple and a negative triple."""


class MarginPairwiseLoss(PairwiseLoss):
    r"""The generalized margin ranking loss.

    .. math ::
        L(k, \bar{k}) = g(f(\bar{k}) - f(k) + \lambda)

    Where $k$ are the positive triples, $\bar{k}$ are the negative triples, $f$ is the interaction function (e.g.,
    :class:`pykeen.models.TransE` has $f(h,r,t)=\mathbf{e}_h+\mathbf{r}_r-\mathbf{e}_t$), $g(x)$ is an activation
    function like the ReLU or softmax, and $\lambda$ is the margin.
    """

    def __init__(
        self,
        margin: float,
        margin_activation: Hint[nn.Module],
        reduction: str = "mean",
    ):
        r"""Initialize the margin loss instance.

        :param margin:
            The margin by which positive and negative scores should be apart.
        :param margin_activation:
            A margin activation. Defaults to ``'relu'``, i.e. $h(\Delta) = max(0, \Delta + \lambda)$, which is the
            default "margin loss". Using ``'softplus'`` leads to a "soft-margin" formulation as discussed in
            https://arxiv.org/abs/1703.07737.
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.
        """
        super().__init__(reduction=reduction)
        self.margin = margin
        self.margin_activation = margin_activation_resolver.make(margin_activation)

    # docstr-coverage: inherited
    def process_slcwa_scores(
        self,
        positive_scores: torch.FloatTensor,
        negative_scores: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        batch_filter: Optional[torch.BoolTensor] = None,
        num_entities: Optional[int] = None,
        **kwargs
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        # prepare for broadcasting, shape: (batch_size, 1, 3)
        positive_scores = positive_scores.unsqueeze(dim=1)

        if batch_filter is not None:
            # negative_scores have already been filtered in the sampler!
            num_neg_per_pos = batch_filter.shape[1]
            positive_scores = positive_scores.repeat(1, num_neg_per_pos, 1)[batch_filter]
            # shape: (nnz,)

        return self(pos_scores=positive_scores,
                    neg_scores=negative_scores,
                    **kwargs)

    # docstr-coverage: inherited
    def process_lcwa_scores(
        self,
        predictions: torch.FloatTensor,
        labels: torch.FloatTensor,
        label_smoothing: Optional[float] = None,
        num_entities: Optional[int] = None,
        **kwargs
    ) -> torch.FloatTensor:  # noqa: D102
        # Sanity check
        if label_smoothing:
            raise UnsupportedLabelSmoothingError(self)

        # for LCWA scores, we consider all pairs of positive and negative scores for a single batch element.
        # note: this leads to non-uniform memory requirements for different batches, depending on the total number of
        # positive entries in the labels tensor.

        # This shows how often one row has to be repeated,
        # shape: (batch_num_positives,), if row i has k positive entries, this tensor will have k entries with i
        repeat_rows = (labels == 1).nonzero(as_tuple=False)[:, 0]
        # Create boolean indices for negative labels in the repeated rows, shape: (batch_num_positives, num_entities)
        labels_negative = labels[repeat_rows] == 0
        # Repeat the predictions and filter for negative labels, shape: (batch_num_pos_neg_pairs,)
        negative_scores = predictions[repeat_rows][labels_negative]

        # This tells us how often each true label should be repeated
        repeat_true_labels = (labels[repeat_rows] == 0).nonzero(as_tuple=False)[:, 0]
        # First filter the predictions for true labels and then repeat them based on the repeat vector
        positive_scores = predictions[labels == 1][repeat_true_labels]

        return self(pos_scores=positive_scores,
                    neg_scores=negative_scores,
                    **kwargs)

    def forward(
        self,
        pos_scores: torch.FloatTensor,
        neg_scores: torch.FloatTensor,
        **kwargs
    ) -> torch.FloatTensor:
        """
        Compute the margin loss.

        The scores have to be in broadcastable shape.

        :param pos_scores:
            The positive scores.
        :param neg_scores:
            The negative scores.

        :return:
            A scalar loss term.

        Note:
            Fix the formulation bug in the original implementation.
        """
        margin_weights = kwargs.pop("margin_weights", 1.0)
        margins = margin_weights*self.margin
        if isinstance(margins, torch.Tensor):
            if len(margins.shape) < len(neg_scores.shape):
                margins = margins[:,None]
        return self._reduction_method(
            self.margin_activation(
                pos_scores - neg_scores + margins,
            )
        )


@parse_docdata
class MarginRankingLoss(MarginPairwiseLoss):
    r"""The pairwise hinge loss (i.e., margin ranking loss).

    .. math ::
        L(k, \bar{k}) = \max(0, f(k) - f(\bar{k}) + \lambda)

    Where $k$ are the positive triples, $\bar{k}$ are the negative triples, $f$ is the interaction function (e.g.,
    TransE has $f(h,r,t)=h+r-t$), $g(x)=\max(0,x)$ is the ReLU activation function,
    and $\lambda$ is the margin.

    .. seealso::

        MRL is closely related to :class:`pykeen.losses.SoftMarginRankingLoss`, only differing in that this loss
        uses the ReLU activation and :class:`pykeen.losses.SoftMarginRankingLoss` uses the softmax activation. MRL
        is also related to the :class:`pykeen.losses.PairwiseLogisticLoss` as this is a special case of the
        :class:`pykeen.losses.SoftMarginRankingLoss` with no margin.

    .. note::

        The related :mod:`torch` module is :class:`torch.nn.MarginRankingLoss`, but it can not be used
        interchangeably in PyKEEN because of the extended functionality implemented in PyKEEN's loss functions.
    ---
    name: Margin ranking
    """

    synonyms = {"Pairwise Hinge Loss"}

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=DEFAULT_MARGIN_HPO_STRATEGY,
    )

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        r"""Initialize the margin loss instance.

        :param margin:
            The margin by which positive and negative scores should be apart.
        :param reduction:
            The name of the reduction operation to aggregate the individual loss values from a batch to a scalar loss
            value. From {'mean', 'sum'}.
        """
        super().__init__(margin=margin, margin_activation="relu", reduction=reduction)


@parse_docdata
class SoftMarginRankingLoss(MarginPairwiseLoss):
    r"""The soft pairwise hinge loss (i.e., soft margin ranking loss).

    .. math ::
        L(k, \bar{k}) = \log(1 + \exp(f(k) - f(\bar{k}) + \lambda))

    Where $k$ are the positive triples, $\bar{k}$ are the negative triples, $f$ is the interaction function (e.g.,
    :class:`pykeen.models.TransE` has $f(h,r,t)=\mathbf{e}_h+\mathbf{r}_r-\mathbf{e}_t$), $g(x)=\log(1 + \exp(x))$
    is the softmax activation function, and $\lambda$ is the margin.

    .. seealso::

        When choosing `margin=0``, this loss becomes equivalent to :class:`pykeen.losses.SoftMarginRankingLoss`.
        It is also closely related to :class:`pykeen.losses.MarginRankingLoss`, only differing in that this loss
        uses the softmax activation and :class:`pykeen.losses.MarginRankingLoss` uses the ReLU activation.
    ---
    name: Soft margin ranking
    """

    hpo_default: ClassVar[Mapping[str, Any]] = dict(
        margin=DEFAULT_MARGIN_HPO_STRATEGY,
    )

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        """
        Initialize the loss.

        :param margin:
            the margin, cf. :meth:`MarginPairwiseLoss.__init__`
        :param reduction:
            the reduction, cf. :meth:`MarginPairwiseLoss.__init__`
        """
        super().__init__(margin=margin, margin_activation="softplus", reduction=reduction)
