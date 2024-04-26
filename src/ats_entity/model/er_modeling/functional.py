from __future__ import unicode_literals, print_function, division
"""
    Tailored complex_interaction based on pykeen's implementation.
    (Refer to doc site https://pykeen.readthedocs.io/)
"""
import torch
import torch.nn.functional as F

from pykeen.utils import (
    ensure_complex,
    tensor_product,
)


def complex_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    h, t = ensure_complex(h, t)

    if r is None:
        return torch.real(tensor_product(h, torch.conj(t)).sum(dim=-1))
    else:
        r = next(ensure_complex(r))
        return torch.real(tensor_product(h, r, torch.conj(t)).sum(dim=-1))


def similarity_interaction(
    h: torch.FloatTensor,
    r: torch.FloatTensor,
    t: torch.FloatTensor,
) -> torch.FloatTensor:
    assert r is None, "similarity_interaction does not interact with relation."
    return F.cosine_similarity(h, t, dim=-1)
