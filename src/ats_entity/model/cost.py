from __future__ import unicode_literals, print_function, division
import math
import torch
import torch.nn.functional as F
from model.datautil import fill_ignore_value

debugging=False

def tag_cost(logits, tag_targets, x_mask=None, ignore_index=-100):
    '''
        logits: logits
        Assumption: batch dim comes first
    '''
    # Loss
    proba = logits.reshape(-1, logits.shape[-1]) # (N, C) shape
    if x_mask:
        tag_targets = fill_ignore_value(tag_targets, x_mask, ignore_value=ignore_index)
    tag_targets_flat = tag_targets.reshape(-1)
    if debugging:
        if not torch.any(torch.logical_or(tag_targets_flat >= 0, \
                                        tag_targets_flat == ignore_index)):
            assert "Target class index underflow."
        if torch.any(tag_targets_flat >= proba.shape[-1]):
            assert "Target class index overflow."
    loss = F.cross_entropy(proba, tag_targets_flat.long(), ignore_index=ignore_index)
    return loss

def tag_acc(tag_proba, tag_targets, x_mask):
    '''
        tag_proba: either proba or logits
    '''
    # Acc
    tag_hat = tag_proba.argmax(dim=-1, keepdim=False)
    acc = torch.sum(torch.eq(tag_hat, tag_targets)*x_mask, dim=-1, keepdim=True)
    total = torch.sum(x_mask, dim=-1, keepdim=True) # Over length dim
    # Average over length dim and batch dim
    acc = torch.sum(acc.data.float() / total.data.float()) / tag_targets.shape[0]
    return acc


def span_iou(span1, span2, x1x2=True, DIoU=False, eps=1e-9):
    # Get the coordinates of bounding boxes
    if x1x2:
        # Offset x2 by 1 to ensure span size at least 1 for calculation.
        b1_x1, b1_x2 = span1[:,0], span1[:,1]+1
        b2_x1, b2_x2 = span2[:,0], span2[:,1]+1
    else:  # transform from xw to xx
        b1_x1, b1_x2 = span1[:,0] - span1[:,1] / 2, span1[:,0] + span1[:,1] / 2
        b2_x1, b2_x2 = span2[:,0] - span2[:,1] / 2, span2[:,0] + span2[:,1] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)

    # Union Area
    w1 = b1_x2 - b1_x1
    w2 = b2_x2 - b2_x1
    union = w1 + w2 - inter + eps

    iou = inter / union

    # The smallest enclosing width covers the two spans.
    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    if DIoU:  # https://ojs.aaai.org/index.php/AAAI/article/view/6999
        c2 = cw**2 + eps  # Squared diagonal distance.
        rho2 = (((b2_x1+b2_x2) - (b1_x1+b1_x2))**2) / 4  # Squared center distance.
        iou = iou - rho2/c2

    return iou
