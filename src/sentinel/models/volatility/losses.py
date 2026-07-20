"""Focal loss, for the rare high-volatility class."""
import torch
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = (1 - p_t) ** gamma * ce
    if alpha is not None:
        a_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = a_t * loss
    return loss.mean()
