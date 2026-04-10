# focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    alpha : weight for positive class (>0.5 - up-weight rare class)
    gamma : focusing exponent (higher - more focus on hard examples)
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()
