"""Readout heads (Design A): tiny MLPs on the frozen backbone's final hidden
state, one per primitive.

Per-primitive semantics (the load-bearing asymmetries of the reference):
  - Noul   is ABSOLUTE:  sigmoid(z) -> P(true), trained with BCE against soft
           targets. Never a yes/no pair — that would be the relative readout,
           and P(A) + P(not-A) != 1 is reference behavior we must reproduce.
  - Choice is RELATIVE:  per-option score z_i from a shared head; softmax over
           the row's option group (in-loss, so gradients shape separation);
           distribution sums to exactly 1 at the API layer.
  - Score  levels are INDEPENDENT: per-level score z_l, trained with
           per-level BCE ("does the state match level l?"), normalized at the
           API layer only (reference: numbers-only levels fail; each level is
           judged alone, never against its neighbors).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DROPOUT = 0.05
HIDDEN = 1024


class PassMLP(nn.Module):
    """in_dim -> HIDDEN -> GELU -> dropout -> 1."""

    def __init__(self, in_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (N, in_dim) -> (N,) scores."""
        return self.net(h).squeeze(-1)


def noul_loss(logits, targets, weights) -> torch.Tensor:
    """Soft-target BCE = a proper scoring rule generalized to distributional
    targets; weights down-weight teacher-disagreement rows."""
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)


def choice_loss(scores: torch.Tensor, group_sizes: list[int], targets: torch.Tensor) -> torch.Tensor:
    """CE over each row's option group. scores is flat (sum(group_sizes),);
    targets are 1.0 for the gold option's pass, 0.0 otherwise (one-hot)."""
    loss = scores.new_zeros(())
    offset = 0
    for k in group_sizes:
        s = scores[offset : offset + k]
        t = targets[offset : offset + k]
        loss = loss - (F.log_softmax(s, dim=0) * t).sum() / k
        offset += k
    return loss


def score_level_loss(logits, targets) -> torch.Tensor:
    """Per-level independent BCE; targets are 0/1 'state matches this level'."""
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def head_scores(head: PassMLP, h: torch.Tensor) -> torch.Tensor:
    was_training = head.training
    head.eval()
    z = head(h)
    if was_training:
        head.train()
    return z