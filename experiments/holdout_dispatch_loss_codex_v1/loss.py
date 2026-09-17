"""Differentiable cost-sensitive ranking over the existing discrete actions."""

import torch
from torch.nn import functional as F

from experiments.holdout_dispatch_loss_v1.loss import (
    block_sums, day_center, dispatch_loss as cursor_dispatch_loss, window_scores,
)


def all_action_scores(prices: torch.Tensor) -> torch.Tensor:
    """3321 block spreads followed by idle=0; prices have shape (B,96)."""
    active = window_scores(block_sums(day_center(prices)))
    return torch.cat((active, torch.zeros_like(active[:, :1])), dim=1)


def soft_regret_from_scores(
    predicted: torch.Tensor, actual: torch.Tensor, tau: float
) -> torch.Tensor:
    """Expected hindsight shortfall under soft predicted action weights.

    No per-day normalization: costly days keep their raw score importance.
    Labels are detached; the finite-temperature surrogate is not a guarantee
    of better hard-argmax performance. Temperature is chosen on development.
    """
    if tau <= 0 or predicted.shape != actual.shape or predicted.ndim != 2:
        raise ValueError("matching (days, actions) scores and positive tau required")
    target = actual.detach()
    regret = target.max(dim=1, keepdim=True).values - target
    return (torch.softmax(predicted / tau, dim=1) * regret).sum(dim=1).mean()


def soft_regret_loss(
    pred: torch.Tensor, y: torch.Tensor, *, lam: float = 0.5, tau: float = 0.1
) -> torch.Tensor:
    if pred.shape != y.shape or lam < 0:
        raise ValueError("matching predictions/labels and non-negative lambda required")
    target = y.detach()
    block_loss = F.mse_loss(block_sums(day_center(pred)), block_sums(day_center(target)))
    regret_loss = soft_regret_from_scores(all_action_scores(pred), all_action_scores(target), tau)
    return block_loss + lam * regret_loss


def training_loss(pred, y, method, lam=0.5, tau=0.1):
    if method == "soft_regret":
        return soft_regret_loss(pred, y, lam=lam, tau=tau)
    if method == "cursor_ce":
        return cursor_dispatch_loss(pred, y, lam=lam, tau=tau)
    if method == "mse":
        return F.mse_loss(pred, y)
    raise ValueError(f"unknown method {method}")
