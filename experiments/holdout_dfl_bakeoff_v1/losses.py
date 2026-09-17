"""Decision-focused surrogates on the official 8+8 action space.

All methods consume a predicted 96-slot price curve and the training-day
true ``A``. Window scores reuse the notebook-12 convolution
``S_td - S_tc``. ``optimize_day`` remains the lock-in solver; these losses
only change the training surrogate.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from experiments.holdout_dispatch_loss_v1.loss import (
    block_sums,
    day_center,
    oracle_window_index,
    window_scores,
)
from src.phase_b.dispatch import LEGAL_WINDOWS, build_day_power, optimize_day

N_HARD_NEGATIVES = 64


def _as_price_batch(pred: torch.Tensor, y: torch.Tensor) -> None:
    if pred.shape != y.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != y shape {tuple(y.shape)}")
    if pred.ndim != 2 or pred.shape[1] != 96:
        raise ValueError(f"pred must have shape (batch, 96); got {tuple(pred.shape)}")


def _numpy_days(price: torch.Tensor) -> np.ndarray:
    return np.asarray(price.detach().cpu().numpy(), dtype=float)


def _power_stack(days: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    powers = [optimize_day(row).power.astype(np.float32, copy=False) for row in days]
    return torch.tensor(np.stack(powers, axis=0), device=device, dtype=dtype)


def expected_profit_loss(pred: torch.Tensor, y: torch.Tensor, *, tau: float = 0.1) -> torch.Tensor:
    """Soft expected window score: ``-sum softmax(s/τ) * true_score``.

    Cross-entropy on the oracle window is the one-hot special case. Scores are
    unscaled ``S_td - S_tc`` (same argmax as ``optimize_day``).
    """

    _as_price_batch(pred, y)
    if tau <= 0:
        raise ValueError("tau must be positive")
    hat_s = window_scores(block_sums(day_center(pred)))
    true_s = window_scores(block_sums(day_center(y.detach())))
    log_prob = F.log_softmax(hat_s / tau, dim=1)
    return -(torch.exp(log_prob) * true_s).sum(dim=1).mean()


def pairwise_margin_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    margin: float = 0.5,
    n_negatives: int = N_HARD_NEGATIVES,
) -> torch.Tensor:
    """Mandi-style pairwise hinge on hard negatives.

    ``L = mean max(0, m - (s_oracle - s_j))`` for the ``n_negatives`` true-score
    runners-up that day (excluding the oracle window).
    """

    _as_price_batch(pred, y)
    if margin < 0:
        raise ValueError("margin must be non-negative")
    hat_s = window_scores(block_sums(day_center(pred)))
    true_s = window_scores(block_sums(day_center(y.detach())))
    oracle = oracle_window_index(y)
    masked = true_s.clone()
    masked.scatter_(1, oracle.unsqueeze(1), torch.finfo(masked.dtype).min)
    k = min(int(n_negatives), masked.shape[1] - 1)
    _, neg_index = masked.topk(k, dim=1)
    s_oracle = hat_s.gather(1, oracle.unsqueeze(1))
    s_neg = hat_s.gather(1, neg_index)
    return F.relu(float(margin) - (s_oracle - s_neg)).mean()


def spoplus_loss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Linear-objective SPO+ on the 3,321 legal power vectors.

    Maximization form matching ``repairs.py``:
    ``L = (2p-y)·a*(2p-y) - 2p·a*(y) + y·a*(y)``, with both actions detached
    from ``optimize_day``. This trains the 96-slot head, not a P01 calibrator.
    """

    _as_price_batch(pred, y)
    y_det = y.detach()
    reflected = _numpy_days(2.0 * pred.detach() - y_det)
    oracle_days = _numpy_days(y_det)
    a_reflected = _power_stack(reflected, pred.device, pred.dtype)
    a_oracle = _power_stack(oracle_days, pred.device, pred.dtype)
    two_p = 2.0 * pred
    return (
        (two_p - y_det) * a_reflected - two_p * a_oracle + y_det * a_oracle
    ).sum(dim=1).mean()


class _DBBPower(torch.autograd.Function):
    """Pogancic ICLR 2020 black-box differentiation through ``optimize_day``."""

    @staticmethod
    def forward(ctx, pred: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        power = _power_stack(_numpy_days(pred), pred.device, pred.dtype)
        ctx.save_for_backward(pred, power, gamma)
        return power

    @staticmethod
    def backward(ctx, grad_power: torch.Tensor):  # type: ignore[override]
        pred, power, gamma = ctx.saved_tensors
        lam = pred.detach() - float(gamma.item()) * grad_power.detach()
        power_lam = _power_stack(_numpy_days(lam), pred.device, pred.dtype)
        return power - power_lam, None


def dbb_realized_loss(pred: torch.Tensor, y: torch.Tensor, *, gamma: float = 1.0) -> torch.Tensor:
    """Direct realized objective ``L = -y · power(optimize_day(p))`` with DBB."""

    _as_price_batch(pred, y)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    gamma_t = pred.new_tensor(float(gamma))
    power = _DBBPower.apply(pred, gamma_t)
    return -(y.detach() * power).sum(dim=1).mean()


def make_loss_fn(kind: str, **params):
    """Return ``fn(pred, y)`` for the bakeoff trainer."""

    if kind == "expected":
        tau = float(params.get("tau", 0.1))
        return lambda pred, y: expected_profit_loss(pred, y, tau=tau)
    if kind == "pairwise":
        margin = float(params.get("margin", 0.5))
        n_negatives = int(params.get("n_negatives", N_HARD_NEGATIVES))
        return lambda pred, y: pairwise_margin_loss(
            pred, y, margin=margin, n_negatives=n_negatives
        )
    if kind == "spoplus":
        return spoplus_loss
    if kind == "dbb":
        gamma = float(params.get("gamma", 1.0))
        return lambda pred, y: dbb_realized_loss(pred, y, gamma=gamma)
    raise ValueError(f"unknown DFL kind: {kind}")


def legal_window_count() -> int:
    return len(LEGAL_WINDOWS)


def idle_or_power(tc: int | None, td: int | None) -> np.ndarray:
    return build_day_power(tc, td)
