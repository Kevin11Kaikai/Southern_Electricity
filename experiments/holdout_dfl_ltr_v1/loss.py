"""Decision-focused and learning-to-rank losses for the discrete 8+8 contract.

Every loss here is built on the *same* block-sum pipeline that ``optimize_day``
uses, reusing ``experiments.holdout_dispatch_loss_v1.loss`` so the convention is
provably identical to notebook 12:

    day_center -> conv1d(8) -> 89 block sums -> s(w) = S[td] - S[tc]  (3321 windows)

The official score is ``1000 * s(w)``; the factor is dropped throughout (as in
nb12) so softmax temperatures stay in a usable range.  Because the score is
linear in the price vector and the feasible set has only 3321 elements, the
Gibbs distribution over actions is *exact* -- no solution cache (Mandi et al.,
ICML 2022), no blackbox differentiation (Pogancic et al., ICLR 2020) and no
Monte-Carlo perturbation (Berthet et al., NeurIPS 2020) is required.

Routes
------
R1 ``listwise_loss``      Mandi et al. ICML 2022, Eq. 16 -- target is a softmax
                          over the TRUE window values, not a one-hot.  The nb12
                          loss is the ``tau_tgt -> 0`` limit of this.
R2 ``pairwise_diff_loss`` Mandi et al. ICML 2022, Eq. 13 -- regress the window
                          score *difference* onto the true difference.
R3 ``spo_plus_loss``      Elmachtoub & Grigas, Management Science 2022, adapted
                          to a maximisation objective.
R6 ``pinball_loss`` and ``peak_weighted_loss`` -- cheap task-flavoured
                          baselines that the decision-focused routes must beat.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from experiments.holdout_dispatch_loss_v1.loss import (
    BLOCKS_PER_DAY,
    N_WINDOWS,
    block_sums,
    day_center,
    oracle_window_index,
    window_scores,
)
from src.phase_b.contracts import BLOCK_STEPS, STEPS_PER_DAY

__all__ = [
    "BLOCKS_PER_DAY",
    "N_WINDOWS",
    "block_sum_mse",
    "listwise_loss",
    "pairwise_diff_loss",
    "peak_weighted_loss",
    "pinball_loss",
    "spo_plus_loss",
    "build_loss",
]


def _check(pred: torch.Tensor, y: torch.Tensor) -> None:
    if pred.shape != y.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != y shape {tuple(y.shape)}")
    if pred.ndim != 2 or pred.shape[1] != STEPS_PER_DAY:
        raise ValueError(f"expected (batch, {STEPS_PER_DAY}); got {tuple(pred.shape)}")


def _scores(pred: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(s_hat, s_true)`` of shape ``(batch, 3321)``; ``s_true`` detached.

    Day centering is applied first.  It changes nothing about the scores -- the
    contract is invariant to ``p -> a*p + b`` for ``a > 0`` -- but it keeps the
    block-sum MSE anchor on the decision-relevant part of the signal.
    """

    hat_blocks = block_sums(day_center(pred))
    true_blocks = block_sums(day_center(y)).detach()
    return window_scores(hat_blocks), window_scores(true_blocks)


def block_sum_mse(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """MSE on the 89 day-centred block sums -- the statistic dispatch reads."""

    _check(pred, y)
    return F.mse_loss(block_sums(day_center(pred)), block_sums(day_center(y)).detach())


# ---------------------------------------------------------------- R1 listwise


def listwise_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    lam: float = 1.0,
    tau_tgt: float = 0.25,
    tau_pred: float = 0.1,
) -> torch.Tensor:
    """Block-sum MSE plus a value-weighted listwise cross-entropy.

    Mandi et al. (ICML 2022) Eq. 16, maximisation form over the full feasible
    set::

        q     = softmax( s(y)    / tau_tgt )      # target, detached
        pi    = softmax( s(pred) / tau_pred )
        L     = MSE(block sums) + lam * CE(pi, q)

    Two temperatures are used rather than Mandi's single ``tau``: ``tau_tgt``
    controls how much probability mass the near-optimal windows keep (a property
    of the data -- the 50th best window is worth 92.8% of the oracle on this
    holdout) while ``tau_pred`` controls how sharp the prediction is asked to be
    (an optimisation knob).  As ``tau_tgt -> 0`` the target collapses to the
    one-hot oracle window and this reduces exactly to the notebook-12 loss.
    """

    _check(pred, y)
    if tau_tgt <= 0 or tau_pred <= 0:
        raise ValueError("tau_tgt and tau_pred must be positive")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    s_hat, s_true = _scores(pred, y)
    target = F.softmax(s_true / tau_tgt, dim=1)
    log_pi = F.log_softmax(s_hat / tau_pred, dim=1)
    loss_rank = -(target * log_pi).sum(dim=1).mean()
    return block_sum_mse(pred, y) + float(lam) * loss_rank


# -------------------------------------------------------- R2 pairwise difference


def pairwise_diff_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    lam: float = 1.0,
    n_pairs: int = 32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Block-sum MSE plus Mandi et al. Eq. 13 on sampled window pairs.

    ``[(s_u(pred) - s_v(pred)) - (s_u(y) - s_v(y))]^2`` averaged over pairs.  The
    first pair of every sample is the *hard* one: the true optimum against the
    model's own current argmax.  The rest are drawn uniformly from the 3321
    legal windows.  Score differences are invariant to any ``p -> a*p + b`` with
    ``a = 1``, so this term never spends capacity on the daily level.
    """

    _check(pred, y)
    if n_pairs < 1:
        raise ValueError("n_pairs must be at least 1")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    s_hat, s_true = _scores(pred, y)
    batch = s_hat.shape[0]
    device = s_hat.device
    rows = torch.arange(batch, device=device).unsqueeze(1)

    best_true = s_true.argmax(dim=1, keepdim=True)
    best_hat = s_hat.detach().argmax(dim=1, keepdim=True)
    extra = max(0, int(n_pairs) - 1)
    if extra:
        left = torch.randint(
            0, N_WINDOWS, (batch, extra), device=device, generator=generator
        )
        right = torch.randint(
            0, N_WINDOWS, (batch, extra), device=device, generator=generator
        )
        u = torch.cat([best_true, left], dim=1)
        v = torch.cat([best_hat, right], dim=1)
    else:
        u, v = best_true, best_hat

    gap_hat = s_hat[rows, u] - s_hat[rows, v]
    gap_true = s_true[rows, u] - s_true[rows, v]
    loss_pair = ((gap_hat - gap_true) ** 2).mean()
    return block_sum_mse(pred, y) + float(lam) * loss_pair


# ------------------------------------------------------------------- R3 SPO+


def spo_plus_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    w_spo: float = 1.0,
    anchor: float = 1.0,
) -> torch.Tensor:
    """SPO+ (Elmachtoub & Grigas 2022) for the maximisation contract.

    With ``w*(c) = argmax_w c' z_w``::

        L_SPO+(p_hat, p) = max_w (2*p_hat - p)' z_w
                           - 2 * p_hat' z_{w*(p)}
                           + p' z_{w*(p)}

    It is convex in ``p_hat``, equals 0 at ``p_hat = p`` and upper-bounds the
    decision regret.  Its subgradient is ``2 (z_{w*(2 p_hat - p)} - z_{w*(p)})``;
    here both ``argmax`` calls are the same exhaustive search ``optimize_day``
    performs, executed as an argmax over the ``(batch, 3321)`` score tensor, so
    no solver round-trip is needed.  The official 1000x scale is dropped.

    ``anchor`` weights a block-sum MSE term that keeps the predicted level from
    drifting; SPO+ alone is scale-free in a way that destroys point accuracy.
    """

    _check(pred, y)
    if w_spo < 0 or anchor < 0:
        raise ValueError("w_spo and anchor must be non-negative")

    s_hat, s_true = _scores(pred, y)
    s_mixed = window_scores(block_sums(day_center(2.0 * pred - y)))

    best = s_true.argmax(dim=1, keepdim=True)
    term_max = s_mixed.max(dim=1).values
    term_hat = s_hat.gather(1, best).squeeze(1)
    term_true = s_true.gather(1, best).squeeze(1)
    loss_spo = (term_max - 2.0 * term_hat + term_true).mean()
    return float(anchor) * block_sum_mse(pred, y) + float(w_spo) * loss_spo


# ----------------------------------------------------------- R6 cheap baselines


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, *, quantile: float = 0.5) -> torch.Tensor:
    """Quantile (pinball) loss over all 96 slots."""

    _check(pred, y)
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in (0, 1)")
    error = y - pred
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def peak_weighted_loss(pred: torch.Tensor, y: torch.Tensor, *, alpha: float = 4.0) -> torch.Tensor:
    """Point MSE with the 16 slots of the training-day oracle window upweighted.

    The mask comes from the true prices of that training day only, which is the
    same supervision ``optimize_day`` labels provide elsewhere in this campaign.
    """

    _check(pred, y)
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    with torch.no_grad():
        index = oracle_window_index(y)
        from src.phase_b.dispatch import LEGAL_WINDOWS

        tc = torch.tensor(
            [LEGAL_WINDOWS[int(i)][0] for i in index.tolist()], device=y.device
        ).unsqueeze(1)
        td = torch.tensor(
            [LEGAL_WINDOWS[int(i)][1] for i in index.tolist()], device=y.device
        ).unsqueeze(1)
        slots = torch.arange(STEPS_PER_DAY, device=y.device).unsqueeze(0)
        in_charge = (slots >= tc) & (slots < tc + BLOCK_STEPS)
        in_discharge = (slots >= td) & (slots < td + BLOCK_STEPS)
        weight = 1.0 + float(alpha) * (in_charge | in_discharge).to(y.dtype)

    squared = (pred - y) ** 2
    return (weight * squared).sum() / weight.sum()


# ------------------------------------------------------------------- registry


def build_loss(route: str, params: dict[str, float]):
    """Return ``fn(pred, y) -> loss`` for one route/config pair."""

    if route == "r1_listwise":
        return lambda p, t: listwise_loss(
            p,
            t,
            lam=params["lam"],
            tau_tgt=params["tau_tgt"],
            tau_pred=params["tau_pred"],
        )
    if route == "r2_pairdiff":
        return lambda p, t: pairwise_diff_loss(
            p, t, lam=params["lam"], n_pairs=int(params["n_pairs"])
        )
    if route == "r3_spo":
        return lambda p, t: spo_plus_loss(p, t, w_spo=params["w_spo"], anchor=1.0)
    if route == "r6_cheap":
        kind = str(params["kind"])
        if kind == "pinball":
            return lambda p, t: pinball_loss(p, t, quantile=params["param"])
        if kind == "peakw":
            return lambda p, t: peak_weighted_loss(p, t, alpha=params["param"])
        raise ValueError(f"unknown r6 kind: {kind}")
    raise ValueError(f"unknown route: {route}")
