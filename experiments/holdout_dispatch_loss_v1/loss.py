"""Dispatch-aligned loss on a 96-slot day.

``optimize_day`` scores legal 8+8 windows from eight-step block sums. This
module applies the same convolution, then combines block-sum MSE with a
cross-entropy that pushes the training-day oracle window to rank first.

Window scores are ``S_td - S_tc`` and are **not** multiplied by 1000, so the
softmax temperature stays in a usable range. Daily means are subtracted before
the block MSE; they cancel in the window scores anyway.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from src.phase_b.contracts import BLOCK_STEPS, STEPS_PER_DAY
from src.phase_b.dispatch import LEGAL_WINDOWS

BLOCKS_PER_DAY = STEPS_PER_DAY - BLOCK_STEPS + 1
N_WINDOWS = len(LEGAL_WINDOWS)

_INDEX_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _tc_td(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = f"{device.type}:{device.index}"
    cached = _INDEX_CACHE.get(key)
    if cached is not None and cached[0].device == device:
        return cached
    tc = torch.tensor([window[0] for window in LEGAL_WINDOWS], dtype=torch.long, device=device)
    td = torch.tensor([window[1] for window in LEGAL_WINDOWS], dtype=torch.long, device=device)
    _INDEX_CACHE[key] = (tc, td)
    return tc, td


def day_center(price: torch.Tensor) -> torch.Tensor:
    """Subtract the per-day mean. ``price`` is ``(batch, 96)``."""

    if price.ndim != 2 or price.shape[1] != STEPS_PER_DAY:
        raise ValueError(f"price must have shape (batch, {STEPS_PER_DAY}); got {tuple(price.shape)}")
    return price - price.mean(dim=1, keepdim=True)


def block_sums(price: torch.Tensor) -> torch.Tensor:
    """Eight-step block sums, ``(batch, 96) -> (batch, 89)``.

    Matches ``np.convolve(values, ones(8), mode='valid')`` inside ``optimize_day``.
    """

    if price.ndim != 2 or price.shape[1] != STEPS_PER_DAY:
        raise ValueError(f"price must have shape (batch, {STEPS_PER_DAY}); got {tuple(price.shape)}")
    kernel = torch.ones(1, 1, BLOCK_STEPS, dtype=price.dtype, device=price.device)
    return F.conv1d(price.unsqueeze(1), kernel).squeeze(1)


def window_scores(blocks: torch.Tensor) -> torch.Tensor:
    """Legal-window spreads ``S_td - S_tc``, shape ``(batch, 3321)``.

    Same sign and argmax as ``optimize_day`` (the official 1000× scale is omitted).
    """

    if blocks.ndim != 2 or blocks.shape[1] != BLOCKS_PER_DAY:
        raise ValueError(
            f"blocks must have shape (batch, {BLOCKS_PER_DAY}); got {tuple(blocks.shape)}"
        )
    tc, td = _tc_td(blocks.device)
    return blocks[:, td] - blocks[:, tc]


def oracle_window_index(price: torch.Tensor) -> torch.Tensor:
    """Argmax legal window from a detached true-price day, shape ``(batch,)``."""

    scores = window_scores(block_sums(price.detach()))
    return scores.argmax(dim=1)


def dispatch_loss(
    pred: torch.Tensor,
    y: torch.Tensor,
    *,
    lam: float = 0.5,
    tau: float = 0.1,
) -> torch.Tensor:
    """Block-sum MSE plus window ranking CE.

    ``pred`` and ``y`` are ``(batch, 96)`` absolute prices. Oracle labels come
    from ``y`` only and do not receive gradients. Idle is ignored: the target
    is always the best active window on that training day.
    """

    if pred.shape != y.shape:
        raise ValueError(f"pred shape {tuple(pred.shape)} != y shape {tuple(y.shape)}")
    if tau <= 0:
        raise ValueError("tau must be positive")
    if lam < 0:
        raise ValueError("lam must be non-negative")

    pred_c = day_center(pred)
    y_c = day_center(y)
    hat_blocks = block_sums(pred_c)
    true_blocks = block_sums(y_c)
    loss_block = F.mse_loss(hat_blocks, true_blocks)

    hat_scores = window_scores(hat_blocks)
    oracle = oracle_window_index(y)
    loss_rank = F.cross_entropy(hat_scores / tau, oracle)
    return loss_block + float(lam) * loss_rank
