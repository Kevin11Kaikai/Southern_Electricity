"""Combine per-seed day curves before ``optimize_day`` locks a window.

The headline rule is equal-weight averaging of the 96-step price curves.
Window voting is a pre-registered ablation: majority ``(tc, td)`` among seeds,
ties broken by ``optimize_day`` on the mean curve.  Combining happens on the
curves (or on locked windows), never by averaging already-scored realized profit.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.dispatch import build_day_power, optimize_day

Window = tuple[int | None, int | None]


def _as_stack(curves: np.ndarray) -> np.ndarray:
    stack = np.asarray(curves, dtype=float)
    if stack.ndim != 3 or stack.shape[-1] != STEPS_PER_DAY:
        raise ValueError(
            f"expected (n_seeds, n_days, {STEPS_PER_DAY}); got {stack.shape}"
        )
    if stack.shape[0] < 1:
        raise ValueError("need at least one seed curve")
    return stack


def mean_curves(curves: np.ndarray) -> np.ndarray:
    """Equal-weight mean over seeds. Shape ``(n_days, 96)``."""

    return _as_stack(curves).mean(axis=0)


def day_window(prices: np.ndarray) -> Window:
    decision = optimize_day(np.asarray(prices, dtype=float).reshape(STEPS_PER_DAY))
    return (decision.tc, decision.td)


def vote_windows(curves: np.ndarray) -> list[Window]:
    """Per-day majority ``(tc, td)``; ties use the mean curve's window."""

    stack = _as_stack(curves)
    averaged = mean_curves(stack)
    n_seeds, n_days, _ = stack.shape
    chosen: list[Window] = []
    for day in range(n_days):
        ballots = [day_window(stack[seed, day]) for seed in range(n_seeds)]
        counts = Counter(ballots)
        top = counts.most_common()
        best = top[0][1]
        tied = [key for key, count in top if count == best]
        if len(tied) == 1:
            chosen.append(tied[0])
        else:
            chosen.append(day_window(averaged[day]))
    return chosen


def windows_to_power(windows: Sequence[Window]) -> np.ndarray:
    rows = []
    for tc, td in windows:
        if tc is None and td is None:
            rows.append(build_day_power())
        else:
            rows.append(build_day_power(tc, td))
    return np.stack(rows, axis=0)


def mean_then_windows(curves: np.ndarray) -> list[Window]:
    """Headline combiner: average curves, then lock one window per day."""

    averaged = mean_curves(curves)
    return [day_window(day) for day in averaged]
