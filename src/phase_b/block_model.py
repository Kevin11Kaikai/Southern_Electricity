"""Two-hour forward-block targets and exact block-value dispatch.

The official action always charges for eight consecutive quarter-hours and
later discharges for eight consecutive quarter-hours.  Consequently, the
decision depends on the 89 possible two-hour block values, rather than on
independent point-price accuracy alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .contracts import (
    BLOCK_STEPS,
    CHARGE_POWER,
    DISCHARGE_POWER,
    STEPS_PER_DAY,
)
from .dispatch import (
    DispatchDecision,
    LEGAL_WINDOWS,
    _stable_first_argmax,
    build_day_power,
)

BLOCKS_PER_DAY = STEPS_PER_DAY - BLOCK_STEPS + 1
_WINDOW_TC = np.fromiter((tc for tc, _ in LEGAL_WINDOWS), dtype=np.int16)
_WINDOW_TD = np.fromiter((td for _, td in LEGAL_WINDOWS), dtype=np.int16)


def forward_block_means(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return the 89 forward means of consecutive eight-step price blocks.

    Element ``i`` is ``mean(prices[i:i+8])``.  This function represents one
    complete 96-step day and therefore never crosses a day boundary.
    """

    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or values.size != STEPS_PER_DAY:
        raise ValueError(
            f"prices must be a one-dimensional vector of length "
            f"{STEPS_PER_DAY}; got shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("prices contains NaN or infinite values")
    windows = np.lib.stride_tricks.sliding_window_view(values, BLOCK_STEPS)
    return windows.mean(axis=1)


def _column_or_values(
    frame: pd.DataFrame,
    value: str | Sequence[object] | pd.Series | pd.Index,
    role: str,
) -> tuple[pd.Series, str]:
    if isinstance(value, str):
        if value not in frame.columns:
            raise KeyError(f"frame is missing {role} column {value!r}")
        return frame[value], value
    if len(value) != len(frame):
        raise ValueError(
            f"{role} length must match frame length: {len(value)} != {len(frame)}"
        )
    name = getattr(value, "name", None) or role
    return pd.Series(value, index=frame.index), str(name)


def make_forward_block_target(
    frame: pd.DataFrame,
    times: str | Sequence[object] | pd.Series | pd.Index = "times",
    target: str | Sequence[float] | pd.Series = "A",
) -> pd.Series:
    """Create an eight-step forward-mean target without crossing days.

    ``times`` and ``target`` may be column names or aligned sequences.  The
    returned Series preserves ``frame.index``.  The last seven rows of every
    complete day are NaN because no full forward block starts there.  Missing
    quarter-hours also make every affected block NaN; rows are never silently
    bridged across a timestamp gap.
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    time_values, _ = _column_or_values(frame, times, "times")
    target_values, target_name = _column_or_values(frame, target, "target")

    try:
        index = pd.DatetimeIndex(pd.to_datetime(time_values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"times cannot be parsed as datetimes: {exc}") from exc
    if index.hasnans:
        raise ValueError("times contains missing timestamps")
    if index.has_duplicates:
        raise ValueError("times contains duplicate timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError("times must be strictly increasing")
    if not (
        (index.minute % 15 == 0)
        & (index.second == 0)
        & (index.microsecond == 0)
    ).all():
        raise ValueError("times must lie on the 15-minute grid")

    try:
        numeric_target = pd.to_numeric(target_values, errors="raise").to_numpy(
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"target must be numeric: {exc}") from exc

    result = np.full(len(frame), np.nan, dtype=float)
    normalized = index.normalize()
    step = pd.Timedelta(minutes=15)

    for day in normalized.unique().sort_values():
        positions = np.flatnonzero(normalized == day)
        day_times = index[positions]
        offsets = ((day_times - day) / step).to_numpy(dtype=int)
        if np.any(offsets < 0) or np.any(offsets >= STEPS_PER_DAY):
            raise ValueError(f"{day.date()} contains a timestamp outside its market day")

        # Reindex each day to the complete grid.  This prevents a row-based
        # rolling operation from treating t-30min as adjacent to t-15min when
        # a label is missing.
        full_day = np.full(STEPS_PER_DAY, np.nan, dtype=float)
        full_day[offsets] = numeric_target[positions]
        block_windows = np.lib.stride_tricks.sliding_window_view(
            full_day, BLOCK_STEPS
        )
        means = block_windows.mean(axis=1)
        valid_starts = offsets < BLOCKS_PER_DAY
        result[positions[valid_starts]] = means[offsets[valid_starts]]

    return pd.Series(
        result,
        index=frame.index,
        name=f"{target_name}_forward_block_mean_{BLOCK_STEPS}",
    )


def optimize_from_block_values(
    block_values: Sequence[float] | np.ndarray,
    allow_idle: bool = True,
    *,
    score_scale: float = 1.0,
) -> DispatchDecision:
    """Select the best legal daily decision from 89 predicted block means.

    Active profit is exactly the eight-step price-power score implied by the
    block means.  All 3,321 legal ``(tc, td)`` pairs from :mod:`dispatch` are
    considered.  Idle wins a non-positive tie, matching :func:`optimize_day`.
    """

    values = np.asarray(block_values, dtype=float)
    if values.ndim != 1 or values.size != BLOCKS_PER_DAY:
        raise ValueError(
            f"block_values must be a one-dimensional vector of length "
            f"{BLOCKS_PER_DAY}; got shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("block_values contains NaN or infinite values")
    if not isinstance(allow_idle, (bool, np.bool_)):
        raise TypeError("allow_idle must be a boolean")
    if not np.isfinite(score_scale) or score_scale <= 0:
        raise ValueError("score_scale must be a finite positive number")

    active_scores = score_scale * BLOCK_STEPS * (
        DISCHARGE_POWER * values[_WINDOW_TD]
        + CHARGE_POWER * values[_WINDOW_TC]
    )
    best_index = _stable_first_argmax(active_scores)
    best_profit = float(active_scores[best_index])
    if allow_idle and best_profit <= 0.0:
        return DispatchDecision(
            power=build_day_power(),
            tc=None,
            td=None,
            predicted_profit=0.0,
            idle=True,
        )

    tc, td = LEGAL_WINDOWS[best_index]
    return DispatchDecision(
        power=build_day_power(tc, td),
        tc=tc,
        td=td,
        predicted_profit=best_profit,
        idle=False,
    )
