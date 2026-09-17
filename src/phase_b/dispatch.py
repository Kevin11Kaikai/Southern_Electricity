"""Exact enumeration of the official daily dispatch action space."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import (
    ACTIVE_WINDOW_COUNT,
    BLOCK_STEPS,
    CHARGE_POWER,
    CHARGE_START_MAX,
    DISCHARGE_POWER,
    DISCHARGE_START_MAX,
    STEPS_PER_DAY,
    assert_valid_day_power,
)


def enumerate_legal_windows() -> tuple[tuple[int, int], ...]:
    """Return all 3,321 legal ``(tc, td)`` active-day windows.

    Windows are ordered by earliest charge start and then earliest discharge
    start.  This order is also the deterministic tie-break used by
    :func:`optimize_day`.
    """

    windows = tuple(
        (tc, td)
        for tc in range(CHARGE_START_MAX + 1)
        for td in range(tc + BLOCK_STEPS, DISCHARGE_START_MAX + 1)
    )
    if len(windows) != ACTIVE_WINDOW_COUNT:  # defensive guard on constants
        raise RuntimeError(
            f"contract constants generated {len(windows)} windows, "
            f"expected {ACTIVE_WINDOW_COUNT}"
        )
    return windows


LEGAL_WINDOWS = enumerate_legal_windows()
_TC = np.fromiter((tc for tc, _ in LEGAL_WINDOWS), dtype=np.int16)
_TD = np.fromiter((td for _, td in LEGAL_WINDOWS), dtype=np.int16)

# Computing an eight-step spread takes several reductions, a subtraction, and
# one or more multiplications.  Algebraically identical paths (point prices vs
# pre-computed block means) can therefore differ by a handful of float64 ULPs.
# Sixty-four ULPs remains many orders of magnitude below a material market
# score difference while comfortably covering those alternate operation
# orders.  It is relative to the score magnitude, so positive unit changes do
# not change which window wins.
_SCORE_TIE_RTOL = 64.0 * np.finfo(np.float64).eps


def _stable_first_argmax(scores: Sequence[float] | np.ndarray) -> int:
    """Return the first index within a scale-aware tolerance of the maximum.

    Input order is the deterministic priority order.  For dispatch scores this
    is exactly ``LEGAL_WINDOWS`` order: earliest charge start, then earliest
    discharge start.  Near ties are intentionally resolved by that order rather
    than by insignificant differences introduced by floating-point reduction.
    """

    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty one-dimensional vector")
    if not np.isfinite(values).all():
        raise ValueError("scores contains NaN or infinite values")

    maximum = float(np.max(values))
    scale = float(np.max(np.abs(values)))
    tolerance = _SCORE_TIE_RTOL * scale
    near_best = np.flatnonzero(values >= maximum - tolerance)
    # ``maximum`` is an element of the finite vector, so this cannot be empty.
    return int(near_best[0])


@dataclass(frozen=True)
class DispatchDecision:
    """The exact decision selected for one market day."""

    power: np.ndarray
    tc: int | None
    td: int | None
    predicted_profit: float
    idle: bool

    def __post_init__(self) -> None:
        values = np.asarray(self.power, dtype=float).copy()
        assert_valid_day_power(values)
        values.setflags(write=False)
        object.__setattr__(self, "power", values)

        if self.idle:
            if self.tc is not None or self.td is not None:
                raise ValueError("idle decision must have tc=None and td=None")
            if not np.all(values == 0.0):
                raise ValueError("idle decision must contain all-zero power")
        elif self.tc is None or self.td is None:
            raise ValueError("active decision must define both tc and td")
        else:
            expected = build_day_power(self.tc, self.td)
            if not np.array_equal(values, expected):
                raise ValueError("tc/td do not describe the supplied power vector")
        if not np.isfinite(self.predicted_profit):
            raise ValueError("predicted_profit must be finite")


def _price_vector(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(prices, dtype=float)
    if values.ndim != 1 or values.size != STEPS_PER_DAY:
        raise ValueError(
            f"prices must be a one-dimensional vector of length "
            f"{STEPS_PER_DAY}; got shape {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("prices contains NaN or infinite values")
    return values


def build_day_power(tc: int | None = None, td: int | None = None) -> np.ndarray:
    """Build one legal daily power vector.

    Passing ``tc=None, td=None`` creates the idle action.  Active actions must
    satisfy ``0 <= tc <= 80`` and ``tc + 8 <= td <= 88``.
    """

    if tc is None and td is None:
        return np.zeros(STEPS_PER_DAY, dtype=float)
    if tc is None or td is None:
        raise ValueError("tc and td must either both be integers or both be None")
    if isinstance(tc, bool) or isinstance(td, bool):
        raise TypeError("tc and td must be integer indices, not booleans")
    if not isinstance(tc, (int, np.integer)) or not isinstance(td, (int, np.integer)):
        raise TypeError("tc and td must be integer indices")
    tc = int(tc)
    td = int(td)
    if not 0 <= tc <= CHARGE_START_MAX:
        raise ValueError(f"tc must be in [0, {CHARGE_START_MAX}]; got {tc}")
    if not tc + BLOCK_STEPS <= td <= DISCHARGE_START_MAX:
        raise ValueError(
            f"td must be in [tc + {BLOCK_STEPS}, {DISCHARGE_START_MAX}]; "
            f"got tc={tc}, td={td}"
        )

    power = np.zeros(STEPS_PER_DAY, dtype=float)
    power[tc : tc + BLOCK_STEPS] = CHARGE_POWER
    power[td : td + BLOCK_STEPS] = DISCHARGE_POWER
    assert_valid_day_power(power)
    return power


def optimize_day(
    prices: Sequence[float] | np.ndarray,
    allow_idle: bool = True,
    *,
    score_scale: float = 1.0,
) -> DispatchDecision:
    """Select the globally best legal daily action under ``prices``.

    ``score_scale`` multiplies ``sum(price * power)``.  Keep it at ``1.0`` for
    the official raw price-power score.  A value such as ``0.25`` may be used
    for a physical 15-minute energy-revenue report, without changing the
    selected action as long as it is positive.

    When idle is tied with the best active action, idle wins.  Active-action
    ties are resolved by the stable ordering returned by
    :func:`enumerate_legal_windows`.
    """

    values = _price_vector(prices)
    if not isinstance(allow_idle, (bool, np.bool_)):
        raise TypeError("allow_idle must be a boolean")
    if not np.isfinite(score_scale) or score_scale <= 0:
        raise ValueError("score_scale must be a finite positive number")

    # There are 89 possible consecutive eight-step blocks in a 96-step day.
    block_sums = np.convolve(values, np.ones(BLOCK_STEPS), mode="valid")
    active_scores = score_scale * (
        DISCHARGE_POWER * block_sums[_TD] + CHARGE_POWER * block_sums[_TC]
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


def optimize_days(
    prices: Sequence[Sequence[float]] | np.ndarray,
    allow_idle: bool = True,
    *,
    score_scale: float = 1.0,
) -> tuple[DispatchDecision, ...]:
    """Optimize a two-dimensional ``(n_days, 96)`` price matrix."""

    matrix = np.asarray(prices, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != STEPS_PER_DAY:
        raise ValueError(
            f"prices must have shape (n_days, {STEPS_PER_DAY}); "
            f"got {matrix.shape}"
        )
    return tuple(
        optimize_day(day, allow_idle=allow_idle, score_scale=score_scale)
        for day in matrix
    )
