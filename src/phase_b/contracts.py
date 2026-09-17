"""Official discrete dispatch contract for one 96-step market day."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

STEPS_PER_DAY = 96
BLOCK_STEPS = 8
POWER_MAGNITUDE = 1000.0
CHARGE_POWER = -POWER_MAGNITUDE
DISCHARGE_POWER = POWER_MAGNITUDE
CHARGE_START_MAX = 80
DISCHARGE_START_MAX = 88
ACTIVE_WINDOW_COUNT = 3321
IDLE_ACTION_COUNT = 1


class ContractViolation(ValueError):
    """Raised when a power schedule violates the official dispatch rules."""


def _as_numeric_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"expected a one-dimensional vector, got shape {array.shape}")
    return array


def _is_contiguous(indices: np.ndarray) -> bool:
    return bool(indices.size and np.all(np.diff(indices) == 1))


def validate_day_power(
    power: Sequence[float] | np.ndarray,
    *,
    atol: float = 0.0,
) -> list[str]:
    """Return all violations for one day of dispatch power.

    A legal active day has exactly eight consecutive ``-1000`` values followed
    by exactly eight consecutive ``+1000`` values.  All other values are zero.
    The all-zero idle action is also legal.

    Parameters
    ----------
    power:
        One-dimensional sequence containing exactly 96 values.
    atol:
        Absolute tolerance for values written by a numeric solver or CSV
        round-trip.  The default is deliberately strict (exact values).
    """

    errors: list[str] = []
    if not np.isfinite(atol) or atol < 0 or atol >= POWER_MAGNITUDE / 2:
        return [
            "atol must be finite, non-negative, and smaller than half the "
            "distance between legal power levels"
        ]

    try:
        values = _as_numeric_vector(power)
    except (TypeError, ValueError) as exc:
        return [str(exc)]

    if values.size != STEPS_PER_DAY:
        return [f"expected {STEPS_PER_DAY} power values, got {values.size}"]

    if not np.isfinite(values).all():
        errors.append("power contains NaN or infinite values")

    is_charge = np.isclose(values, CHARGE_POWER, rtol=0.0, atol=atol)
    is_idle = np.isclose(values, 0.0, rtol=0.0, atol=atol)
    is_discharge = np.isclose(values, DISCHARGE_POWER, rtol=0.0, atol=atol)
    allowed = is_charge | is_idle | is_discharge
    if not allowed.all():
        bad = np.flatnonzero(~allowed)
        preview = ", ".join(str(int(i)) for i in bad[:8])
        errors.append(f"power has values outside {{-1000, 0, 1000}} at indices {preview}")

    # An all-zero schedule is the one legal action without charge/discharge.
    if is_idle.all():
        return errors

    charge_indices = np.flatnonzero(is_charge)
    discharge_indices = np.flatnonzero(is_discharge)

    if charge_indices.size != BLOCK_STEPS:
        errors.append(
            f"active day must contain exactly {BLOCK_STEPS} charge steps; "
            f"got {charge_indices.size}"
        )
    if discharge_indices.size != BLOCK_STEPS:
        errors.append(
            f"active day must contain exactly {BLOCK_STEPS} discharge steps; "
            f"got {discharge_indices.size}"
        )

    charge_contiguous = _is_contiguous(charge_indices)
    discharge_contiguous = _is_contiguous(discharge_indices)
    if charge_indices.size and not charge_contiguous:
        errors.append("charge steps must form one consecutive block")
    if discharge_indices.size and not discharge_contiguous:
        errors.append("discharge steps must form one consecutive block")

    if charge_indices.size:
        tc = int(charge_indices[0])
        if tc > CHARGE_START_MAX:
            errors.append(f"charge start tc must be <= {CHARGE_START_MAX}; got {tc}")
    else:
        tc = None

    if discharge_indices.size:
        td = int(discharge_indices[0])
        if td > DISCHARGE_START_MAX:
            errors.append(
                f"discharge start td must be <= {DISCHARGE_START_MAX}; got {td}"
            )
    else:
        td = None

    if tc is not None and td is not None and td < tc + BLOCK_STEPS:
        errors.append(
            f"discharge must start after charging completes: td >= tc + "
            f"{BLOCK_STEPS}; got tc={tc}, td={td}"
        )

    return errors


def assert_valid_day_power(
    power: Sequence[float] | np.ndarray,
    *,
    atol: float = 0.0,
) -> None:
    """Raise :class:`ContractViolation` if a daily schedule is illegal."""

    errors = validate_day_power(power, atol=atol)
    if errors:
        raise ContractViolation("; ".join(errors))


def validate_dispatch(
    times: Sequence[object] | pd.Series | pd.Index,
    power: Sequence[float] | np.ndarray,
    *,
    atol: float = 0.0,
    require_consecutive_days: bool = True,
) -> list[str]:
    """Return timestamp and daily-action violations for a full submission.

    Each included date must contain the exact local-time grid from 00:00 to
    23:45 in 15-minute increments.  By default, included dates must also be
    consecutive.
    """

    errors: list[str] = []
    try:
        values = _as_numeric_vector(power)
    except (TypeError, ValueError) as exc:
        return [str(exc)]

    try:
        index = pd.DatetimeIndex(pd.to_datetime(times, errors="raise"))
    except (TypeError, ValueError) as exc:
        return [f"times cannot be parsed as datetimes: {exc}"]

    if len(index) != values.size:
        return [f"times/power length mismatch: {len(index)} != {values.size}"]
    if len(index) == 0:
        return ["dispatch must contain at least one complete day"]
    if index.hasnans:
        # Normalising/grouping NaT can itself fail, so stop after reporting the
        # structural error rather than letting validation raise incidentally.
        return ["times contains missing timestamps"]
    if index.has_duplicates:
        errors.append("times contains duplicate timestamps")
    if not index.is_monotonic_increasing:
        errors.append("times must be strictly increasing")

    days = index.normalize().unique().sort_values()
    if require_consecutive_days and len(days) > 1:
        expected_days = pd.date_range(days[0], days[-1], freq="D")
        if not days.equals(expected_days):
            errors.append("dispatch dates must be consecutive")

    for day in days:
        positions = np.flatnonzero(index.normalize() == day)
        day_times = index[positions]
        expected = pd.date_range(day, periods=STEPS_PER_DAY, freq="15min")
        if len(day_times) != STEPS_PER_DAY or not day_times.equals(expected):
            errors.append(
                f"{day.date()} must contain exactly 96 ordered timestamps "
                "from 00:00 through 23:45"
            )
            continue
        for violation in validate_day_power(values[positions], atol=atol):
            errors.append(f"{day.date()}: {violation}")

    return errors


def assert_valid_dispatch(
    times: Sequence[object] | pd.Series | pd.Index,
    power: Sequence[float] | np.ndarray,
    *,
    atol: float = 0.0,
    require_consecutive_days: bool = True,
) -> None:
    """Raise :class:`ContractViolation` if a full dispatch is illegal."""

    errors = validate_dispatch(
        times,
        power,
        atol=atol,
        require_consecutive_days=require_consecutive_days,
    )
    if errors:
        raise ContractViolation("; ".join(errors))
