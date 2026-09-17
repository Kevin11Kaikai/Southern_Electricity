"""Official-score evaluation for predicted prices and discrete dispatch."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .contracts import STEPS_PER_DAY, assert_valid_day_power
from .dispatch import DispatchDecision, optimize_day


def _finite_day(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size != STEPS_PER_DAY:
        raise ValueError(
            f"{name} must be a one-dimensional vector of length "
            f"{STEPS_PER_DAY}; got shape {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return array


def score_day(
    prices: Sequence[float] | np.ndarray,
    power: Sequence[float] | np.ndarray,
    *,
    score_scale: float = 1.0,
    validate: bool = True,
) -> float:
    """Return one day's raw official price-power score.

    The default is ``sum(prices * power)``.  No unverified efficiency or SOC
    term is introduced.  ``score_scale`` can represent a documented constant
    such as interval duration when a separate physical revenue report needs it.
    """

    price_values = _finite_day(prices, "prices")
    power_values = np.asarray(power, dtype=float)
    if validate:
        assert_valid_day_power(power_values)
    elif power_values.ndim != 1 or power_values.size != STEPS_PER_DAY:
        raise ValueError(
            f"power must be a one-dimensional vector of length "
            f"{STEPS_PER_DAY}; got shape {power_values.shape}"
        )
    if not np.isfinite(power_values).all():
        raise ValueError("power contains NaN or infinite values")
    if not np.isfinite(score_scale) or score_scale <= 0:
        raise ValueError("score_scale must be a finite positive number")
    return float(score_scale * np.dot(price_values, power_values))


def oracle_day(
    actual_prices: Sequence[float] | np.ndarray,
    allow_idle: bool = True,
    *,
    score_scale: float = 1.0,
) -> DispatchDecision:
    """Return the hindsight-optimal legal action for one actual-price day."""

    return optimize_day(
        actual_prices,
        allow_idle=allow_idle,
        score_scale=score_scale,
    )


def _capture_rate(realized: float, oracle: float) -> float:
    if oracle > 0:
        return float(realized / oracle)
    if np.isclose(realized, 0.0, rtol=0.0, atol=1e-12):
        return 1.0
    return float("nan")


def _coerce_evaluation_frame(
    frame_or_times: pd.DataFrame | Sequence[object] | pd.Series | pd.Index,
    y_true: Sequence[float] | np.ndarray | str | None,
    y_pred: Sequence[float] | np.ndarray | str | None,
    *,
    time_col: str,
    true_col: str,
    pred_col: str,
) -> pd.DataFrame:
    if isinstance(frame_or_times, pd.DataFrame):
        frame = frame_or_times
        actual_name = y_true if isinstance(y_true, str) else true_col
        predicted_name = y_pred if isinstance(y_pred, str) else pred_col
        required = [time_col, actual_name, predicted_name]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise KeyError(f"evaluation frame is missing columns: {missing}")
        result = frame[required].copy()
        result.columns = ["times", "y_true", "y_pred"]
    else:
        if y_true is None or y_pred is None or isinstance(y_true, str) or isinstance(y_pred, str):
            raise TypeError(
                "array form requires evaluate_price_predictions(times, y_true, y_pred)"
            )
        result = pd.DataFrame(
            {"times": frame_or_times, "y_true": y_true, "y_pred": y_pred}
        )

    try:
        result["times"] = pd.to_datetime(result["times"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"times cannot be parsed as datetimes: {exc}") from exc
    if result["times"].isna().any():
        raise ValueError("times contains missing timestamps")
    if result["times"].duplicated().any():
        raise ValueError("times contains duplicate timestamps")
    if not result["times"].is_monotonic_increasing:
        raise ValueError("times must be strictly increasing")

    for column in ("y_true", "y_pred"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def evaluate_price_predictions(
    frame_or_times: pd.DataFrame | Sequence[object] | pd.Series | pd.Index,
    y_true: Sequence[float] | np.ndarray | str | None = None,
    y_pred: Sequence[float] | np.ndarray | str | None = None,
    *,
    time_col: str = "times",
    true_col: str = "y_true",
    pred_col: str = "y_pred",
    allow_idle: bool = True,
    score_scale: float = 1.0,
    incomplete: str = "drop",
) -> dict[str, object]:
    """Evaluate price predictions through the official daily decision.

    Two call forms are supported::

        evaluate_price_predictions(times, y_true, y_pred)
        evaluate_price_predictions(frame, true_col="A", pred_col="prediction")

    A mapping is returned with ``summary`` (a plain dictionary) and ``daily``
    (one row per evaluated day).  A day is complete only if it has all 96
    timestamps from 00:00 to 23:45 and finite actual/predicted prices.
    ``incomplete='drop'`` records and skips incomplete days; ``'raise'`` makes
    them a hard error.  Missing actual prices are never imputed for scoring.
    """

    if incomplete not in {"drop", "raise"}:
        raise ValueError("incomplete must be either 'drop' or 'raise'")
    frame = _coerce_evaluation_frame(
        frame_or_times,
        y_true,
        y_pred,
        time_col=time_col,
        true_col=true_col,
        pred_col=pred_col,
    )

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    expected_offsets = pd.timedelta_range(
        start="0min", periods=STEPS_PER_DAY, freq="15min"
    )

    for day, group in frame.groupby(frame["times"].dt.normalize(), sort=True):
        expected = pd.DatetimeIndex(day + expected_offsets)
        actual_times = pd.DatetimeIndex(group["times"])
        finite = np.isfinite(group[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
        complete = len(group) == STEPS_PER_DAY and actual_times.equals(expected) and finite
        if not complete:
            reason = (
                f"requires 96 ordered quarter-hour timestamps and finite prices; "
                f"got {len(group)} rows"
            )
            if incomplete == "raise":
                raise ValueError(f"incomplete evaluation day {day.date()}: {reason}")
            skipped.append({"date": day, "reason": reason})
            continue

        actual = group["y_true"].to_numpy(dtype=float)
        predicted = group["y_pred"].to_numpy(dtype=float)
        decision = optimize_day(
            predicted,
            allow_idle=allow_idle,
            score_scale=score_scale,
        )
        oracle = oracle_day(
            actual,
            allow_idle=allow_idle,
            score_scale=score_scale,
        )
        realized = score_day(
            actual,
            decision.power,
            score_scale=score_scale,
        )
        oracle_profit = score_day(
            actual,
            oracle.power,
            score_scale=score_scale,
        )
        regret = float(oracle_profit - realized)

        rows.append(
            {
                "date": day,
                "tc": decision.tc,
                "td": decision.td,
                "idle": decision.idle,
                "predicted_profit": decision.predicted_profit,
                "realized_profit": realized,
                "oracle_tc": oracle.tc,
                "oracle_td": oracle.td,
                "oracle_profit": oracle_profit,
                "regret": regret,
                "capture_rate": _capture_rate(realized, oracle_profit),
                "rmse": float(np.sqrt(np.mean(np.square(actual - predicted)))),
                "mae": float(np.mean(np.abs(actual - predicted))),
            }
        )

    daily = pd.DataFrame(rows)
    skipped_days = pd.DataFrame(skipped)
    if daily.empty:
        raise ValueError("no complete days are available for evaluation")

    total_realized = float(daily["realized_profit"].sum())
    total_oracle = float(daily["oracle_profit"].sum())
    aggregate_capture = _capture_rate(total_realized, total_oracle)
    summary: dict[str, int | float] = {
        "days_evaluated": int(len(daily)),
        "days_skipped": int(len(skipped_days)),
        "mean_realized_profit": float(daily["realized_profit"].mean()),
        "mean_oracle_profit": float(daily["oracle_profit"].mean()),
        "mean_regret": float(daily["regret"].mean()),
        "total_realized_profit": total_realized,
        "total_oracle_profit": total_oracle,
        "oracle_capture_rate": aggregate_capture,
        "negative_profit_day_rate": float((daily["realized_profit"] < 0).mean()),
        "idle_day_rate": float(daily["idle"].mean()),
        "mean_daily_rmse": float(daily["rmse"].mean()),
        "mean_daily_mae": float(daily["mae"].mean()),
    }
    return {"summary": summary, "daily": daily, "skipped_days": skipped_days}


def contiguous_block_starts(
    dates: Sequence | np.ndarray | pd.Series,
    block_days: int,
) -> np.ndarray:
    """Return positional starts whose ``block_days`` span is calendar-contiguous.

    A moving-block bootstrap preserves short-range serial dependence only when
    every resampled block is a run of consecutive calendar days.  When the
    evaluated day set contains a calendar gap, positional slicing silently
    stitches non-adjacent days into one "block", which breaks that assumption
    and distorts the variance estimate.  This helper keeps exactly the starts
    whose span crosses no gap.

    ``dates`` must already be sorted ascending; duplicates are rejected because
    a paired daily table must hold one row per evaluated day.
    """

    index = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates)), errors="raise"))
    if len(index) == 0:
        raise ValueError("contiguous_block_starts requires at least one date")
    if not index.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending before block selection")
    if index.has_duplicates:
        raise ValueError("dates must be unique; a paired daily table holds one row per day")
    if block_days <= 0:
        raise ValueError("block_days must be a positive number of days")
    if len(index) < block_days:
        raise ValueError("not enough paired days for the requested block length")

    offsets = (index - index[0]).days.to_numpy(dtype=np.int64)
    last = len(offsets) - block_days
    candidates = np.arange(last + 1)
    spans = offsets[candidates + block_days - 1] - offsets[candidates]
    return candidates[spans == block_days - 1]
