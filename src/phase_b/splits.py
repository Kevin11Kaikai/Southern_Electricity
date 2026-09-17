"""Whole-day expanding-window validation for decision-oriented backtests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DayFold:
    fold: int
    train_dates: pd.DatetimeIndex
    validation_dates: pd.DatetimeIndex

    @property
    def train_start(self) -> pd.Timestamp:
        return self.train_dates[0]

    @property
    def train_end(self) -> pd.Timestamp:
        return self.train_dates[-1]

    @property
    def validation_start(self) -> pd.Timestamp:
        return self.validation_dates[0]

    @property
    def validation_end(self) -> pd.Timestamp:
        return self.validation_dates[-1]


@dataclass(frozen=True)
class DaySplitPlan:
    development_dates: pd.DatetimeIndex
    holdout_dates: pd.DatetimeIndex
    folds: tuple[DayFold, ...]


def make_day_split_plan(
    complete_dates: pd.DatetimeIndex,
    *,
    n_splits: int = 5,
    validation_days: int = 30,
    holdout_days: int = 59,
) -> DaySplitPlan:
    dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(complete_dates).unique()))
    required = holdout_days + n_splits * validation_days + 1
    if len(dates) < required:
        raise ValueError(f"Need at least {required} complete days, found {len(dates)}")
    development = dates[:-holdout_days]
    holdout = dates[-holdout_days:]
    first_validation = len(development) - n_splits * validation_days
    folds: list[DayFold] = []
    for fold in range(n_splits):
        start = first_validation + fold * validation_days
        stop = start + validation_days
        train_dates = development[:start]
        validation_dates = development[start:stop]
        if train_dates[-1] >= validation_dates[0]:
            raise AssertionError("Training dates must strictly precede validation dates")
        folds.append(DayFold(fold=fold + 1, train_dates=train_dates, validation_dates=validation_dates))
    return DaySplitPlan(development_dates=development, holdout_dates=holdout, folds=tuple(folds))


def mask_for_dates(times: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    normalized = pd.to_datetime(times).dt.normalize()
    return normalized.isin(dates).to_numpy()

