"""Small-data models aligned with day-ahead price-shape decisions.

These models intentionally complement boosted trees.  With only 360 complete
decision days, a regularised linear benchmark and a smoothed historical shape
are useful guards against high-variance model improvements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import TARGET_COL, TIME_COL


def day_center_target(frame: pd.DataFrame, target_col: str = TARGET_COL) -> np.ndarray:
    day = frame[TIME_COL].dt.normalize()
    target = frame[target_col].astype(float)
    return (target - target.groupby(day).transform("mean")).to_numpy()


def quarter_slot(times: pd.Series) -> np.ndarray:
    return (times.dt.hour.to_numpy() * 4 + times.dt.minute.to_numpy() // 15).astype(np.int16)


@dataclass
class SmoothedShapeClimatology:
    """Past-only weekday/quarter profile with global-profile shrinkage."""

    weekday_weight: float = 0.7
    target_mode: str = "day_centered"

    def fit(self, frame: pd.DataFrame) -> "SmoothedShapeClimatology":
        work = frame[[TIME_COL, TARGET_COL]].copy()
        work["slot"] = quarter_slot(work[TIME_COL])
        work["dow"] = work[TIME_COL].dt.dayofweek
        if self.target_mode == "day_centered":
            work["target"] = day_center_target(work)
        elif self.target_mode == "level":
            work["target"] = work[TARGET_COL].astype(float)
        else:
            raise ValueError(f"Unknown target mode: {self.target_mode}")
        self.global_slot_ = work.groupby("slot")["target"].mean()
        self.weekday_slot_ = work.groupby(["dow", "slot"])["target"].mean()
        self.global_mean_ = float(work["target"].mean())
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "global_slot_"):
            raise RuntimeError("Climatology must be fit before predict")
        slots = quarter_slot(frame[TIME_COL])
        dows = frame[TIME_COL].dt.dayofweek.to_numpy()
        global_values = np.array([self.global_slot_.get(int(slot), self.global_mean_) for slot in slots])
        weekday_values = np.array(
            [self.weekday_slot_.get((int(dow), int(slot)), global_value) for dow, slot, global_value in zip(dows, slots, global_values)]
        )
        return self.weekday_weight * weekday_values + (1.0 - self.weekday_weight) * global_values


def make_ridge_pipeline(alpha: float = 100.0) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=float(alpha))),
        ]
    )

