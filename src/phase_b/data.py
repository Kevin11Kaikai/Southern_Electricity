"""Data contracts and loading utilities for the Phase B pipeline.

The important design choice is that lag/rolling features are built on the
complete forecast-feature timeline before sparse labels are joined.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PhaseBConfig


TIME_COL = "times"
TARGET_COL = "A"
FORECAST_COLS = [
    "系统负荷预测值",
    "风光总加预测值",
    "联络线预测值",
    "风电预测值",
    "光伏预测值",
    "水电预测值",
    "非市场化机组预测值",
]
ACTUAL_MARKER = "实际值"


@dataclass(frozen=True)
class DataAudit:
    train_feature_rows: int
    train_label_rows: int
    test_feature_rows: int
    supervised_rows: int
    missing_label_timestamps: int
    complete_label_days: int
    target_min: float
    target_max: float
    negative_target_rows: int
    negative_target_share: float
    first_train_time: str
    last_train_time: str
    first_test_time: str
    last_test_time: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _read_time_csv(path: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=usecols)
    if TIME_COL not in frame:
        raise ValueError(f"Missing required column {TIME_COL!r} in {path}")
    frame[TIME_COL] = pd.to_datetime(frame[TIME_COL], errors="raise")
    if frame[TIME_COL].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path}")
    return frame.sort_values(TIME_COL).reset_index(drop=True)


def load_raw_frames(config: PhaseBConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features = _read_time_csv(config.path("train_features"))
    labels = _read_time_csv(config.path("train_labels"))
    test_features = _read_time_csv(config.path("test_features"))
    assert_forecast_columns(train_features, context="training feature file")
    assert_forecast_columns(test_features, context="test feature file")
    if TARGET_COL not in labels:
        raise ValueError(f"Training label file is missing target {TARGET_COL!r}")
    return train_features, labels, test_features


def assert_forecast_columns(frame: pd.DataFrame, context: str = "model frame") -> None:
    missing = [column for column in FORECAST_COLS if column not in frame]
    if missing:
        raise ValueError(f"{context} is missing forecast columns: {missing}")


def assert_model_features_are_inference_safe(columns: Iterable[str]) -> None:
    forbidden_markers = (ACTUAL_MARKER, "实时价格", "target", "label")
    offenders = [
        column
        for column in columns
        if str(column) == TARGET_COL
        or any(marker in str(column).lower() for marker in forbidden_markers)
    ]
    if offenders:
        raise ValueError(f"Inference-unsafe feature columns detected: {offenders}")


def build_forecast_timeline(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate forecast-only train and test rows on one timestamp axis."""

    train = train_features[[TIME_COL, *FORECAST_COLS]].copy()
    train["dataset_split"] = "train"
    test = test_features[[TIME_COL, *FORECAST_COLS]].copy()
    test["dataset_split"] = "test"
    combined = pd.concat([train, test], ignore_index=True).sort_values(TIME_COL)
    if combined[TIME_COL].duplicated().any():
        duplicates = combined.loc[combined[TIME_COL].duplicated(False), TIME_COL].head().tolist()
        raise ValueError(f"Train/test timestamp overlap: {duplicates}")
    return combined.reset_index(drop=True)


def join_labels(feature_timeline: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Attach labels only after all causal features have been constructed."""

    return feature_timeline.merge(labels[[TIME_COL, TARGET_COL]], on=TIME_COL, how="left", validate="one_to_one")


def complete_day_index(frame: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DatetimeIndex:
    labeled = frame.loc[frame[target_col].notna(), [TIME_COL, target_col]].copy()
    labeled["date"] = labeled[TIME_COL].dt.normalize()
    counts = labeled.groupby("date", sort=True)[target_col].size()
    complete: list[pd.Timestamp] = []
    expected_offsets = pd.timedelta_range("0min", "23h45min", freq="15min")
    for date in counts[counts == 96].index:
        observed = pd.DatetimeIndex(labeled.loc[labeled["date"] == date, TIME_COL]) - date
        if pd.TimedeltaIndex(observed).equals(expected_offsets):
            complete.append(pd.Timestamp(date))
    return pd.DatetimeIndex(complete)


def supervised_complete_days(frame: pd.DataFrame) -> pd.DataFrame:
    complete = complete_day_index(frame)
    mask = frame[TARGET_COL].notna() & frame[TIME_COL].dt.normalize().isin(complete)
    return frame.loc[mask].sort_values(TIME_COL).reset_index(drop=True)


def audit_data(config: PhaseBConfig) -> DataAudit:
    train_features, labels, test_features = load_raw_frames(config)
    timeline = build_forecast_timeline(train_features, test_features)
    joined = join_labels(timeline, labels)
    train_joined = joined.loc[joined["dataset_split"] == "train"]
    complete = complete_day_index(train_joined)
    target = labels[TARGET_COL].astype(float)
    negative = int((target < 0).sum())
    return DataAudit(
        train_feature_rows=len(train_features),
        train_label_rows=len(labels),
        test_feature_rows=len(test_features),
        supervised_rows=int(train_joined[TARGET_COL].notna().sum()),
        missing_label_timestamps=int(train_joined[TARGET_COL].isna().sum()),
        complete_label_days=len(complete),
        target_min=float(target.min()),
        target_max=float(target.max()),
        negative_target_rows=negative,
        negative_target_share=float(negative / len(target)),
        first_train_time=str(train_features[TIME_COL].iloc[0]),
        last_train_time=str(train_features[TIME_COL].iloc[-1]),
        first_test_time=str(test_features[TIME_COL].iloc[0]),
        last_test_time=str(test_features[TIME_COL].iloc[-1]),
    )


def frame_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    """Stable SHA-256 fingerprint over selected columns and row order."""

    import hashlib

    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()
