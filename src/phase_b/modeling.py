"""Deterministic LightGBM training, persistence, and fold prediction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .data import TARGET_COL, TIME_COL, assert_model_features_are_inference_safe
from .splits import DayFold, mask_for_dates


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, Any]
    num_boost_round: int = 1500
    early_stopping_rounds: int = 100
    target_mode: str = "level"


@dataclass(frozen=True)
class FoldModelResult:
    fold: int
    best_iteration: int
    rmse: float
    mae: float
    train_rows: int
    validation_rows: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def deterministic_params(seed: int = 42, n_jobs: int = 4, **overrides: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": n_jobs,
    }
    params.update(overrides)
    return params


def _target_values(frame: pd.DataFrame, mode: str) -> np.ndarray:
    target = frame[TARGET_COL].astype(float)
    if mode == "level":
        return target.to_numpy()
    if mode == "day_centered":
        day = frame[TIME_COL].dt.normalize()
        return (target - target.groupby(day).transform("mean")).to_numpy()
    raise ValueError(f"Unknown target mode: {mode}")


def train_fold_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fold: DayFold,
    spec: ModelSpec,
) -> tuple[lgb.Booster, np.ndarray, FoldModelResult]:
    assert_model_features_are_inference_safe(feature_columns)
    train_mask = mask_for_dates(frame[TIME_COL], fold.train_dates)
    validation_mask = mask_for_dates(frame[TIME_COL], fold.validation_dates)
    train = frame.loc[train_mask]
    validation = frame.loc[validation_mask]
    x_train = train[feature_columns]
    x_validation = validation[feature_columns]
    y_train = _target_values(train, spec.target_mode)
    y_validation = _target_values(validation, spec.target_mode)
    train_set = lgb.Dataset(x_train, label=y_train, feature_name=feature_columns, free_raw_data=False)
    validation_set = lgb.Dataset(x_validation, label=y_validation, feature_name=feature_columns, reference=train_set, free_raw_data=False)
    callbacks = [lgb.early_stopping(spec.early_stopping_rounds, verbose=False)]
    booster = lgb.train(
        spec.params,
        train_set,
        num_boost_round=spec.num_boost_round,
        valid_sets=[validation_set],
        valid_names=["validation"],
        callbacks=callbacks,
    )
    prediction = booster.predict(x_validation, num_iteration=booster.best_iteration)
    result = FoldModelResult(
        fold=fold.fold,
        best_iteration=int(booster.best_iteration),
        rmse=float(mean_squared_error(y_validation, prediction) ** 0.5),
        mae=float(mean_absolute_error(y_validation, prediction)),
        train_rows=len(train),
        validation_rows=len(validation),
        train_start=str(train[TIME_COL].iloc[0]),
        train_end=str(train[TIME_COL].iloc[-1]),
        validation_start=str(validation[TIME_COL].iloc[0]),
        validation_end=str(validation[TIME_COL].iloc[-1]),
    )
    return booster, np.asarray(prediction), result


def train_final_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: ModelSpec,
    *,
    num_boost_round: int,
) -> lgb.Booster:
    assert_model_features_are_inference_safe(feature_columns)
    train_set = lgb.Dataset(
        frame[feature_columns],
        label=_target_values(frame, spec.target_mode),
        feature_name=feature_columns,
        free_raw_data=False,
    )
    return lgb.train(spec.params, train_set, num_boost_round=int(num_boost_round))


def save_model_bundle(
    booster: lgb.Booster,
    output_dir: str | Path,
    *,
    feature_columns: list[str],
    spec: ModelSpec,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / f"{spec.name}.txt"
    manifest_path = root / f"{spec.name}.manifest.json"
    booster.save_model(model_path)
    manifest = {
        "model_name": spec.name,
        "feature_columns": feature_columns,
        "spec": {"name": spec.name, "params": spec.params, "num_boost_round": spec.num_boost_round, "early_stopping_rounds": spec.early_stopping_rounds, "target_mode": spec.target_mode},
        **metadata,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, manifest_path


def load_model_bundle(model_path: str | Path, manifest_path: str | Path) -> tuple[lgb.Booster, dict[str, Any]]:
    booster = lgb.Booster(model_file=str(model_path))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return booster, manifest

