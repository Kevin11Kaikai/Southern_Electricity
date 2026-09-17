"""Train-301 / score-59 holdout RMSE bakeoff.

Every candidate is fit on all 301 development complete days and ranked by
point RMSE on the last 59 holdout days. Nested CV and expanding 5-fold are
not used for model choice or reporting.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
TORCH_RUNTIME = ROOT / ".torch_runtime"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
if TORCH_RUNTIME.exists() and str(TORCH_RUNTIME) not in sys.path:
    sys.path.insert(0, str(TORCH_RUNTIME))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_b.advanced_models import make_ridge_pipeline, quarter_slot
from src.phase_b.config import load_config
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.data import TARGET_COL, TIME_COL, complete_day_index
from src.phase_b.experiments import prepare_feature_frame
from src.phase_b.features import FeatureSets
from src.phase_b.modeling import ModelSpec, deterministic_params, train_final_model
from src.phase_b.splits import make_day_split_plan, mask_for_dates

SEED = 42
HOLD_OUT_DAYS = 59
SEQUENCE_INNER_VAL_DAYS = 14
TREE_ROUNDS = 1500
RIDGE_ALPHA = 1.0
SEQUENCE_FEATURE_SET = "contextual"
REPORT_DIR = ROOT / "reports" / "holdout_rmse_v1"
LEADERBOARD_COLUMNS = ["model", "train_days", "test_days", "RMSE", "MAE", "n_points", "status"]


def _feature_columns(sets: FeatureSets, name: str) -> list[str]:
    columns = getattr(sets, name)
    return list(columns)


def _metrics(y_true: np.ndarray, yhat: np.ndarray) -> tuple[float, float, int]:
    y_true = np.asarray(y_true, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y_true.shape != yhat.shape:
        raise ValueError(f"prediction shape {yhat.shape} != label shape {y_true.shape}")
    rmse = float(mean_squared_error(y_true, yhat) ** 0.5)
    mae = float(mean_absolute_error(y_true, yhat))
    return rmse, mae, int(y_true.size)


def _ok_row(name: str, train_days: int, test_days: int, y_true: np.ndarray, yhat: np.ndarray) -> dict[str, Any]:
    rmse, mae, n_points = _metrics(y_true, yhat)
    return {
        "model": name,
        "train_days": int(train_days),
        "test_days": int(test_days),
        "RMSE": rmse,
        "MAE": mae,
        "n_points": n_points,
        "status": "ok",
    }


def _skip_row(name: str, train_days: int, test_days: int, status: str) -> dict[str, Any]:
    return {
        "model": name,
        "train_days": int(train_days),
        "test_days": int(test_days),
        "RMSE": np.nan,
        "MAE": np.nan,
        "n_points": np.nan,
        "status": status,
    }


def _lgb_spec(name: str, seed: int, n_jobs: int) -> ModelSpec:
    return ModelSpec(
        name=name,
        params=deterministic_params(seed=seed, n_jobs=n_jobs),
        num_boost_round=TREE_ROUNDS,
        early_stopping_rounds=0,
        target_mode="level",
    )


def predict_climatology(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    work = train[[TIME_COL, TARGET_COL]].copy()
    work["dow"] = work[TIME_COL].dt.dayofweek
    work["slot"] = quarter_slot(work[TIME_COL])
    means = work.groupby(["dow", "slot"], sort=True)[TARGET_COL].mean()
    global_mean = float(work[TARGET_COL].mean())
    keys = list(zip(test[TIME_COL].dt.dayofweek.to_numpy(), quarter_slot(test[TIME_COL])))
    return np.array([means.get(key, global_mean) for key in keys], dtype=float)


def predict_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    name: str,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    spec = _lgb_spec(name, seed, n_jobs)
    booster = train_final_model(train, feature_columns, spec, num_boost_round=TREE_ROUNDS)
    return np.asarray(booster.predict(test[feature_columns]), dtype=float)


def predict_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    alpha: float,
) -> np.ndarray:
    pipeline = make_ridge_pipeline(alpha=alpha)
    pipeline.fit(train[feature_columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(pipeline.predict(test[feature_columns]), dtype=float)


def predict_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=TREE_ROUNDS,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
        objective="reg:squarederror",
    )
    model.fit(train[feature_columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(model.predict(test[feature_columns]), dtype=float)


def predict_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
    n_jobs: int,
) -> np.ndarray:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=TREE_ROUNDS,
        learning_rate=0.03,
        depth=6,
        random_seed=seed,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        thread_count=n_jobs,
    )
    model.fit(train[feature_columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(model.predict(test[feature_columns]), dtype=float)


def predict_hist_gb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int,
) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=300,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(train[feature_columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(model.predict(test[feature_columns]), dtype=float)


def _frame_to_day_arrays(
    frame: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    ordered = frame.sort_values(TIME_COL)
    dates = ordered[TIME_COL].dt.normalize()
    x_days: list[np.ndarray] = []
    y_days: list[np.ndarray] = []
    day_index: list[pd.Timestamp] = []
    for date, group in ordered.groupby(dates, sort=True):
        if len(group) != STEPS_PER_DAY:
            raise ValueError(f"{date.date()} has {len(group)} rows, expected {STEPS_PER_DAY}")
        if feature_columns:
            x_days.append(group[feature_columns].to_numpy(dtype=np.float32))
        y_days.append(group[TARGET_COL].to_numpy(dtype=np.float32))
        day_index.append(pd.Timestamp(date))
    x_stack = np.stack(x_days) if x_days else np.zeros((len(y_days), STEPS_PER_DAY, 0), dtype=np.float32)
    return x_stack, np.stack(y_days), pd.DatetimeIndex(day_index)


def predict_sequence_family(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
    *,
    family: str,
    seed: int,
    inner_val_days: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from experiments.holdout_rmse_v1.sequence_models import (
        CompactDLinear,
        CompactDayLSTM,
        CompactDayTransformer,
        predict_sequence,
        train_sequence_model,
    )

    x_all, y_all, train_dates = _frame_to_day_arrays(train, feature_columns)
    x_test, _, test_dates = _frame_to_day_arrays(test, feature_columns)
    if inner_val_days <= 0 or inner_val_days >= len(train_dates):
        raise ValueError("inner_val_days must be a strict tail of the 301 development days")
    split = len(train_dates) - inner_val_days
    x_fit, y_fit = x_all[:split], y_all[:split]
    x_val, y_val = x_all[split:], y_all[split:]
    n_features = int(x_all.shape[-1])
    if family == "lstm":
        model = CompactDayLSTM(n_features)
    elif family == "transformer":
        model = CompactDayTransformer(n_features)
    elif family == "dlinear":
        model = CompactDLinear(n_features)
    else:
        raise ValueError(f"unknown sequence family: {family}")
    result = train_sequence_model(
        model,
        x_fit,
        y_fit,
        x_val,
        y_val,
        seed=seed,
    )
    yhat_days = predict_sequence(result, x_test)
    if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
        raise ValueError(f"{family} prediction shape {yhat_days.shape} does not match holdout days")
    return yhat_days.reshape(-1).astype(float), {
        "inner_val_days": inner_val_days,
        "fit_days": int(split),
        "best_inner_val_rmse": result.best_val_rmse,
        "epochs_run": result.epochs_run,
        "feature_set": SEQUENCE_FEATURE_SET,
        "n_features": n_features,
    }


def _climatology_day_matrix(train: pd.DataFrame) -> np.ndarray:
    work = train[[TIME_COL, TARGET_COL]].copy()
    work["dow"] = work[TIME_COL].dt.dayofweek.to_numpy()
    work["slot"] = quarter_slot(work[TIME_COL])
    means = work.groupby(["dow", "slot"], sort=True)[TARGET_COL].mean()
    global_mean = float(work[TARGET_COL].mean())
    table = np.full((7, STEPS_PER_DAY), global_mean, dtype=np.float32)
    for (dow, slot), value in means.items():
        table[int(dow), int(slot)] = np.float32(value)
    return table


def predict_dlinear_prevday(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    seed: int,
    inner_val_days: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map yesterday's actual 96-step price curve onto today's curve.

    Yesterday's settled price is available in a true day-ahead setting, including
    sequentially through the 59 holdout days. The first training day uses the
    weekday climatology curve as lookback.
    """
    from experiments.holdout_rmse_v1.sequence_models import (
        CompactDLinear,
        predict_sequence,
        train_sequence_model,
    )

    _, y_train, train_dates = _frame_to_day_arrays(train, [])
    _, y_test, test_dates = _frame_to_day_arrays(test, [])
    climatology = _climatology_day_matrix(train)
    train_lookback = np.concatenate([climatology[train_dates[0].dayofweek][None, :], y_train[:-1]], axis=0)
    x_all = train_lookback[:, :, None]
    y_all = y_train
    if inner_val_days <= 0 or inner_val_days >= len(train_dates):
        raise ValueError("inner_val_days must be a strict tail of the 301 development days")
    split = len(train_dates) - inner_val_days
    result = train_sequence_model(
        CompactDLinear(n_features=1),
        x_all[:split],
        y_all[:split],
        x_all[split:],
        y_all[split:],
        seed=seed,
    )
    test_lookback = np.concatenate([y_train[-1][None, :], y_test[:-1]], axis=0)[:, :, None]
    yhat_days = predict_sequence(result, test_lookback)
    if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
        raise ValueError(f"dlinear_prevday shape {yhat_days.shape} does not match holdout days")
    return yhat_days.reshape(-1).astype(float), {
        "inner_val_days": inner_val_days,
        "fit_days": int(split),
        "best_inner_val_rmse": result.best_val_rmse,
        "epochs_run": result.epochs_run,
        "feature_set": "prevday_actual_A",
        "n_features": 1,
        "lookback": "previous_complete_day_actual_price",
    }


def _run_named(
    name: str,
    train_days: int,
    test_days: int,
    y_true: np.ndarray,
    predict_fn: Callable[[], np.ndarray],
) -> dict[str, Any]:
    print(f"[holdout_rmse_v1] fitting {name}", flush=True)
    try:
        yhat = predict_fn()
        row = _ok_row(name, train_days, test_days, y_true, yhat)
        print(f"[holdout_rmse_v1] {name} RMSE={row['RMSE']:.4f} MAE={row['MAE']:.4f}", flush=True)
        return row
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        row = _skip_row(name, train_days, test_days, f"skipped: missing package ({missing})")
        print(f"[holdout_rmse_v1] {name} {row['status']}", flush=True)
        return row
    except Exception as exc:  # noqa: BLE001 — keep the table going
        detail = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        row = _skip_row(name, train_days, test_days, f"failed: {detail}")
        print(f"[holdout_rmse_v1] {name} {row['status']}", flush=True)
        return row


def build_leaderboard(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sets: FeatureSets,
    *,
    seed: int,
    n_jobs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y_true = test[TARGET_COL].astype(float).to_numpy()
    train_days = int(train[TIME_COL].dt.normalize().nunique())
    test_days = int(test[TIME_COL].dt.normalize().nunique())
    extras: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    rows.append(
        _run_named(
            "climatology_weekday_slot",
            train_days,
            test_days,
            y_true,
            lambda: predict_climatology(train, test),
        )
    )
    for feature_set in ("baseline", "contextual", "temporal", "full"):
        columns = _feature_columns(sets, feature_set)
        rows.append(
            _run_named(
                f"lgb_{feature_set}",
                train_days,
                test_days,
                y_true,
                lambda cols=columns, name=f"lgb_{feature_set}": predict_lightgbm(
                    train, test, cols, name=name, seed=seed, n_jobs=n_jobs
                ),
            )
        )
    for feature_set in ("contextual", "temporal"):
        columns = _feature_columns(sets, feature_set)
        rows.append(
            _run_named(
                f"ridge_{feature_set}_a1",
                train_days,
                test_days,
                y_true,
                lambda cols=columns: predict_ridge(train, test, cols, alpha=RIDGE_ALPHA),
            )
        )

    full_cols = _feature_columns(sets, "full")
    contextual_cols = _feature_columns(sets, "contextual")
    for feature_set, columns in (("contextual", contextual_cols), ("full", full_cols)):
        rows.append(
            _run_named(
                f"xgboost_{feature_set}",
                train_days,
                test_days,
                y_true,
                lambda cols=columns: predict_xgboost(train, test, cols, seed=seed, n_jobs=n_jobs),
            )
        )
        rows.append(
            _run_named(
                f"catboost_{feature_set}",
                train_days,
                test_days,
                y_true,
                lambda cols=columns: predict_catboost(train, test, cols, seed=seed, n_jobs=n_jobs),
            )
        )
    rows.append(
        _run_named(
            "hgb_full",
            train_days,
            test_days,
            y_true,
            lambda: predict_hist_gb(train, test, full_cols, seed=seed),
        )
    )

    sequence_cols = _feature_columns(sets, SEQUENCE_FEATURE_SET)
    for family in ("lstm", "transformer", "dlinear"):
        captured: dict[str, Any] = {}

        def _predict(family_name: str = family, extra: dict[str, Any] = captured) -> np.ndarray:
            yhat, info = predict_sequence_family(
                train,
                test,
                sequence_cols,
                family=family_name,
                seed=seed,
                inner_val_days=SEQUENCE_INNER_VAL_DAYS,
            )
            extra.update(info)
            return yhat

        row = _run_named(f"{family}_{SEQUENCE_FEATURE_SET}", train_days, test_days, y_true, _predict)
        rows.append(row)
        if captured:
            extras[family] = captured

    prevday_info: dict[str, Any] = {}

    def _predict_prevday(extra: dict[str, Any] = prevday_info) -> np.ndarray:
        yhat, info = predict_dlinear_prevday(
            train,
            test,
            seed=seed,
            inner_val_days=SEQUENCE_INNER_VAL_DAYS,
        )
        extra.update(info)
        return yhat

    rows.append(_run_named("dlinear_prevday", train_days, test_days, y_true, _predict_prevday))
    if prevday_info:
        extras["dlinear_prevday"] = prevday_info
    return rows, extras


def write_outputs(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)
    ranked = leaderboard.copy()
    ranked["_rank"] = ranked["RMSE"].rank(method="min", na_option="bottom")
    ranked = ranked.sort_values(["_rank", "model"], kind="mergesort").drop(columns="_rank")
    csv_path = output_dir / "leaderboard.csv"
    json_path = output_dir / "manifest.json"
    display = ranked.copy()
    for column in ("RMSE", "MAE"):
        display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    display["n_points"] = display["n_points"].map(lambda value: "" if pd.isna(value) else str(int(value)))
    display.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path, json_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="301/59 holdout RMSE bakeoff")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase_b.toml"))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.seed)
    n_jobs = int(config.section("model")["n_jobs"])
    print("[holdout_rmse_v1] preparing feature frame", flush=True)
    feature_frame, sets, audit = prepare_feature_frame(config)
    print(
        f"[holdout_rmse_v1] feature rows={len(feature_frame)} "
        f"sets={{baseline:{len(sets.baseline)}, contextual:{len(sets.contextual)}, "
        f"temporal:{len(sets.temporal)}, full:{len(sets.full)}}}",
        flush=True,
    )
    labeled = feature_frame.loc[feature_frame[TARGET_COL].notna()].copy()
    complete = complete_day_index(labeled)
    plan = make_day_split_plan(
        complete,
        n_splits=int(config.section("validation")["n_splits"]),
        validation_days=int(config.section("validation")["validation_days"]),
        holdout_days=HOLD_OUT_DAYS,
    )
    # Folds exist on the plan object but are ignored: ranking is 59-day RMSE only.
    train_mask = mask_for_dates(labeled[TIME_COL], plan.development_dates)
    test_mask = mask_for_dates(labeled[TIME_COL], plan.holdout_dates)
    train = labeled.loc[train_mask].sort_values(TIME_COL).reset_index(drop=True)
    test = labeled.loc[test_mask].sort_values(TIME_COL).reset_index(drop=True)
    train_days = int(plan.development_dates.size)
    test_days = int(plan.holdout_dates.size)
    if train_days != 301 or test_days != 59:
        raise AssertionError(
            f"expected 301/59 split, got {train_days}/{test_days} complete days"
        )

    rows, sequence_info = build_leaderboard(train, test, sets, seed=seed, n_jobs=n_jobs)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    winner = min(ok_rows, key=lambda row: row["RMSE"]) if ok_rows else None
    manifest = {
        "experiment": "holdout_rmse_v1",
        "protocol": "train_all_301_complete_days_then_rmse_on_last_59_holdout_days",
        "nested_cv": False,
        "expanding_5fold": False,
        "early_stopping_used_for_ranking": False,
        "seed": seed,
        "target": TARGET_COL,
        "metric": "RMSE",
        "secondary_metric": "MAE",
        "train_days": train_days,
        "test_days": test_days,
        "train_start": str(plan.development_dates[0].date()),
        "train_end": str(plan.development_dates[-1].date()),
        "holdout_start": str(plan.holdout_dates[0].date()),
        "holdout_end": str(plan.holdout_dates[-1].date()),
        "n_train_points": int(len(train)),
        "n_holdout_points": int(len(test)),
        "tree_num_boost_round": TREE_ROUNDS,
        "tree_early_stopping": False,
        "ridge_alpha": RIDGE_ALPHA,
        "sequence_feature_set": SEQUENCE_FEATURE_SET,
        "sequence_inner_val_days": SEQUENCE_INNER_VAL_DAYS,
        "sequence_inner_val_role": "training-time early stop only; never used for ranking",
        "folds_ignored": True,
        "dispatch_not_in_this_run": True,
        "feature_set_sizes": {name: len(getattr(sets, name)) for name in ("baseline", "contextual", "temporal", "full")},
        "feature_frame_rows": audit.get("feature_frame_rows"),
        "sequence": sequence_info,
        "winner": None if winner is None else {"model": winner["model"], "RMSE": winner["RMSE"], "MAE": winner["MAE"]},
        "n_models": len(rows),
        "n_ok": len(ok_rows),
    }
    csv_path, json_path = write_outputs(rows, manifest, Path(args.output_dir))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(pd.read_csv(csv_path).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
