"""271/30/59 holdout RMSE v2.

Select on the last 30 development days, retrain on all 301, score the last 59
once. The 59-day window is not used for early stopping or model choice.
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

from experiments.holdout_rmse_v1.run import (  # noqa: E402
    _climatology_day_matrix,
    _frame_to_day_arrays,
    predict_climatology,
    predict_lightgbm,
    predict_ridge,
)
from src.phase_b.advanced_models import make_ridge_pipeline  # noqa: E402
from src.phase_b.config import load_config  # noqa: E402
from src.phase_b.contracts import STEPS_PER_DAY  # noqa: E402
from src.phase_b.data import TARGET_COL, TIME_COL, complete_day_index  # noqa: E402
from src.phase_b.experiments import prepare_feature_frame  # noqa: E402
from src.phase_b.features import FeatureSets  # noqa: E402
from src.phase_b.modeling import ModelSpec, deterministic_params, train_final_model, train_fold_model  # noqa: E402
from src.phase_b.splits import DayFold, make_day_split_plan, mask_for_dates  # noqa: E402

SEED = 42
HOLD_OUT_DAYS = 59
VAL_DAYS = 30
DEV_DAYS = 301
FIT_DAYS = DEV_DAYS - VAL_DAYS
TREE_ROUND_CAP = 1500
TREE_ES_ROUNDS = 100
SEQ_MAX_EPOCHS = 80
SEQ_PATIENCE = 12
SEQ_EPOCH_CAP = 80
V1_LEADERBOARD = ROOT / "reports" / "holdout_rmse_v1" / "leaderboard.csv"
REPORT_DIR = ROOT / "reports" / "holdout_rmse_v2"
LEAVES_GRID = (31, 63, 127)
LR_GRID = (0.03, 0.05)
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
LEADERBOARD_COLUMNS = [
    "model",
    "train_days",
    "test_days",
    "RMSE",
    "MAE",
    "n_points",
    "val30_RMSE",
    "best_rounds_or_epochs",
    "batch",
    "status",
]


def _feature_columns(sets: FeatureSets, name: str) -> list[str]:
    return list(getattr(sets, name))


def _metrics(y_true: np.ndarray, yhat: np.ndarray) -> tuple[float, float, int]:
    y_true = np.asarray(y_true, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    if y_true.shape != yhat.shape:
        raise ValueError(f"prediction shape {yhat.shape} != label shape {y_true.shape}")
    return float(mean_squared_error(y_true, yhat) ** 0.5), float(mean_absolute_error(y_true, yhat)), int(y_true.size)


def _rmse(y_true: np.ndarray, yhat: np.ndarray) -> float:
    return float(mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(yhat, dtype=float)) ** 0.5)


def scaled_count(best: int, *, cap: int) -> int:
    return min(int(cap), max(1, int(round(int(best) * DEV_DAYS / FIT_DAYS))))


def _ok_row(
    name: str,
    y_true: np.ndarray,
    yhat: np.ndarray,
    *,
    batch: str,
    val30_rmse: float | None = None,
    best_rounds_or_epochs: int | None = None,
) -> dict[str, Any]:
    rmse, mae, n_points = _metrics(y_true, yhat)
    return {
        "model": name,
        "train_days": DEV_DAYS,
        "test_days": HOLD_OUT_DAYS,
        "RMSE": rmse,
        "MAE": mae,
        "n_points": n_points,
        "val30_RMSE": np.nan if val30_rmse is None else float(val30_rmse),
        "best_rounds_or_epochs": np.nan if best_rounds_or_epochs is None else int(best_rounds_or_epochs),
        "batch": batch,
        "status": "ok",
    }


def _skip_row(name: str, status: str, *, batch: str) -> dict[str, Any]:
    return {
        "model": name,
        "train_days": DEV_DAYS,
        "test_days": HOLD_OUT_DAYS,
        "RMSE": np.nan,
        "MAE": np.nan,
        "n_points": np.nan,
        "val30_RMSE": np.nan,
        "best_rounds_or_epochs": np.nan,
        "batch": batch,
        "status": status,
    }


def _run_named(name: str, batch: str, predict_fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    print(f"[holdout_rmse_v2] fitting {name}", flush=True)
    try:
        row = predict_fn()
        extra = ""
        if pd.notna(row.get("val30_RMSE")):
            extra += f" val30={float(row['val30_RMSE']):.4f}"
        if pd.notna(row.get("best_rounds_or_epochs")):
            extra += f" rounds/epochs={int(row['best_rounds_or_epochs'])}"
        print(f"[holdout_rmse_v2] {name} RMSE={row['RMSE']:.4f} MAE={row['MAE']:.4f}{extra}", flush=True)
        return row
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        row = _skip_row(name, f"skipped: missing package ({missing})", batch=batch)
        print(f"[holdout_rmse_v2] {name} {row['status']}", flush=True)
        return row
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        row = _skip_row(name, f"failed: {type(exc).__name__}: {exc}", batch=batch)
        print(f"[holdout_rmse_v2] {name} {row['status']}", flush=True)
        return row


def select_lightgbm(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    fold: DayFold,
    feature_columns: list[str],
    y_true: np.ndarray,
    *,
    name: str,
    batch: str,
    seed: int,
    n_jobs: int,
    grid: bool,
) -> dict[str, Any]:
    leaves_grid = LEAVES_GRID if grid else (63,)
    lr_grid = LR_GRID if grid else (0.03,)
    best: dict[str, Any] | None = None
    for num_leaves in leaves_grid:
        for learning_rate in lr_grid:
            spec = ModelSpec(
                name=f"{name}_l{num_leaves}_lr{learning_rate}",
                params=deterministic_params(
                    seed=seed,
                    n_jobs=n_jobs,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                ),
                num_boost_round=TREE_ROUND_CAP,
                early_stopping_rounds=TREE_ES_ROUNDS,
                target_mode="level",
            )
            booster, val_pred, result = train_fold_model(development, feature_columns, fold, spec)
            candidate = {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "best_iteration": int(result.best_iteration),
                "val_rmse": float(result.rmse),
                "val_pred": np.asarray(val_pred, dtype=float),
                "spec": spec,
            }
            print(
                f"[holdout_rmse_v2]   {name} leaves={num_leaves} lr={learning_rate} "
                f"best_iter={result.best_iteration} val30={result.rmse:.4f}",
                flush=True,
            )
            if best is None or candidate["val_rmse"] < best["val_rmse"]:
                best = candidate
    assert best is not None
    final_rounds = scaled_count(best["best_iteration"], cap=TREE_ROUND_CAP)
    final = train_final_model(development, feature_columns, best["spec"], num_boost_round=final_rounds)
    yhat = np.asarray(final.predict(holdout[feature_columns]), dtype=float)
    row = _ok_row(
        name,
        y_true,
        yhat,
        batch=batch,
        val30_rmse=best["val_rmse"],
        best_rounds_or_epochs=final_rounds,
    )
    row["_val_pred"] = best["val_pred"]
    row["_holdout_pred"] = yhat
    row["_selected"] = {
        "num_leaves": best["num_leaves"],
        "learning_rate": best["learning_rate"],
        "best_iteration": best["best_iteration"],
        "final_rounds": final_rounds,
    }
    return row


def select_xgboost(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: list[str],
    y_true: np.ndarray,
    *,
    name: str,
    batch: str,
    seed: int,
    n_jobs: int,
    grid: bool,
) -> dict[str, Any]:
    import xgboost as xgb

    leaves_grid = LEAVES_GRID if grid else (63,)
    lr_grid = LR_GRID if grid else (0.03,)
    y_fit = fit[TARGET_COL].astype(float).to_numpy()
    y_val = val[TARGET_COL].astype(float).to_numpy()
    best: dict[str, Any] | None = None
    for num_leaves in leaves_grid:
        for learning_rate in lr_grid:
            max_depth = 5 if num_leaves <= 31 else 6 if num_leaves <= 63 else 7
            model = xgb.XGBRegressor(
                n_estimators=TREE_ROUND_CAP,
                learning_rate=learning_rate,
                max_depth=max_depth,
                max_leaves=num_leaves,
                min_child_weight=20,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=n_jobs,
                tree_method="hist",
                objective="reg:squarederror",
                early_stopping_rounds=TREE_ES_ROUNDS,
            )
            model.fit(
                fit[feature_columns],
                y_fit,
                eval_set=[(val[feature_columns], y_val)],
                verbose=False,
            )
            best_iteration = int(getattr(model, "best_iteration", TREE_ROUND_CAP - 1)) + 1
            val_pred = np.asarray(model.predict(val[feature_columns]), dtype=float)
            val_rmse = _rmse(y_val, val_pred)
            print(
                f"[holdout_rmse_v2]   {name} leaves={num_leaves} lr={learning_rate} "
                f"best_iter={best_iteration} val30={val_rmse:.4f}",
                flush=True,
            )
            candidate = {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "best_iteration": best_iteration,
                "val_rmse": val_rmse,
            }
            if best is None or val_rmse < best["val_rmse"]:
                best = candidate
    assert best is not None
    final_rounds = scaled_count(best["best_iteration"], cap=TREE_ROUND_CAP)
    final = xgb.XGBRegressor(
        n_estimators=final_rounds,
        learning_rate=best["learning_rate"],
        max_depth=best["max_depth"],
        max_leaves=best["num_leaves"],
        min_child_weight=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
        objective="reg:squarederror",
    )
    final.fit(development[feature_columns], development[TARGET_COL].astype(float).to_numpy())
    yhat = np.asarray(final.predict(holdout[feature_columns]), dtype=float)
    return _ok_row(
        name,
        y_true,
        yhat,
        batch=batch,
        val30_rmse=best["val_rmse"],
        best_rounds_or_epochs=final_rounds,
    )


def select_catboost(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: list[str],
    y_true: np.ndarray,
    *,
    name: str,
    batch: str,
    seed: int,
    n_jobs: int,
    grid: bool,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    leaves_grid = LEAVES_GRID if grid else (63,)
    lr_grid = LR_GRID if grid else (0.03,)
    y_fit = fit[TARGET_COL].astype(float).to_numpy()
    y_val = val[TARGET_COL].astype(float).to_numpy()
    best: dict[str, Any] | None = None
    for num_leaves in leaves_grid:
        for learning_rate in lr_grid:
            depth = 5 if num_leaves <= 31 else 6 if num_leaves <= 63 else 7
            model = CatBoostRegressor(
                iterations=TREE_ROUND_CAP,
                learning_rate=learning_rate,
                depth=depth,
                random_seed=seed,
                loss_function="RMSE",
                verbose=False,
                allow_writing_files=False,
                thread_count=n_jobs,
                od_type="Iter",
                od_wait=TREE_ES_ROUNDS,
                use_best_model=True,
            )
            model.fit(fit[feature_columns], y_fit, eval_set=(val[feature_columns], y_val))
            best_iteration = int(model.get_best_iteration()) + 1
            val_pred = np.asarray(model.predict(val[feature_columns]), dtype=float)
            val_rmse = _rmse(y_val, val_pred)
            print(
                f"[holdout_rmse_v2]   {name} depth={depth} lr={learning_rate} "
                f"best_iter={best_iteration} val30={val_rmse:.4f}",
                flush=True,
            )
            candidate = {
                "depth": depth,
                "learning_rate": learning_rate,
                "best_iteration": best_iteration,
                "val_rmse": val_rmse,
            }
            if best is None or val_rmse < best["val_rmse"]:
                best = candidate
    assert best is not None
    final_rounds = scaled_count(best["best_iteration"], cap=TREE_ROUND_CAP)
    final = CatBoostRegressor(
        iterations=final_rounds,
        learning_rate=best["learning_rate"],
        depth=best["depth"],
        random_seed=seed,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        thread_count=n_jobs,
    )
    final.fit(development[feature_columns], development[TARGET_COL].astype(float).to_numpy())
    yhat = np.asarray(final.predict(holdout[feature_columns]), dtype=float)
    return _ok_row(
        name,
        y_true,
        yhat,
        batch=batch,
        val30_rmse=best["val_rmse"],
        best_rounds_or_epochs=final_rounds,
    )


def select_hgb(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: list[str],
    y_true: np.ndarray,
    *,
    name: str,
    batch: str,
    seed: int,
) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    y_fit = fit[TARGET_COL].astype(float).to_numpy()
    y_val = val[TARGET_COL].astype(float).to_numpy()
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=20,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=False,
        warm_start=True,
        random_state=seed,
    )
    best_rmse = float("inf")
    best_iter = 20
    wait = 0
    for n_iter in range(20, 301, 20):
        model.max_iter = n_iter
        model.fit(fit[feature_columns], y_fit)
        val_rmse = _rmse(y_val, model.predict(val[feature_columns]))
        if val_rmse < best_rmse - 1e-6:
            best_rmse = val_rmse
            best_iter = n_iter
            wait = 0
        else:
            wait += 1
            if wait >= 3:
                break
    final_iter = scaled_count(best_iter, cap=300)
    final = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_iter=final_iter,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=seed,
    )
    final.fit(development[feature_columns], development[TARGET_COL].astype(float).to_numpy())
    yhat = np.asarray(final.predict(holdout[feature_columns]), dtype=float)
    return _ok_row(
        name,
        y_true,
        yhat,
        batch=batch,
        val30_rmse=best_rmse,
        best_rounds_or_epochs=final_iter,
    )


def select_ridge(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: list[str],
    y_true: np.ndarray,
    *,
    name: str,
    batch: str,
) -> dict[str, Any]:
    y_val = val[TARGET_COL].astype(float).to_numpy()
    best_alpha = RIDGE_ALPHAS[0]
    best_rmse = float("inf")
    for alpha in RIDGE_ALPHAS:
        pipeline = make_ridge_pipeline(alpha=alpha)
        pipeline.fit(fit[feature_columns], fit[TARGET_COL].astype(float).to_numpy())
        val_rmse = _rmse(y_val, pipeline.predict(val[feature_columns]))
        print(f"[holdout_rmse_v2]   {name} alpha={alpha:g} val30={val_rmse:.4f}", flush=True)
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = alpha
    yhat = predict_ridge(development, holdout, feature_columns, alpha=best_alpha)
    row = _ok_row(name, y_true, yhat, batch=batch, val30_rmse=best_rmse, best_rounds_or_epochs=None)
    row["_selected"] = {"alpha": best_alpha}
    return row


def _sequence_ctor(family: str, n_features: int):
    from experiments.holdout_rmse_v1.sequence_models import CompactDLinear, CompactDayLSTM, CompactDayTransformer

    if family == "lstm":
        return lambda: CompactDayLSTM(n_features)
    if family == "transformer":
        return lambda: CompactDayTransformer(n_features)
    if family == "dlinear":
        return lambda: CompactDLinear(n_features)
    raise ValueError(f"unknown sequence family: {family}")


def select_sequence(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: list[str],
    y_true: np.ndarray,
    *,
    name: str,
    family: str,
    batch: str,
    seed: int,
) -> dict[str, Any]:
    from experiments.holdout_rmse_v1.sequence_models import predict_sequence, train_sequence_fixed_epochs, train_sequence_model

    x_dev, y_dev, _dates = _frame_to_day_arrays(development, feature_columns)
    x_test, _, test_dates = _frame_to_day_arrays(holdout, feature_columns)
    split = FIT_DAYS
    x_fit, y_fit = x_dev[:split], y_dev[:split]
    x_val, y_val = x_dev[split:], y_dev[split:]
    n_features = int(x_dev.shape[-1])
    ctor = _sequence_ctor(family, n_features)
    selected = train_sequence_model(
        ctor(),
        x_fit,
        y_fit,
        x_val,
        y_val,
        seed=seed,
        max_epochs=SEQ_MAX_EPOCHS,
        patience=SEQ_PATIENCE,
    )
    val_pred = predict_sequence(selected, x_val).reshape(-1)
    val_rmse = _rmse(y_val.reshape(-1), val_pred)
    final_epochs = scaled_count(selected.best_epoch, cap=SEQ_EPOCH_CAP)
    retrained = train_sequence_fixed_epochs(ctor(), x_dev, y_dev, num_epochs=final_epochs, seed=seed)
    yhat_days = predict_sequence(retrained, x_test)
    if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
        raise ValueError(f"{family} prediction shape {yhat_days.shape} does not match holdout days")
    yhat = yhat_days.reshape(-1).astype(float)
    row = _ok_row(
        name,
        y_true,
        yhat,
        batch=batch,
        val30_rmse=val_rmse,
        best_rounds_or_epochs=final_epochs,
    )
    row["_val_pred"] = val_pred
    row["_holdout_pred"] = yhat
    row["_selected"] = {
        "best_epoch": selected.best_epoch,
        "epochs_run": selected.epochs_run,
        "final_epochs": final_epochs,
        "select_val_rmse": selected.best_val_rmse,
    }
    return row


def select_dlinear_prevday(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    y_true: np.ndarray,
    *,
    seed: int,
    batch: str,
) -> dict[str, Any]:
    from experiments.holdout_rmse_v1.sequence_models import (
        CompactDLinear,
        predict_sequence,
        train_sequence_fixed_epochs,
        train_sequence_model,
    )

    _, y_dev, dev_dates = _frame_to_day_arrays(development, [])
    _, y_test, test_dates = _frame_to_day_arrays(holdout, [])
    climatology = _climatology_day_matrix(development)
    lookback = np.concatenate([climatology[dev_dates[0].dayofweek][None, :], y_dev[:-1]], axis=0)[:, :, None]
    split = FIT_DAYS
    selected = train_sequence_model(
        CompactDLinear(n_features=1),
        lookback[:split],
        y_dev[:split],
        lookback[split:],
        y_dev[split:],
        seed=seed,
        max_epochs=SEQ_MAX_EPOCHS,
        patience=SEQ_PATIENCE,
    )
    val_pred = predict_sequence(selected, lookback[split:]).reshape(-1)
    val_rmse = _rmse(y_dev[split:].reshape(-1), val_pred)
    final_epochs = scaled_count(selected.best_epoch, cap=SEQ_EPOCH_CAP)
    retrained = train_sequence_fixed_epochs(
        CompactDLinear(n_features=1),
        lookback,
        y_dev,
        num_epochs=final_epochs,
        seed=seed,
    )
    test_lookback = np.concatenate([y_dev[-1][None, :], y_test[:-1]], axis=0)[:, :, None]
    yhat_days = predict_sequence(retrained, test_lookback)
    if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
        raise ValueError(f"dlinear_prevday shape {yhat_days.shape} does not match holdout days")
    return _ok_row(
        "dlinear_prevday",
        y_true,
        yhat_days.reshape(-1).astype(float),
        batch=batch,
        val30_rmse=val_rmse,
        best_rounds_or_epochs=final_epochs,
    )


def write_leaderboard(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    display_rows = [{key: row.get(key) for key in LEADERBOARD_COLUMNS} for row in rows]
    frame = pd.DataFrame(display_rows, columns=LEADERBOARD_COLUMNS)
    ranked = frame.copy()
    ranked["_rank"] = ranked["RMSE"].rank(method="min", na_option="bottom")
    ranked = ranked.sort_values(["_rank", "model"], kind="mergesort").drop(columns="_rank")
    out = ranked.copy()
    for column in ("RMSE", "MAE", "val30_RMSE"):
        out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    for column in ("n_points", "best_rounds_or_epochs"):
        out[column] = out[column].map(lambda value: "" if pd.isna(value) else str(int(value)))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return ranked


def write_comparison(v2_rows: list[dict[str, Any]], path: Path) -> None:
    v1 = pd.read_csv(V1_LEADERBOARD)
    v2 = pd.DataFrame([{key: row.get(key) for key in LEADERBOARD_COLUMNS} for row in v2_rows])
    merged = v1[["model", "RMSE"]].rename(columns={"RMSE": "v1_RMSE"}).merge(
        v2[["model", "RMSE", "batch", "status"]].rename(columns={"RMSE": "v2_RMSE"}),
        on="model",
        how="outer",
    )
    merged["delta"] = merged["v2_RMSE"] - merged["v1_RMSE"]
    merged = merged.sort_values(["delta", "model"], na_position="last", kind="mergesort")
    out = merged.copy()
    for column in ("v1_RMSE", "v2_RMSE", "delta"):
        out[column] = out[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.6f}")
    path.write_text(out.to_csv(index=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="271/30/59 holdout RMSE v2")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase_b.toml"))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)

    config = load_config(args.config)
    seed = int(config.seed)
    n_jobs = int(config.section("model")["n_jobs"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[holdout_rmse_v2] preparing feature frame", flush=True)
    feature_frame, sets, audit = prepare_feature_frame(config)
    labeled = feature_frame.loc[feature_frame[TARGET_COL].notna()].copy()
    complete = complete_day_index(labeled)
    plan = make_day_split_plan(
        complete,
        n_splits=int(config.section("validation")["n_splits"]),
        validation_days=int(config.section("validation")["validation_days"]),
        holdout_days=HOLD_OUT_DAYS,
    )
    development_dates = plan.development_dates
    if int(development_dates.size) != DEV_DAYS:
        raise AssertionError(f"expected {DEV_DAYS} development days, got {development_dates.size}")
    fit_dates = development_dates[:FIT_DAYS]
    val_dates = development_dates[-VAL_DAYS:]
    if int(fit_dates.size) != FIT_DAYS or int(val_dates.size) != VAL_DAYS:
        raise AssertionError("expected 271/30 inside development")
    fold = DayFold(fold=0, train_dates=fit_dates, validation_dates=val_dates)

    development = labeled.loc[mask_for_dates(labeled[TIME_COL], development_dates)].sort_values(TIME_COL).reset_index(drop=True)
    fit = labeled.loc[mask_for_dates(labeled[TIME_COL], fit_dates)].sort_values(TIME_COL).reset_index(drop=True)
    val = labeled.loc[mask_for_dates(labeled[TIME_COL], val_dates)].sort_values(TIME_COL).reset_index(drop=True)
    holdout = labeled.loc[mask_for_dates(labeled[TIME_COL], plan.holdout_dates)].sort_values(TIME_COL).reset_index(drop=True)
    y_true = holdout[TARGET_COL].astype(float).to_numpy()
    y_val = val[TARGET_COL].astype(float).to_numpy()
    extras: dict[str, Any] = {
        "fit_start": str(fit_dates[0].date()),
        "fit_end": str(fit_dates[-1].date()),
        "val_start": str(val_dates[0].date()),
        "val_end": str(val_dates[-1].date()),
        "holdout_start": str(plan.holdout_dates[0].date()),
        "holdout_end": str(plan.holdout_dates[-1].date()),
    }

    rows: list[dict[str, Any]] = []
    print("[holdout_rmse_v2] batch A", flush=True)
    rows.append(
        _run_named(
            "climatology_weekday_slot",
            "A",
            lambda: _ok_row(
                "climatology_weekday_slot",
                y_true,
                predict_climatology(development, holdout),
                batch="A",
            ),
        )
    )
    rows.append(
        _run_named(
            "lgb_baseline",
            "A",
            lambda: _ok_row(
                "lgb_baseline",
                y_true,
                predict_lightgbm(
                    development,
                    holdout,
                    _feature_columns(sets, "baseline"),
                    name="lgb_baseline",
                    seed=seed,
                    n_jobs=n_jobs,
                ),
                batch="A",
            ),
        )
    )

    contextual = _feature_columns(sets, "contextual")
    lgb_row = _run_named(
        "lgb_contextual",
        "A",
        lambda: select_lightgbm(
            development,
            holdout,
            fold,
            contextual,
            y_true,
            name="lgb_contextual",
            batch="A",
            seed=seed,
            n_jobs=n_jobs,
            grid=True,
        ),
    )
    rows.append(lgb_row)
    rows.append(
        _run_named(
            "catboost_contextual",
            "A",
            lambda: select_catboost(
                fit, val, development, holdout, contextual, y_true,
                name="catboost_contextual", batch="A", seed=seed, n_jobs=n_jobs, grid=True,
            ),
        )
    )
    rows.append(
        _run_named(
            "xgboost_contextual",
            "A",
            lambda: select_xgboost(
                fit, val, development, holdout, contextual, y_true,
                name="xgboost_contextual", batch="A", seed=seed, n_jobs=n_jobs, grid=True,
            ),
        )
    )
    transformer_row = _run_named(
        "transformer_contextual",
        "A",
        lambda: select_sequence(
            development,
            holdout,
            contextual,
            y_true,
            name="transformer_contextual",
            family="transformer",
            batch="A",
            seed=seed,
        ),
    )
    rows.append(transformer_row)

    def _blend() -> dict[str, Any]:
        if lgb_row.get("status") != "ok" or transformer_row.get("status") != "ok":
            raise RuntimeError("blend needs both lgb_contextual and transformer_contextual")
        lgb_val = np.asarray(lgb_row["_val_pred"], dtype=float)
        tr_val = np.asarray(transformer_row["_val_pred"], dtype=float)
        if lgb_val.shape != tr_val.shape or lgb_val.shape[0] != y_val.shape[0]:
            raise ValueError("blend validation prediction shapes do not match")
        best_w = 0.0
        best_rmse = float("inf")
        for weight in BLEND_WEIGHTS:
            mixed = weight * tr_val + (1.0 - weight) * lgb_val
            val_rmse = _rmse(y_val, mixed)
            print(f"[holdout_rmse_v2]   blend w_transformer={weight:g} val30={val_rmse:.4f}", flush=True)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_w = weight
        holdout_hat = best_w * np.asarray(transformer_row["_holdout_pred"], dtype=float) + (
            1.0 - best_w
        ) * np.asarray(lgb_row["_holdout_pred"], dtype=float)
        row = _ok_row(
            "blend_lgb_transformer",
            y_true,
            holdout_hat,
            batch="A",
            val30_rmse=best_rmse,
            best_rounds_or_epochs=None,
        )
        row["_selected"] = {"w_transformer": best_w, "w_lgb": 1.0 - best_w}
        extras["blend"] = row["_selected"]
        return row

    rows.append(_run_named("blend_lgb_transformer", "A", _blend))
    write_leaderboard(rows, output_dir / "leaderboard.csv")

    print("[holdout_rmse_v2] batch B", flush=True)
    temporal = _feature_columns(sets, "temporal")
    full = _feature_columns(sets, "full")
    for feature_set, columns in (("temporal", temporal), ("full", full)):
        rows.append(
            _run_named(
                f"lgb_{feature_set}",
                "B",
                lambda cols=columns, label=f"lgb_{feature_set}": select_lightgbm(
                    development, holdout, fold, cols, y_true,
                    name=label, batch="B", seed=seed, n_jobs=n_jobs, grid=False,
                ),
            )
        )
        rows.append(
            _run_named(
                f"xgboost_{feature_set}",
                "B",
                lambda cols=columns, label=f"xgboost_{feature_set}": select_xgboost(
                    fit, val, development, holdout, cols, y_true,
                    name=label, batch="B", seed=seed, n_jobs=n_jobs, grid=False,
                ),
            )
        )
        rows.append(
            _run_named(
                f"catboost_{feature_set}",
                "B",
                lambda cols=columns, label=f"catboost_{feature_set}": select_catboost(
                    fit, val, development, holdout, cols, y_true,
                    name=label, batch="B", seed=seed, n_jobs=n_jobs, grid=False,
                ),
            )
        )
    rows.append(
        _run_named(
            "hgb_full",
            "B",
            lambda: select_hgb(fit, val, development, holdout, full, y_true, name="hgb_full", batch="B", seed=seed),
        )
    )
    for feature_set, columns in (("contextual", contextual), ("temporal", temporal)):
        rows.append(
            _run_named(
                f"ridge_{feature_set}_a1",
                "B",
                lambda cols=columns, label=f"ridge_{feature_set}_a1": select_ridge(
                    fit, val, development, holdout, cols, y_true, name=label, batch="B",
                ),
            )
        )
    for family in ("lstm", "dlinear"):
        rows.append(
            _run_named(
                f"{family}_contextual",
                "B",
                lambda family_name=family: select_sequence(
                    development, holdout, contextual, y_true,
                    name=f"{family_name}_contextual", family=family_name, batch="B", seed=seed,
                ),
            )
        )
    rows.append(_run_named("dlinear_prevday", "B", lambda: select_dlinear_prevday(development, holdout, y_true, seed=seed, batch="B")))

    public_rows = [{key: row.get(key) for key in LEADERBOARD_COLUMNS} for row in rows]
    ranked = write_leaderboard(rows, output_dir / "leaderboard.csv")
    write_comparison(rows, output_dir / "comparison_0914.csv")
    ok_rows = [row for row in public_rows if row["status"] == "ok"]
    winner = min(ok_rows, key=lambda row: row["RMSE"]) if ok_rows else None
    selected = {row["model"]: row.get("_selected") for row in rows if row.get("_selected")}
    manifest = {
        "experiment": "holdout_rmse_v2",
        "protocol": "select_on_271_30_retrain_301_score_59",
        "nested_cv": False,
        "expanding_5fold": False,
        "selection_split": "271/30",
        "holdout_not_used_for_selection": True,
        "seed": seed,
        "target": TARGET_COL,
        "metric": "RMSE",
        "train_days": DEV_DAYS,
        "val_days": VAL_DAYS,
        "test_days": HOLD_OUT_DAYS,
        "tree_round_cap": TREE_ROUND_CAP,
        "tree_early_stopping_rounds": TREE_ES_ROUNDS,
        "sequence_max_epochs": SEQ_MAX_EPOCHS,
        "sequence_patience": SEQ_PATIENCE,
        "feature_frame_rows": audit.get("feature_frame_rows"),
        "selection": selected,
        "split": extras,
        "winner": None if winner is None else {"model": winner["model"], "RMSE": winner["RMSE"], "MAE": winner["MAE"]},
        "n_models": len(public_rows),
        "n_ok": len(ok_rows),
        "v1_leaderboard": str(V1_LEADERBOARD.relative_to(ROOT)).replace("\\", "/"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output_dir / 'leaderboard.csv'}")
    print(f"wrote {output_dir / 'comparison_0914.csv'}")
    print(f"wrote {output_dir / 'manifest.json'}")
    print(ranked.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
