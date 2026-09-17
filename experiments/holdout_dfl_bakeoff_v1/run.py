"""271/30/59 DFL bakeoff: LSTM surrogates plus a 30-day realized blend.

Select hyperparameters on the last 30 development days by locked realized
profit, retrain on 301, score the last 59 once. Does not overwrite notebook 12.
"""

from __future__ import annotations

import argparse
import itertools
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

import torch
from torch import nn

from experiments.holdout_dispatch_loss_v1.loss import dispatch_loss as window_ce_loss
from experiments.holdout_dispatch_loss_v1.run import (
    DEV_DAYS,
    FIT_DAYS,
    HOLD_OUT_DAYS,
    SEQ_EPOCH_CAP,
    SEQ_MAX_EPOCHS,
    SEQ_PATIENCE,
    VAL_DAYS,
    json_safe,
    mean_realized,
    scaled_count,
    score_predictions,
)
from experiments.holdout_dfl_bakeoff_v1.losses import make_loss_fn
from experiments.holdout_rmse_v1.run import _frame_to_day_arrays, predict_ridge
from experiments.holdout_rmse_v1.sequence_models import (
    CompactDayLSTM,
    SequenceTrainResult,
    apply_preprocess,
    fit_preprocess_stats,
    predict_sequence,
)
from src.phase_b.config import load_config
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.data import TARGET_COL, TIME_COL, complete_day_index
from src.phase_b.experiments import prepare_feature_frame
from src.phase_b.features import FeatureSets
from src.phase_b.modeling import ModelSpec, deterministic_params, train_final_model, train_fold_model
from src.phase_b.splits import DayFold, make_day_split_plan, mask_for_dates

SEED = 42
SEQUENCE_FEATURE_SET = "contextual"
RIDGE_ALPHA = 1.0
TREE_ROUND_CAP = 1500
TREE_ES_ROUNDS = 100
CE_LAM = 0.5
CE_TAU = 0.2
INCUMBENT_REALIZED = 6301.331965
REPORT_DIR = ROOT / "reports" / "holdout_dfl_bakeoff_v1"
PHASE_B_METRICS = ROOT / "reports" / "phase_b" / "final_metrics.json"
KIND_GRIDS: dict[str, tuple[dict[str, float], ...]] = {
    "expected": ({"tau": 0.05}, {"tau": 0.1}, {"tau": 0.2}),
    "pairwise": ({"margin": 0.2}, {"margin": 0.5}),
    "spoplus": ({},),
    "dbb": ({"gamma": 1.0}, {"gamma": 10.0}),
}
LEADERBOARD_COLUMNS = [
    "model",
    "kind",
    "hparams",
    "val30_realized",
    "best_epoch",
    "final_epochs",
    "train_days",
    "test_days",
    "RMSE",
    "MAE",
    "n_points",
    "pred_profit_mean",
    "realized_profit_mean",
    "profit_gap",
    "oracle_profit_mean",
    "capture",
    "beat_lstm_dispatch",
    "status",
]


def _feature_columns(sets: FeatureSets, name: str) -> list[str]:
    return list(getattr(sets, name))


def _format_float(value: object, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    display = frame[LEADERBOARD_COLUMNS].copy()
    float_cols = [
        "val30_realized",
        "RMSE",
        "MAE",
        "pred_profit_mean",
        "realized_profit_mean",
        "profit_gap",
        "oracle_profit_mean",
        "capture",
    ]
    int_cols = ["best_epoch", "final_epochs", "train_days", "test_days", "n_points"]
    for column in float_cols:
        display[column] = display[column].map(_format_float)
    for column in int_cols:
        display[column] = display[column].map(
            lambda value: "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(int(value))
        )
    display["hparams"] = display["hparams"].map(lambda value: value if isinstance(value, str) else json.dumps(value))
    display.to_csv(path, index=False)


def train_lstm_loss(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    seed: int = SEED,
    max_epochs: int = SEQ_MAX_EPOCHS,
    patience: int = SEQ_PATIENCE,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
    log_prefix: str = "lstm",
) -> tuple[SequenceTrainResult, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)

    fill_values, mean, scale = fit_preprocess_stats(x_train)
    x_train_z = apply_preprocess(x_train, fill_values, mean, scale)
    x_val_z = apply_preprocess(x_val, fill_values, mean, scale)
    y_train = np.asarray(y_train, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_state: dict[str, torch.Tensor] | None = None
    best_realized = float("-inf")
    best_epoch = 0
    wait = 0
    epochs_run = 0
    n_train = int(x_train_z.shape[0])
    x_val_t = torch.from_numpy(x_val_z).to(device)
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        permutation = np.random.permutation(n_train)
        running = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(x_train_z[index]).to(device)
            batch_y = torch.from_numpy(y_train[index]).to(device)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.detach().cpu())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t).detach().cpu().numpy()
        val_realized = mean_realized(y_val, val_pred)
        epochs_run = epoch + 1
        train_loss = running / max(1, n_batches)
        history.append({"epoch": float(epochs_run), "train_loss": train_loss, "val_realized": val_realized})
        print(
            f"[holdout_dfl_bakeoff_v1] {log_prefix} epoch {epochs_run:02d} "
            f"loss={train_loss:.4f} val_realized={val_realized:.1f}",
            flush=True,
        )
        if val_realized > best_realized + 1e-3:
            best_realized = val_realized
            best_epoch = epochs_run
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is None:
        raise RuntimeError("LSTM produced no training state")
    model.load_state_dict(best_state)
    model.eval()
    result = SequenceTrainResult(
        model=model,
        best_val_rmse=float("nan"),
        epochs_run=int(epochs_run),
        best_epoch=int(best_epoch),
        fill_values=fill_values,
        mean=mean,
        scale=scale,
    )
    extra = {
        "best_val_realized": float(best_realized),
        "best_epoch": int(best_epoch),
        "epochs_run": int(epochs_run),
        "history": history,
    }
    return result, extra


def train_lstm_fixed_epochs(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    num_epochs: int,
    seed: int = SEED,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
) -> SequenceTrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    fill_values, mean, scale = fit_preprocess_stats(x_train)
    x_train_z = apply_preprocess(x_train, fill_values, mean, scale)
    y_train = np.asarray(y_train, dtype=np.float32)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = int(x_train_z.shape[0])
    epochs = max(1, int(num_epochs))
    for _epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(x_train_z[index]).to(device)
            batch_y = torch.from_numpy(y_train[index]).to(device)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return SequenceTrainResult(
        model=model,
        best_val_rmse=float("nan"),
        epochs_run=epochs,
        best_epoch=epochs,
        fill_values=fill_values,
        mean=mean,
        scale=scale,
    )


def fit_kind(
    kind: str,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    n_features = int(x_fit.shape[-1])
    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for params in KIND_GRIDS[kind]:
        label = json.dumps(params, sort_keys=True)
        print(f"[holdout_dfl_bakeoff_v1] {kind} {label} fit on {FIT_DAYS}", flush=True)
        loss_fn = make_loss_fn(kind, **params)
        try:
            selected, extra = train_lstm_loss(
                CompactDayLSTM(n_features),
                x_fit,
                y_fit,
                x_val,
                y_val,
                loss_fn,
                seed=seed,
                log_prefix=f"{kind} {label}",
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            grid_rows.append({"params": params, "status": f"failed: {type(exc).__name__}: {exc}"})
            continue
        candidate = {
            "params": dict(params),
            "best_epoch": int(extra["best_epoch"]),
            "epochs_run": int(extra["epochs_run"]),
            "val30_realized": float(extra["best_val_realized"]),
            "history": extra["history"],
            "status": "ok",
        }
        print(
            f"[holdout_dfl_bakeoff_v1] {kind} {label} "
            f"val30_realized={candidate['val30_realized']:.1f} best_epoch={candidate['best_epoch']}",
            flush=True,
        )
        grid_rows.append(candidate)
        if best is None or candidate["val30_realized"] > best["val30_realized"]:
            best = candidate

    if best is None:
        raise RuntimeError(f"{kind} produced no successful grid point")

    final_epochs = scaled_count(int(best["best_epoch"]), cap=SEQ_EPOCH_CAP)
    print(
        f"[holdout_dfl_bakeoff_v1] {kind} retrain 301 epochs={final_epochs} params={best['params']}",
        flush=True,
    )
    retrained = train_lstm_fixed_epochs(
        CompactDayLSTM(n_features),
        x_dev,
        y_dev,
        make_loss_fn(kind, **best["params"]),
        num_epochs=final_epochs,
        seed=seed,
    )
    yhat_days = predict_sequence(retrained, x_test)
    return {
        "kind": kind,
        "model": f"lstm_{kind}",
        "params": best["params"],
        "best_epoch": int(best["best_epoch"]),
        "final_epochs": int(final_epochs),
        "val30_realized": float(best["val30_realized"]),
        "yhat_days": yhat_days,
        "grid": grid_rows,
    }


def _simplex_weights(step: float = 0.25) -> list[tuple[float, float, float]]:
    n = int(round(1.0 / step))
    weights = []
    for i, j in itertools.product(range(n + 1), repeat=2):
        k = n - i - j
        if k < 0:
            continue
        weights.append((i * step, j * step, k * step))
    return weights


def fit_blend(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    fold: DayFold,
    feature_columns: list[str],
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    n_features = int(x_fit.shape[-1])
    print("[holdout_dfl_bakeoff_v1] blend ridge fit", flush=True)
    ridge_val = predict_ridge(fit, val, feature_columns, alpha=RIDGE_ALPHA).reshape(-1, STEPS_PER_DAY)
    print("[holdout_dfl_bakeoff_v1] blend lightgbm fit", flush=True)
    spec = ModelSpec(
        name="blend_lgb_contextual",
        params=deterministic_params(seed=seed, n_jobs=n_jobs, num_leaves=63, learning_rate=0.03),
        num_boost_round=TREE_ROUND_CAP,
        early_stopping_rounds=TREE_ES_ROUNDS,
        target_mode="level",
    )
    _booster, lgb_val_flat, lgb_result = train_fold_model(development, feature_columns, fold, spec)
    lgb_val = np.asarray(lgb_val_flat, dtype=float).reshape(-1, STEPS_PER_DAY)
    print("[holdout_dfl_bakeoff_v1] blend lstm window-CE fit", flush=True)
    ce_fn = lambda pred, y: window_ce_loss(pred, y, lam=CE_LAM, tau=CE_TAU)
    lstm_selected, lstm_extra = train_lstm_loss(
        CompactDayLSTM(n_features),
        x_fit,
        y_fit,
        x_val,
        y_val,
        ce_fn,
        seed=seed,
        log_prefix="blend lstm_ce",
    )
    lstm_val = predict_sequence(lstm_selected, x_val)

    best_w = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    best_realized = float("-inf")
    grid = []
    for wr, wl, wm in _simplex_weights(0.25):
        mixed = wr * ridge_val + wl * lgb_val + wm * lstm_val
        realized = mean_realized(y_val, mixed)
        grid.append({"w_ridge": wr, "w_lgb": wl, "w_lstm": wm, "val30_realized": realized})
        if realized > best_realized + 1e-6:
            best_realized = realized
            best_w = (wr, wl, wm)
    print(
        f"[holdout_dfl_bakeoff_v1] blend selected w_ridge={best_w[0]:.2f} "
        f"w_lgb={best_w[1]:.2f} w_lstm={best_w[2]:.2f} val30={best_realized:.1f}",
        flush=True,
    )

    ridge_hold = predict_ridge(development, holdout, feature_columns, alpha=RIDGE_ALPHA)
    lgb_rounds = scaled_count(int(lgb_result.best_iteration), cap=TREE_ROUND_CAP)
    lgb_final = train_final_model(development, feature_columns, spec, num_boost_round=lgb_rounds)
    lgb_hold = np.asarray(lgb_final.predict(holdout[feature_columns]), dtype=float)
    lstm_epochs = scaled_count(int(lstm_extra["best_epoch"]), cap=SEQ_EPOCH_CAP)
    lstm_final = train_lstm_fixed_epochs(
        CompactDayLSTM(n_features),
        x_dev,
        y_dev,
        ce_fn,
        num_epochs=lstm_epochs,
        seed=seed,
    )
    lstm_hold = predict_sequence(lstm_final, x_test).reshape(-1)
    yhat = best_w[0] * ridge_hold + best_w[1] * lgb_hold + best_w[2] * lstm_hold
    return {
        "kind": "blend",
        "model": "blend_val30",
        "params": {
            "w_ridge": best_w[0],
            "w_lgb": best_w[1],
            "w_lstm": best_w[2],
            "lgb_rounds": lgb_rounds,
            "lstm_epochs": lstm_epochs,
            "ce_lam": CE_LAM,
            "ce_tau": CE_TAU,
        },
        "best_epoch": int(lstm_extra["best_epoch"]),
        "final_epochs": int(lstm_epochs),
        "val30_realized": float(best_realized),
        "yhat_days": yhat.reshape(-1, STEPS_PER_DAY),
        "grid": grid,
    }


def _row_from_fit(
    fitted: dict[str, Any],
    holdout: pd.DataFrame,
    y_true: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    yhat_days = np.asarray(fitted["yhat_days"], dtype=float)
    yhat = yhat_days.reshape(-1)
    rmse = float(mean_squared_error(y_true, yhat) ** 0.5)
    mae = float(mean_absolute_error(y_true, yhat))
    scored = score_predictions(holdout[TIME_COL], y_true, yhat)
    daily = scored.pop("daily").copy()
    daily.insert(0, "model", fitted["model"])
    realized = float(scored["realized_profit_mean"])
    row = {
        "model": fitted["model"],
        "kind": fitted["kind"],
        "hparams": json.dumps(fitted["params"], sort_keys=True),
        "val30_realized": fitted["val30_realized"],
        "best_epoch": fitted["best_epoch"],
        "final_epochs": fitted["final_epochs"],
        "train_days": DEV_DAYS,
        "test_days": HOLD_OUT_DAYS,
        "RMSE": rmse,
        "MAE": mae,
        "n_points": int(y_true.size),
        "pred_profit_mean": scored["pred_profit_mean"],
        "realized_profit_mean": realized,
        "profit_gap": scored["profit_gap"],
        "oracle_profit_mean": scored["oracle_profit_mean"],
        "capture": scored["capture"],
        "beat_lstm_dispatch": bool(realized > INCUMBENT_REALIZED),
        "status": "ok",
    }
    return row, daily


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="271/30/59 DFL bakeoff")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase_b.toml"))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    parser.add_argument(
        "--kinds",
        default="expected,pairwise,spoplus,dbb,blend",
        help="comma-separated subset of expected,pairwise,spoplus,dbb,blend",
    )
    args = parser.parse_args(argv)
    if PHASE_B_METRICS.exists():
        print(f"[holdout_dfl_bakeoff_v1] leaving {PHASE_B_METRICS} untouched", flush=True)

    kinds = tuple(item.strip() for item in str(args.kinds).split(",") if item.strip())
    allowed = set(KIND_GRIDS) | {"blend"}
    unknown = [name for name in kinds if name not in allowed]
    if unknown:
        raise ValueError(f"unknown kinds: {unknown}")

    config = load_config(args.config)
    seed = int(config.seed)
    n_jobs = int(config.section("model")["n_jobs"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[holdout_dfl_bakeoff_v1] preparing feature frame", flush=True)
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
    fit_dates = development_dates[:FIT_DAYS]
    val_dates = development_dates[-VAL_DAYS:]
    fold = DayFold(fold=0, train_dates=fit_dates, validation_dates=val_dates)
    development = labeled.loc[mask_for_dates(labeled[TIME_COL], development_dates)].sort_values(TIME_COL).reset_index(drop=True)
    fit = labeled.loc[mask_for_dates(labeled[TIME_COL], fit_dates)].sort_values(TIME_COL).reset_index(drop=True)
    val = labeled.loc[mask_for_dates(labeled[TIME_COL], val_dates)].sort_values(TIME_COL).reset_index(drop=True)
    holdout = labeled.loc[mask_for_dates(labeled[TIME_COL], plan.holdout_dates)].sort_values(TIME_COL).reset_index(drop=True)
    contextual = _feature_columns(sets, SEQUENCE_FEATURE_SET)
    x_dev, y_dev, _ = _frame_to_day_arrays(development, contextual)
    x_test, _, test_dates = _frame_to_day_arrays(holdout, contextual)
    x_fit, y_fit = x_dev[:FIT_DAYS], y_dev[:FIT_DAYS]
    x_val, y_val = x_dev[FIT_DAYS:], y_dev[FIT_DAYS:]
    y_true = holdout[TARGET_COL].astype(float).to_numpy()

    pred_frame = holdout[[TIME_COL, TARGET_COL]].rename(columns={TARGET_COL: "A"}).copy()
    leaderboard_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    selection: dict[str, Any] = {}

    for kind in kinds:
        print(f"[holdout_dfl_bakeoff_v1] kind {kind}", flush=True)
        try:
            if kind == "blend":
                fitted = fit_blend(
                    fit,
                    val,
                    development,
                    holdout,
                    fold,
                    contextual,
                    x_fit,
                    y_fit,
                    x_val,
                    y_val,
                    x_dev,
                    y_dev,
                    x_test,
                    seed=seed,
                    n_jobs=n_jobs,
                )
            else:
                fitted = fit_kind(
                    kind,
                    x_fit,
                    y_fit,
                    x_val,
                    y_val,
                    x_dev,
                    y_dev,
                    x_test,
                    seed=seed,
                )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            leaderboard_rows.append(
                {
                    "model": f"lstm_{kind}" if kind != "blend" else "blend_val30",
                    "kind": kind,
                    "hparams": "{}",
                    "val30_realized": np.nan,
                    "best_epoch": np.nan,
                    "final_epochs": np.nan,
                    "train_days": DEV_DAYS,
                    "test_days": HOLD_OUT_DAYS,
                    "RMSE": np.nan,
                    "MAE": np.nan,
                    "n_points": np.nan,
                    "pred_profit_mean": np.nan,
                    "realized_profit_mean": np.nan,
                    "profit_gap": np.nan,
                    "oracle_profit_mean": np.nan,
                    "capture": np.nan,
                    "beat_lstm_dispatch": False,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        yhat_days = np.asarray(fitted["yhat_days"], dtype=float)
        if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
            raise ValueError(f"{fitted['model']} holdout shape {yhat_days.shape}")
        pred_frame[fitted["model"]] = yhat_days.reshape(-1)
        row, daily = _row_from_fit(fitted, holdout, y_true)
        leaderboard_rows.append(row)
        daily_frames.append(daily)
        selection[kind] = {
            "params": fitted["params"],
            "best_epoch": fitted["best_epoch"],
            "final_epochs": fitted["final_epochs"],
            "val30_realized": fitted["val30_realized"],
            "grid": fitted["grid"],
            "holdout": {
                "RMSE": row["RMSE"],
                "pred_profit_mean": row["pred_profit_mean"],
                "realized_profit_mean": row["realized_profit_mean"],
                "capture": row["capture"],
            },
        }
        print(
            f"[holdout_dfl_bakeoff_v1] {row['model']} "
            f"realized={row['realized_profit_mean']:.1f} capture={row['capture']:.3f} "
            f"beat={row['beat_lstm_dispatch']}",
            flush=True,
        )

    write_leaderboard(output_dir / "leaderboard.csv", leaderboard_rows)
    pred_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    pred_frame.to_csv(output_dir / "predictions.csv", index=False)
    if daily_frames:
        pd.concat(daily_frames, ignore_index=True).to_csv(output_dir / "dispatch_daily.csv", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(json_safe(selection), indent=2),
        encoding="utf-8",
    )
    manifest = {
        "protocol": "select_on_271_30_retrain_301_score_59",
        "holdout_not_used_for_selection": True,
        "seed": seed,
        "feature_set": SEQUENCE_FEATURE_SET,
        "backbone": "CompactDayLSTM",
        "kinds": list(kinds),
        "incumbent": "lstm_dispatch",
        "incumbent_realized": INCUMBENT_REALIZED,
        "does_not_overwrite_holdout_dispatch_loss_v1": True,
        "phase_b_metrics_untouched": True,
        "not_a_new_champion_on_59": True,
        "spo_plus_is_end_to_end_lstm_not_p01_calibrator": True,
        "fit_start": str(fit_dates[0].date()),
        "fit_end": str(fit_dates[-1].date()),
        "val_start": str(val_dates[0].date()),
        "val_end": str(val_dates[-1].date()),
        "holdout_start": str(plan.holdout_dates[0].date()),
        "holdout_end": str(plan.holdout_dates[-1].date()),
        "n_features": int(x_dev.shape[-1]),
        "feature_audit": json_safe(audit.to_dict() if hasattr(audit, "to_dict") else audit),
    }
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    print(f"[holdout_dfl_bakeoff_v1] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
