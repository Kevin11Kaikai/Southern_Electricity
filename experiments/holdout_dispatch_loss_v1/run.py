"""271/30/59 dispatch-loss trial for compact Transformer, LSTM, and DLinear.

Select (lambda, tau) on the last 30 development days by locked realized
profit, retrain on 301, score the last 59 once. The holdout is not used for
early stopping or hyperparameter choice.
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

import torch
from torch import nn

from experiments.holdout_dispatch_loss_v1.loss import dispatch_loss
from experiments.holdout_rmse_v1.run import _frame_to_day_arrays
from experiments.holdout_rmse_v1.sequence_models import (
    CompactDLinear,
    CompactDayLSTM,
    CompactDayTransformer,
    SequenceTrainResult,
    apply_preprocess,
    fit_preprocess_stats,
    predict_sequence,
)
from src.phase_b.config import load_config
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.data import TARGET_COL, TIME_COL, complete_day_index
from src.phase_b.dispatch import optimize_day
from src.phase_b.evaluation import evaluate_price_predictions, score_day
from src.phase_b.experiments import prepare_feature_frame
from src.phase_b.features import FeatureSets
from src.phase_b.splits import make_day_split_plan, mask_for_dates

SEED = 42
HOLD_OUT_DAYS = 59
VAL_DAYS = 30
DEV_DAYS = 301
FIT_DAYS = DEV_DAYS - VAL_DAYS
SEQ_MAX_EPOCHS = 80
SEQ_PATIENCE = 12
SEQ_EPOCH_CAP = 80
SEQUENCE_FEATURE_SET = "contextual"
FAMILIES = ("transformer", "lstm", "dlinear")
PARAM_GRID: tuple[tuple[float, float], ...] = ((0.5, 0.1), (1.0, 0.1), (0.5, 0.2))
REPORT_DIR = ROOT / "reports" / "holdout_dispatch_loss_v1"
PHASE_B_METRICS = ROOT / "reports" / "phase_b" / "final_metrics.json"
LEADERBOARD_COLUMNS = [
    "model",
    "family",
    "lam",
    "tau",
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
    "status",
]


def scaled_count(best: int, *, cap: int) -> int:
    return min(int(cap), max(1, int(round(int(best) * DEV_DAYS / FIT_DAYS))))


def _feature_columns(sets: FeatureSets, name: str) -> list[str]:
    return list(getattr(sets, name))


def _ctor(family: str, n_features: int) -> Callable[[], nn.Module]:
    if family == "lstm":
        return lambda: CompactDayLSTM(n_features)
    if family == "transformer":
        return lambda: CompactDayTransformer(n_features)
    if family == "dlinear":
        return lambda: CompactDLinear(n_features)
    raise ValueError(f"unknown sequence family: {family}")


def mean_realized(y_true_days: np.ndarray, yhat_days: np.ndarray) -> float:
    actual = np.asarray(y_true_days, dtype=float).reshape(-1, STEPS_PER_DAY)
    predicted = np.asarray(yhat_days, dtype=float).reshape(-1, STEPS_PER_DAY)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch {actual.shape} vs {predicted.shape}")
    profits = [
        score_day(true_day, optimize_day(pred_day).power)
        for true_day, pred_day in zip(actual, predicted)
    ]
    return float(np.mean(profits))


def score_predictions(times: pd.Series, y_true: np.ndarray, yhat: np.ndarray) -> dict[str, Any]:
    report = evaluate_price_predictions(times, y_true, yhat, incomplete="raise")
    daily = report["daily"]
    summary = report["summary"]
    pred_mean = float(daily["predicted_profit"].mean())
    realized = float(summary["mean_realized_profit"])
    oracle = float(summary["mean_oracle_profit"])
    capture = float(realized / oracle) if oracle else float("nan")
    return {
        "pred_profit_mean": pred_mean,
        "realized_profit_mean": realized,
        "profit_gap": pred_mean - realized,
        "oracle_profit_mean": oracle,
        "capture": capture,
        "days_evaluated": int(summary["days_evaluated"]),
        "daily": daily,
    }


def train_dispatch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    lam: float,
    tau: float,
    seed: int = SEED,
    max_epochs: int = SEQ_MAX_EPOCHS,
    patience: int = SEQ_PATIENCE,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cpu",
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
            loss = dispatch_loss(prediction, batch_y, lam=lam, tau=tau)
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
            f"[holdout_dispatch_loss_v1]     epoch {epochs_run:02d} "
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
        raise RuntimeError("dispatch-loss model produced no training state")
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
        "lam": float(lam),
        "tau": float(tau),
        "history": history,
    }
    return result, extra


def train_dispatch_fixed_epochs(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    lam: float,
    tau: float,
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
            loss = dispatch_loss(prediction, batch_y, lam=lam, tau=tau)
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


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _format_float(value: object, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{float(value):.{digits}f}"


def write_leaderboard(path: Path, rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    display = frame[LEADERBOARD_COLUMNS].copy()
    float_cols = [
        "lam",
        "tau",
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
    display.to_csv(path, index=False)


def fit_family(
    family: str,
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
    ctor = _ctor(family, n_features)
    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for lam, tau in PARAM_GRID:
        print(
            f"[holdout_dispatch_loss_v1] {family} lam={lam} tau={tau} fit on {FIT_DAYS}",
            flush=True,
        )
        try:
            selected, extra = train_dispatch_model(
                ctor(),
                x_fit,
                y_fit,
                x_val,
                y_val,
                lam=lam,
                tau=tau,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            grid_rows.append(
                {
                    "lam": lam,
                    "tau": tau,
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )
            continue
        candidate = {
            "lam": float(lam),
            "tau": float(tau),
            "best_epoch": int(extra["best_epoch"]),
            "epochs_run": int(extra["epochs_run"]),
            "val30_realized": float(extra["best_val_realized"]),
            "history": extra["history"],
            "status": "ok",
        }
        print(
            f"[holdout_dispatch_loss_v1] {family} lam={lam} tau={tau} "
            f"val30_realized={candidate['val30_realized']:.1f} "
            f"best_epoch={candidate['best_epoch']}",
            flush=True,
        )
        grid_rows.append(candidate)
        if best is None or candidate["val30_realized"] > best["val30_realized"]:
            best = candidate
            best["_selected_result"] = selected

    if best is None:
        raise RuntimeError(f"{family} produced no successful grid point")

    final_epochs = scaled_count(int(best["best_epoch"]), cap=SEQ_EPOCH_CAP)
    print(
        f"[holdout_dispatch_loss_v1] {family} retrain 301 epochs={final_epochs} "
        f"lam={best['lam']} tau={best['tau']}",
        flush=True,
    )
    retrained = train_dispatch_fixed_epochs(
        ctor(),
        x_dev,
        y_dev,
        lam=float(best["lam"]),
        tau=float(best["tau"]),
        num_epochs=final_epochs,
        seed=seed,
    )
    yhat_days = predict_sequence(retrained, x_test)
    return {
        "family": family,
        "model": f"{family}_dispatch",
        "lam": float(best["lam"]),
        "tau": float(best["tau"]),
        "best_epoch": int(best["best_epoch"]),
        "final_epochs": int(final_epochs),
        "val30_realized": float(best["val30_realized"]),
        "yhat_days": yhat_days,
        "grid": [{k: v for k, v in row.items() if k != "_selected_result"} for row in grid_rows],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="271/30/59 dispatch-loss trial")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase_b.toml"))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    parser.add_argument(
        "--families",
        default=",".join(FAMILIES),
        help="comma-separated subset of transformer,lstm,dlinear",
    )
    args = parser.parse_args(argv)
    if PHASE_B_METRICS.exists():
        print(f"[holdout_dispatch_loss_v1] leaving {PHASE_B_METRICS} untouched", flush=True)

    families = tuple(item.strip() for item in str(args.families).split(",") if item.strip())
    unknown = [name for name in families if name not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown families: {unknown}")

    config = load_config(args.config)
    seed = int(config.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[holdout_dispatch_loss_v1] preparing feature frame", flush=True)
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

    development = labeled.loc[mask_for_dates(labeled[TIME_COL], development_dates)].sort_values(TIME_COL).reset_index(drop=True)
    holdout = labeled.loc[mask_for_dates(labeled[TIME_COL], plan.holdout_dates)].sort_values(TIME_COL).reset_index(drop=True)
    contextual = _feature_columns(sets, SEQUENCE_FEATURE_SET)
    x_dev, y_dev, _dev_dates = _frame_to_day_arrays(development, contextual)
    x_test, y_test, test_dates = _frame_to_day_arrays(holdout, contextual)
    x_fit, y_fit = x_dev[:FIT_DAYS], y_dev[:FIT_DAYS]
    x_val, y_val = x_dev[FIT_DAYS:], y_dev[FIT_DAYS:]
    y_true = holdout[TARGET_COL].astype(float).to_numpy()

    pred_frame = holdout[[TIME_COL, TARGET_COL]].rename(columns={TARGET_COL: "A"}).copy()
    leaderboard_rows: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    selection: dict[str, Any] = {}

    for family in families:
        print(f"[holdout_dispatch_loss_v1] family {family}", flush=True)
        try:
            fitted = fit_family(
                family,
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
                    "model": f"{family}_dispatch",
                    "family": family,
                    "lam": np.nan,
                    "tau": np.nan,
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
                    "status": f"failed: {type(exc).__name__}: {exc}",
                }
            )
            continue

        yhat_days = np.asarray(fitted["yhat_days"], dtype=float)
        if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
            raise ValueError(f"{family} holdout shape {yhat_days.shape}")
        yhat = yhat_days.reshape(-1)
        pred_frame[fitted["model"]] = yhat
        rmse = float(mean_squared_error(y_true, yhat) ** 0.5)
        mae = float(mean_absolute_error(y_true, yhat))
        scored = score_predictions(holdout[TIME_COL], y_true, yhat)
        daily = scored.pop("daily").copy()
        daily.insert(0, "model", fitted["model"])
        daily_frames.append(daily)
        row = {
            "model": fitted["model"],
            "family": family,
            "lam": fitted["lam"],
            "tau": fitted["tau"],
            "val30_realized": fitted["val30_realized"],
            "best_epoch": fitted["best_epoch"],
            "final_epochs": fitted["final_epochs"],
            "train_days": DEV_DAYS,
            "test_days": HOLD_OUT_DAYS,
            "RMSE": rmse,
            "MAE": mae,
            "n_points": int(y_true.size),
            "pred_profit_mean": scored["pred_profit_mean"],
            "realized_profit_mean": scored["realized_profit_mean"],
            "profit_gap": scored["profit_gap"],
            "oracle_profit_mean": scored["oracle_profit_mean"],
            "capture": scored["capture"],
            "status": "ok",
        }
        leaderboard_rows.append(row)
        selection[family] = {
            "lam": fitted["lam"],
            "tau": fitted["tau"],
            "best_epoch": fitted["best_epoch"],
            "final_epochs": fitted["final_epochs"],
            "val30_realized": fitted["val30_realized"],
            "grid": fitted["grid"],
            "holdout": {
                "RMSE": rmse,
                "pred_profit_mean": scored["pred_profit_mean"],
                "realized_profit_mean": scored["realized_profit_mean"],
                "capture": scored["capture"],
            },
        }
        print(
            f"[holdout_dispatch_loss_v1] {fitted['model']} "
            f"realized={row['realized_profit_mean']:.1f} capture={row['capture']:.3f} "
            f"pred={row['pred_profit_mean']:.1f} RMSE={rmse:.4f}",
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
        "families": list(families),
        "param_grid": [{"lam": lam, "tau": tau} for lam, tau in PARAM_GRID],
        "selection_metric": "val30_realized_from_optimize_day_lock",
        "loss": "day_centered_block_sum_mse + lambda * window_cross_entropy",
        "fit_start": str(fit_dates[0].date()),
        "fit_end": str(fit_dates[-1].date()),
        "val_start": str(val_dates[0].date()),
        "val_end": str(val_dates[-1].date()),
        "holdout_start": str(plan.holdout_dates[0].date()),
        "holdout_end": str(plan.holdout_dates[-1].date()),
        "n_features": int(x_dev.shape[-1]),
        "feature_audit": json_safe(audit.to_dict() if hasattr(audit, "to_dict") else audit),
        "phase_b_metrics_untouched": True,
        "v1_v2_rmse_columns_untouched": True,
        "not_a_new_champion_on_59": True,
        "money_line": "v2 transformer_contextual realized 6214 / capture 0.690",
    }
    (output_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    print(f"[holdout_dispatch_loss_v1] wrote {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
