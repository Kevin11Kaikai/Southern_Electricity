"""15 frozen 301-day retrains: 3 bags x seeds 42-46, then curve-mean lock.

Does not search grids.  Does not use the 59-day holdout to choose bags or
combiners.  Headline bag is the pre-registered listwise Transformer mean.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
from experiments.holdout_dfl_ltr_v1.loss import build_loss
from experiments.holdout_dfl_ltr_v1.run import score_predictions
from experiments.holdout_rmse_v1.run import _frame_to_day_arrays
from experiments.holdout_rmse_v1.sequence_models import (
    CompactDayLSTM,
    CompactDayTransformer,
    SequenceTrainResult,
    apply_preprocess,
    fit_preprocess_stats,
    predict_sequence,
)
from experiments.holdout_seed_bag_v1.combine import (
    mean_curves,
    vote_windows,
    windows_to_power,
)
from src.phase_b.config import load_config
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.data import TARGET_COL, TIME_COL, complete_day_index
from src.phase_b.dispatch import optimize_day
from src.phase_b.evaluation import score_day
from src.phase_b.experiments import prepare_feature_frame
from src.phase_b.splits import make_day_split_plan, mask_for_dates

FINAL_SEEDS = (42, 43, 44, 45, 46)
HOLD_OUT_DAYS = 59
DEV_DAYS = 301
SEQUENCE_FEATURE_SET = "contextual"
REPORT_DIR = ROOT / "reports" / "holdout_seed_bag_v1"
CACHE = REPORT_DIR / "_feature_cache.npz"
PHASE_B_METRICS = ROOT / "reports" / "phase_b" / "final_metrics.json"
HEADLINE_BAG = "listwise_transformer"
MONEY_LINE = 6214.361782
MONEY_LINE_SOURCE = "nb11 v2 transformer_contextual"

BAGS: tuple[dict[str, Any], ...] = (
    {
        "name": "listwise_transformer",
        "family": "transformer",
        "route": "r1_listwise",
        "params": {"lam": 2.0, "tau_tgt": 0.05, "tau_pred": 0.1},
        "epochs": 13,
        "note": "Claude B val30 capture 0.600; tau_tgt=0.05 is a sharper notebook-12 CE",
    },
    {
        "name": "spo_lstm",
        "family": "lstm",
        "route": "r3_spo",
        "params": {"w_spo": 0.5, "lr": 0.0003},
        "epochs": 27,
        "note": "Claude B SPO+ LSTM; bagging may not lift a low-variance head",
    },
    {
        "name": "window_ce_lstm",
        "family": "lstm",
        "route": "window_ce",
        "params": {"lam": 0.5, "tau": 0.2},
        "epochs": 34,
        "note": "nb12 lstm_dispatch recipe, 5 seeds, seed-before-constructor",
    },
)


def _ctor(family: str, n_features: int) -> Callable[[], nn.Module]:
    if family == "lstm":
        return lambda: CompactDayLSTM(n_features)
    if family == "transformer":
        return lambda: CompactDayTransformer(n_features)
    raise ValueError(f"unknown family {family}")


def _loss_fn(route: str, params: dict[str, Any]):
    if route == "window_ce":
        return lambda pred, y: dispatch_loss(pred, y, lam=params["lam"], tau=params["tau"])
    return build_loss(route, params)


def retrain_one(
    family: str,
    n_features: int,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    x_test: np.ndarray,
    *,
    route: str,
    params: dict[str, Any],
    num_epochs: int,
    seed: int,
    batch_size: int = 16,
    device: str = "cpu",
) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    lr = float(params.get("lr", 1e-3))
    loss_fn = _loss_fn(route, params)
    fill_values, mean, scale = fit_preprocess_stats(x_dev)
    x_dev_z = apply_preprocess(x_dev, fill_values, mean, scale)
    y_dev32 = np.asarray(y_dev, dtype=np.float32)
    model = _ctor(family, n_features)().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = int(x_dev_z.shape[0])
    epochs = max(1, int(num_epochs))
    for epoch in range(epochs):
        model.train()
        permutation = np.random.permutation(n_train)
        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(x_dev_z[index]).to(device)
            batch_y = torch.from_numpy(y_dev32[index]).to(device)
            prediction = model(batch_x)
            loss = loss_fn(prediction, batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch == 0 or epoch + 1 == epochs or (epoch + 1) % 10 == 0:
            print(
                f"[seed_bag]     {family} seed={seed} epoch {epoch + 1}/{epochs} "
                f"loss={float(loss.detach().cpu()):.4f}",
                flush=True,
            )
    model.eval()
    result = SequenceTrainResult(
        model=model,
        best_val_rmse=float("nan"),
        epochs_run=epochs,
        best_epoch=epochs,
        fill_values=fill_values,
        mean=mean,
        scale=scale,
    )
    pred = np.asarray(predict_sequence(result, x_test), dtype=float)
    return pred.reshape(-1, STEPS_PER_DAY)


def load_arrays(config_path: str) -> dict[str, Any]:
    if CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        print(f"[seed_bag] reusing feature cache {CACHE.name}", flush=True)
        return {
            "x_dev": blob["x_dev"],
            "y_dev": blob["y_dev"],
            "x_test": blob["x_test"],
            "y_test": blob["y_test"],
            "test_times": pd.to_datetime(blob["test_times"]),
            "test_y_true": blob["test_y_true"],
            "audit": json.loads(str(blob["audit"])),
        }

    print("[seed_bag] preparing feature frame", flush=True)
    config = load_config(config_path)
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
    development = (
        labeled.loc[mask_for_dates(labeled[TIME_COL], development_dates)]
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    holdout = (
        labeled.loc[mask_for_dates(labeled[TIME_COL], plan.holdout_dates)]
        .sort_values(TIME_COL)
        .reset_index(drop=True)
    )
    contextual = list(getattr(sets, SEQUENCE_FEATURE_SET))
    x_dev, y_dev, _ = _frame_to_day_arrays(development, contextual)
    x_test, y_test, _ = _frame_to_day_arrays(holdout, contextual)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE,
        x_dev=x_dev,
        y_dev=y_dev,
        x_test=x_test,
        y_test=y_test,
        test_times=holdout[TIME_COL].to_numpy(),
        test_y_true=holdout[TARGET_COL].astype(float).to_numpy(),
        audit=json.dumps(audit, default=str),
    )
    return {
        "x_dev": x_dev,
        "y_dev": y_dev,
        "x_test": x_test,
        "y_test": y_test,
        "test_times": holdout[TIME_COL],
        "test_y_true": holdout[TARGET_COL].astype(float).to_numpy(),
        "audit": audit,
    }


def score_locked_power(
    times: pd.Series,
    y_true: np.ndarray,
    power_days: np.ndarray,
    yhat_for_rmse: np.ndarray,
) -> dict[str, Any]:
    y_true_days = np.asarray(y_true, dtype=float).reshape(-1, STEPS_PER_DAY)
    yhat_days = np.asarray(yhat_for_rmse, dtype=float).reshape(-1, STEPS_PER_DAY)
    power_days = np.asarray(power_days, dtype=float).reshape(-1, STEPS_PER_DAY)
    dates = pd.to_datetime(times).dt.normalize().unique()
    rows = []
    for date, actual, pred, power in zip(dates, y_true_days, yhat_days, power_days):
        oracle = optimize_day(actual)
        tc_idx = np.where(power < 0)[0]
        td_idx = np.where(power > 0)[0]
        idle = not (tc_idx.size and td_idx.size)
        tc = int(tc_idx[0]) if not idle else None
        td = int(td_idx[0]) if not idle else None
        predicted_profit = float(np.dot(pred, power))
        realized = float(score_day(actual, power))
        oracle_profit = float(score_day(actual, oracle.power))
        rows.append(
            {
                "date": pd.Timestamp(date).date().isoformat(),
                "tc": tc,
                "td": td,
                "idle": idle,
                "predicted_profit": predicted_profit,
                "realized_profit": realized,
                "oracle_tc": oracle.tc,
                "oracle_td": oracle.td,
                "oracle_profit": oracle_profit,
                "regret": oracle_profit - realized,
                "capture_rate": realized / oracle_profit if oracle_profit else float("nan"),
                "rmse": float(np.sqrt(np.mean((actual - pred) ** 2))),
                "mae": float(np.mean(np.abs(actual - pred))),
            }
        )
    daily = pd.DataFrame(rows)
    realized = float(daily["realized_profit"].mean())
    pred_mean = float(daily["predicted_profit"].mean())
    oracle = float(daily["oracle_profit"].mean())
    return {
        "pred_profit_mean": pred_mean,
        "realized_profit_mean": realized,
        "profit_gap": pred_mean - realized,
        "oracle_profit_mean": oracle,
        "capture": realized / oracle if oracle else float("nan"),
        "daily": daily,
    }


def _leader_row(
    name: str,
    bag: dict[str, Any] | None,
    scored: dict[str, Any],
    y_true: np.ndarray,
    yhat: np.ndarray,
    *,
    combiner: str,
    n_seeds: int,
    seed_realized: list[float] | None,
    headline: bool,
) -> dict[str, Any]:
    yhat_flat = np.asarray(yhat, dtype=float).reshape(-1)
    return {
        "model": name,
        "bag": bag["name"] if bag else "mix",
        "combiner": combiner,
        "headline": bool(headline),
        "family": bag["family"] if bag else "mix",
        "route": bag["route"] if bag else "mean_of_bags",
        "config": json.dumps(bag["params"] if bag else {}, sort_keys=True),
        "n_seeds": int(n_seeds),
        "final_epochs": bag["epochs"] if bag else None,
        "train_days": DEV_DAYS,
        "test_days": HOLD_OUT_DAYS,
        "RMSE": float(mean_squared_error(y_true, yhat_flat) ** 0.5),
        "MAE": float(mean_absolute_error(y_true, yhat_flat)),
        "n_points": int(np.asarray(y_true).size),
        "pred_profit_mean": scored["pred_profit_mean"],
        "realized_profit_mean": scored["realized_profit_mean"],
        "profit_gap": scored["profit_gap"],
        "oracle_profit_mean": scored["oracle_profit_mean"],
        "capture": scored["capture"],
        "seed_realized_min": float(np.min(seed_realized)) if seed_realized else None,
        "seed_realized_max": float(np.max(seed_realized)) if seed_realized else None,
        "beats_nb11_6214": bool(scored["realized_profit_mean"] > MONEY_LINE),
        "status": "ok",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen 5-seed curve-mean wrap-up vs 6214")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase_b.toml"))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if PHASE_B_METRICS.exists():
        print(f"[seed_bag] leaving {PHASE_B_METRICS} untouched", flush=True)

    data = load_arrays(args.config)
    x_dev, y_dev = data["x_dev"], data["y_dev"]
    x_test = data["x_test"]
    test_times = pd.Series(pd.to_datetime(data["test_times"]))
    y_true = np.asarray(data["test_y_true"], dtype=float)
    n_features = int(x_dev.shape[-1])
    pred_frame = pd.DataFrame({"times": test_times, "A": y_true})
    leaderboard: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    bag_means: dict[str, np.ndarray] = {}
    started = time.time()

    for bag in BAGS:
        print(f"[seed_bag] bag {bag['name']} epochs={bag['epochs']}", flush=True)
        per_seed = []
        seed_realized = []
        for seed in FINAL_SEEDS:
            t0 = time.time()
            pred = retrain_one(
                bag["family"],
                n_features,
                x_dev,
                y_dev,
                x_test,
                route=bag["route"],
                params=bag["params"],
                num_epochs=int(bag["epochs"]),
                seed=seed,
            )
            per_seed.append(pred)
            seed_score = score_predictions(test_times, y_true, pred.reshape(-1))
            seed_realized.append(seed_score["realized_profit_mean"])
            pred_frame[f"{bag['name']}_seed{seed}"] = pred.reshape(-1)
            print(
                f"[seed_bag]   seed={seed} realized={seed_score['realized_profit_mean']:.1f} "
                f"seconds={time.time() - t0:.1f}",
                flush=True,
            )
        stack = np.stack(per_seed, axis=0)
        averaged = mean_curves(stack)
        bag_means[bag["name"]] = averaged
        pred_frame[f"{bag['name']}_mean"] = averaged.reshape(-1)
        scored = score_predictions(test_times, y_true, averaged.reshape(-1))
        daily = scored.pop("daily").copy()
        daily.insert(0, "model", f"{bag['name']}_mean")
        daily_frames.append(daily)
        leaderboard.append(
            _leader_row(
                f"{bag['name']}_mean",
                bag,
                scored,
                y_true,
                averaged,
                combiner="curve_mean",
                n_seeds=len(FINAL_SEEDS),
                seed_realized=seed_realized,
                headline=bag["name"] == HEADLINE_BAG,
            )
        )
        voted = vote_windows(stack)
        vote_power = windows_to_power(voted)
        vote_scored = score_locked_power(test_times, y_true, vote_power, averaged)
        vote_daily = vote_scored.pop("daily").copy()
        vote_daily.insert(0, "model", f"{bag['name']}_vote")
        daily_frames.append(vote_daily)
        leaderboard.append(
            _leader_row(
                f"{bag['name']}_vote",
                bag,
                vote_scored,
                y_true,
                averaged,
                combiner="window_vote",
                n_seeds=len(FINAL_SEEDS),
                seed_realized=seed_realized,
                headline=False,
            )
        )
        for seed, pred, realized in zip(FINAL_SEEDS, per_seed, seed_realized):
            seed_scored = score_predictions(test_times, y_true, pred.reshape(-1))
            seed_daily = seed_scored.pop("daily").copy()
            seed_daily.insert(0, "model", f"{bag['name']}_seed{seed}")
            daily_frames.append(seed_daily)
            leaderboard.append(
                _leader_row(
                    f"{bag['name']}_seed{seed}",
                    bag,
                    seed_scored,
                    y_true,
                    pred,
                    combiner="single_seed",
                    n_seeds=1,
                    seed_realized=[realized],
                    headline=False,
                )
            )

    mix = 0.5 * bag_means["listwise_transformer"] + 0.5 * bag_means["spo_lstm"]
    pred_frame["listwise_spo_50_50"] = mix.reshape(-1)
    mix_scored = score_predictions(test_times, y_true, mix.reshape(-1))
    mix_daily = mix_scored.pop("daily").copy()
    mix_daily.insert(0, "model", "listwise_spo_50_50")
    daily_frames.append(mix_daily)
    leaderboard.append(
        _leader_row(
            "listwise_spo_50_50",
            None,
            mix_scored,
            y_true,
            mix,
            combiner="curve_mean_two_bags",
            n_seeds=10,
            seed_realized=None,
            headline=False,
        )
    )

    board = pd.DataFrame(leaderboard)
    board.to_csv(output_dir / "leaderboard.csv", index=False)
    all_daily = pd.concat(daily_frames, ignore_index=True)
    all_daily["date"] = pd.to_datetime(all_daily["date"], format="mixed").dt.strftime("%Y-%m-%d")
    all_daily.to_csv(output_dir / "dispatch_daily.csv", index=False)
    pred_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    pred_frame.to_csv(output_dir / "predictions.csv", index=False)

    headline = board.loc[board["model"] == f"{HEADLINE_BAG}_mean"].iloc[0]
    manifest = {
        "protocol": "seed_bag_curve_mean_then_optimize_day",
        "holdout_not_used_for_selection": True,
        "headline_bag": HEADLINE_BAG,
        "headline_combiner": "curve_mean",
        "seeds": list(FINAL_SEEDS),
        "bags": BAGS,
        "money_line": MONEY_LINE,
        "money_line_source": MONEY_LINE_SOURCE,
        "holdout_se_approx": 550,
        "phase_b_metrics_untouched": True,
        "v1_v2_rmse_columns_untouched": True,
        "not_a_new_champion_on_59": True,
        "headline_realized": float(headline["realized_profit_mean"]),
        "headline_beats_nb11_6214": bool(headline["beats_nb11_6214"]),
        "seconds_total": round(time.time() - started, 1),
        "feature_audit": data["audit"],
        "notes": [
            "tau_tgt=0.05 is the R1 grid minimum: a sharper notebook-12 window CE, not a new loss champion.",
            "Do not rank the three bags on the 59-day holdout; SE of the 59-day mean is about 550.",
            "Window vote and 50/50 mix are pre-registered ablations and cannot replace the headline.",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"[seed_bag] headline {HEADLINE_BAG}_mean realized="
        f"{headline['realized_profit_mean']:.2f} vs 6214 "
        f"beats={headline['beats_nb11_6214']} seconds={manifest['seconds_total']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
