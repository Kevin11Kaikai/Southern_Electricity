"""271/30/301/59 runner for the R1-R3 and R6 losses, with A and B selection tracks.

Protocol is the notebook-12 protocol (``.cursor/plans/dispatch_loss_trial_f7f448b5``)
with four declared deviations, see ``PLAN_ZH.md`` section 3a:

  P1  grid uses 2 seeds; the selected config is retrained with 5 seeds (42-46).
  P2  a second selection statistic, the 30-day *daily-mean capture*.
  P3  much larger grids.
  P4  paired-bootstrap reporting on top of the pass line.

Both selection tracks are reported:

  A  "nb12-identical"  seed 42 only, config and epoch chosen by 30-day realized
                       mean.  Directly comparable to the 6301.33 incumbent.
  B  "robust"          config chosen by mean 30-day daily-mean capture over the
                       grid seeds, scored as the mean over 5 retrain seeds.

One training run per (route, family, config, seed) serves both tracks: the run
records both statistics for every epoch and only the epoch *numbers* are carried
into the 301-day retrain, exactly as nb12 does.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from experiments.holdout_dfl_ltr_v1.loss import build_loss
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
GRID_SEEDS = (42, 43)
FINAL_SEEDS = (42, 43, 44, 45, 46)
HOLD_OUT_DAYS = 59
VAL_DAYS = 30
DEV_DAYS = 301
FIT_DAYS = DEV_DAYS - VAL_DAYS
SEQ_MAX_EPOCHS = 80
SEQ_PATIENCE = 12
SEQ_EPOCH_CAP = 80
SEQUENCE_FEATURE_SET = "contextual"
FAMILIES = ("transformer", "lstm", "dlinear")
REPORT_DIR = ROOT / "reports" / "holdout_dfl_ltr_v1"
CACHE = REPORT_DIR / "_feature_cache.npz"
PHASE_B_METRICS = ROOT / "reports" / "phase_b" / "final_metrics.json"

MONEY_LINE = 6301.331965  # nb12 lstm_dispatch, the incumbent


def _grid_r1() -> list[dict[str, float]]:
    out = []
    for lam in (0.5, 1.0, 2.0):
        for tau_tgt in (0.05, 0.1, 0.25, 0.5):
            for tau_pred in (0.1, 0.25):
                out.append({"lam": lam, "tau_tgt": tau_tgt, "tau_pred": tau_pred})
    return out


def _grid_r2() -> list[dict[str, float]]:
    return [
        {"lam": lam, "n_pairs": n}
        for lam in (0.5, 1.0, 2.0)
        for n in (8, 32, 128)
    ]


def _grid_r3() -> list[dict[str, float]]:
    return [{"w_spo": w, "lr": lr} for w in (0.25, 0.5, 1.0) for lr in (1e-3, 3e-4)]


def _grid_r6() -> list[dict[str, Any]]:
    return [
        {"kind": "pinball", "param": 0.3},
        {"kind": "pinball", "param": 0.5},
        {"kind": "peakw", "param": 4.0},
        {"kind": "peakw", "param": 12.0},
    ]


ROUTES: dict[str, list[dict[str, Any]]] = {
    "r1_listwise": _grid_r1(),
    "r2_pairdiff": _grid_r2(),
    "r3_spo": _grid_r3(),
    "r6_cheap": _grid_r6(),
}


def config_key(params: dict[str, Any]) -> str:
    return "|".join(f"{k}={params[k]}" for k in sorted(params))


def scaled_count(best: int, *, cap: int = SEQ_EPOCH_CAP) -> int:
    """nb12's epoch transfer from the 271-day fit to the 301-day retrain."""

    return min(int(cap), max(1, int(round(int(best) * DEV_DAYS / FIT_DAYS))))


def _ctor(family: str, n_features: int) -> Callable[[], nn.Module]:
    if family == "lstm":
        return lambda: CompactDayLSTM(n_features)
    if family == "transformer":
        return lambda: CompactDayTransformer(n_features)
    if family == "dlinear":
        return lambda: CompactDLinear(n_features)
    raise ValueError(f"unknown sequence family: {family}")


# ------------------------------------------------------------------- scoring


def day_oracles(y_days: np.ndarray) -> np.ndarray:
    actual = np.asarray(y_days, dtype=float).reshape(-1, STEPS_PER_DAY)
    return np.array(
        [score_day(day, optimize_day(day).power) for day in actual], dtype=float
    )


def val_stats(y_days: np.ndarray, yhat_days: np.ndarray, oracle: np.ndarray) -> tuple[float, float]:
    """Return ``(realized_mean, daily_mean_capture)`` on the validation window."""

    actual = np.asarray(y_days, dtype=float).reshape(-1, STEPS_PER_DAY)
    predicted = np.asarray(yhat_days, dtype=float).reshape(-1, STEPS_PER_DAY)
    realized = np.array(
        [
            score_day(true_day, optimize_day(pred_day).power)
            for true_day, pred_day in zip(actual, predicted)
        ],
        dtype=float,
    )
    safe = np.where(np.abs(oracle) > 1e-9, oracle, np.nan)
    capture = np.nanmean(realized / safe)
    return float(realized.mean()), float(capture)


def score_predictions(times: pd.Series, y_true: np.ndarray, yhat: np.ndarray) -> dict[str, Any]:
    report = evaluate_price_predictions(times, y_true, yhat, incomplete="raise")
    daily = report["daily"]
    summary = report["summary"]
    pred_mean = float(daily["predicted_profit"].mean())
    realized = float(summary["mean_realized_profit"])
    oracle = float(summary["mean_oracle_profit"])
    return {
        "pred_profit_mean": pred_mean,
        "realized_profit_mean": realized,
        "profit_gap": pred_mean - realized,
        "oracle_profit_mean": oracle,
        "capture": float(realized / oracle) if oracle else float("nan"),
        "daily": daily,
    }


# ------------------------------------------------------------------ training


def train_grid_run(
    family: str,
    n_features: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_oracle: np.ndarray,
    *,
    route: str,
    params: dict[str, Any],
    seed: int,
    max_epochs: int = SEQ_MAX_EPOCHS,
    patience: int = SEQ_PATIENCE,
    batch_size: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    """Fit on 271 days, recording BOTH selection statistics for every epoch.

    Early stopping waits until neither statistic has improved for ``patience``
    epochs, so the A and B tracks each get a fair epoch search from one run.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    lr = float(params.get("lr", 1e-3))
    loss_fn = build_loss(route, params)

    fill_values, mean, scale = fit_preprocess_stats(x_train)
    x_train_z = apply_preprocess(x_train, fill_values, mean, scale)
    x_val_z = apply_preprocess(x_val, fill_values, mean, scale)
    y_train32 = np.asarray(y_train, dtype=np.float32)

    model = _ctor(family, n_features)().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_val_t = torch.from_numpy(x_val_z).to(device)
    n_train = int(x_train_z.shape[0])

    best_realized, best_epoch_realized = float("-inf"), 1
    best_capture, best_epoch_capture = float("-inf"), 1
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(max_epochs):
        model.train()
        permutation = np.random.permutation(n_train)
        running, n_batches = 0.0, 0
        for start in range(0, n_train, batch_size):
            index = permutation[start : start + batch_size]
            batch_x = torch.from_numpy(x_train_z[index]).to(device)
            batch_y = torch.from_numpy(y_train32[index]).to(device)
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
        realized, capture = val_stats(y_val, val_pred, val_oracle)
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": running / max(1, n_batches),
                "val_realized": realized,
                "val_capture": capture,
            }
        )
        improved = False
        if realized > best_realized + 1e-3:
            best_realized, best_epoch_realized, improved = realized, epoch + 1, True
        if capture > best_capture + 1e-6:
            best_capture, best_epoch_capture, improved = capture, epoch + 1, True
        wait = 0 if improved else wait + 1
        if wait >= patience:
            break

    return {
        "route": route,
        "family": family,
        "params": params,
        "seed": int(seed),
        "epochs_run": len(history),
        "best_val_realized": float(best_realized),
        "best_epoch_realized": int(best_epoch_realized),
        "best_val_capture": float(best_capture),
        "best_epoch_capture": int(best_epoch_capture),
        "history": history,
    }


def retrain_fixed(
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
    """Retrain from scratch on all 301 development days, predict the holdout."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    lr = float(params.get("lr", 1e-3))
    loss_fn = build_loss(route, params)

    fill_values, mean, scale = fit_preprocess_stats(x_dev)
    x_dev_z = apply_preprocess(x_dev, fill_values, mean, scale)
    y_dev32 = np.asarray(y_dev, dtype=np.float32)

    model = _ctor(family, n_features)().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = int(x_dev_z.shape[0])
    for _epoch in range(max(1, int(num_epochs))):
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
    model.eval()
    result = SequenceTrainResult(
        model=model,
        best_val_rmse=float("nan"),
        epochs_run=int(num_epochs),
        best_epoch=int(num_epochs),
        fill_values=fill_values,
        mean=mean,
        scale=scale,
    )
    return np.asarray(predict_sequence(result, x_test), dtype=float)


# ----------------------------------------------------------------- data prep


def load_arrays(config_path: str) -> dict[str, Any]:
    if CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        print(f"[dfl_ltr] reusing feature cache {CACHE.name}", flush=True)
        return {
            "x_dev": blob["x_dev"],
            "y_dev": blob["y_dev"],
            "x_test": blob["x_test"],
            "y_test": blob["y_test"],
            "test_times": pd.to_datetime(blob["test_times"]),
            "test_y_true": blob["test_y_true"],
            "audit": json.loads(str(blob["audit"])),
        }

    print("[dfl_ltr] preparing feature frame (first run)", flush=True)
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


# ----------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R1-R3 + R6 dispatch-loss trial")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase_b.toml"))
    parser.add_argument("--routes", default=",".join(ROUTES))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--output-dir", default=str(REPORT_DIR))
    args = parser.parse_args(argv)

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    for r in routes:
        if r not in ROUTES:
            raise ValueError(f"unknown route {r}")
    for f in families:
        if f not in FAMILIES:
            raise ValueError(f"unknown family {f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if PHASE_B_METRICS.exists():
        print(f"[dfl_ltr] leaving {PHASE_B_METRICS} untouched", flush=True)

    data = load_arrays(args.config)
    x_dev, y_dev = data["x_dev"], data["y_dev"]
    x_test = data["x_test"]
    test_times = pd.Series(pd.to_datetime(data["test_times"]))
    y_true = np.asarray(data["test_y_true"], dtype=float)
    n_features = int(x_dev.shape[-1])
    x_fit, y_fit = x_dev[:FIT_DAYS], y_dev[:FIT_DAYS]
    x_val, y_val = x_dev[FIT_DAYS:], y_dev[FIT_DAYS:]
    val_oracle = day_oracles(y_val)
    print(
        f"[dfl_ltr] dev {x_dev.shape} fit {x_fit.shape} val {x_val.shape} "
        f"test {x_test.shape} n_features={n_features} "
        f"val30 oracle/day={val_oracle.mean():.1f}",
        flush=True,
    )

    grid_records: list[dict[str, Any]] = []
    selection: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    daily_frames: list[pd.DataFrame] = []
    pred_frame = pd.DataFrame({"times": test_times, "A": y_true})
    retrain_cache: dict[tuple, np.ndarray] = {}
    t_start = time.time()

    for route in routes:
        grid = ROUTES[route]
        for family in families:
            tag = f"{route}/{family}"
            runs: list[dict[str, Any]] = []
            for params in grid:
                for seed in GRID_SEEDS:
                    t0 = time.time()
                    try:
                        rec = train_grid_run(
                            family,
                            n_features,
                            x_fit,
                            y_fit,
                            x_val,
                            y_val,
                            val_oracle,
                            route=route,
                            params=params,
                            seed=seed,
                        )
                    except Exception as exc:  # noqa: BLE001
                        traceback.print_exc()
                        rec = {
                            "route": route,
                            "family": family,
                            "params": params,
                            "seed": seed,
                            "status": f"failed: {type(exc).__name__}: {exc}",
                            "best_val_realized": float("-inf"),
                            "best_val_capture": float("-inf"),
                            "best_epoch_realized": 1,
                            "best_epoch_capture": 1,
                            "epochs_run": 0,
                            "history": [],
                        }
                    rec["seconds"] = round(time.time() - t0, 2)
                    rec["config_key"] = config_key(params)
                    runs.append(rec)
                    grid_records.append(rec)
                    print(
                        f"[dfl_ltr] {tag} {rec['config_key']} seed={seed} "
                        f"realized={rec['best_val_realized']:.1f}@{rec['best_epoch_realized']} "
                        f"capture={rec['best_val_capture']:.4f}@{rec['best_epoch_capture']} "
                        f"({rec['seconds']}s)",
                        flush=True,
                    )

            frame = pd.DataFrame(
                [
                    {
                        "config_key": r["config_key"],
                        "seed": r["seed"],
                        "realized": r["best_val_realized"],
                        "capture": r["best_val_capture"],
                        "epoch_realized": r["best_epoch_realized"],
                        "epoch_capture": r["best_epoch_capture"],
                    }
                    for r in runs
                ]
            )
            by_key = {r["config_key"]: r["params"] for r in runs}

            # --- Track A: nb12-identical. seed 42 only, realized mean.
            seed42 = frame.loc[frame["seed"] == GRID_SEEDS[0]]
            a_row = seed42.loc[seed42["realized"].idxmax()]
            a_params = by_key[a_row["config_key"]]
            a_epochs = scaled_count(int(a_row["epoch_realized"]))

            # --- Track B: mean daily-mean capture over grid seeds.
            agg = frame.groupby("config_key").agg(
                capture=("capture", "mean"),
                epoch=("epoch_capture", "median"),
                realized=("realized", "mean"),
            )
            b_key = agg["capture"].idxmax()
            b_params = by_key[b_key]
            b_epochs = scaled_count(int(round(float(agg.loc[b_key, "epoch"]))))

            selection[tag] = {
                "A_nb12_identical": {
                    "params": a_params,
                    "config_key": str(a_row["config_key"]),
                    "val30_realized": float(a_row["realized"]),
                    "best_epoch": int(a_row["epoch_realized"]),
                    "final_epochs": a_epochs,
                    "seeds": [GRID_SEEDS[0]],
                },
                "B_robust": {
                    "params": b_params,
                    "config_key": str(b_key),
                    "val30_capture": float(agg.loc[b_key, "capture"]),
                    "val30_realized": float(agg.loc[b_key, "realized"]),
                    "best_epoch": int(round(float(agg.loc[b_key, "epoch"]))),
                    "final_epochs": b_epochs,
                    "seeds": list(FINAL_SEEDS),
                },
                "grid": frame.to_dict("records"),
            }
            print(
                f"[dfl_ltr] {tag} A={a_row['config_key']} ep={a_epochs} | "
                f"B={b_key} ep={b_epochs}",
                flush=True,
            )

            for track, params, epochs, seeds in (
                ("A", a_params, a_epochs, [GRID_SEEDS[0]]),
                ("B", b_params, b_epochs, list(FINAL_SEEDS)),
            ):
                per_seed = []
                for seed in seeds:
                    ck = (route, family, config_key(params), epochs, seed)
                    if ck not in retrain_cache:
                        retrain_cache[ck] = retrain_fixed(
                            family,
                            n_features,
                            x_dev,
                            y_dev,
                            x_test,
                            route=route,
                            params=params,
                            num_epochs=epochs,
                            seed=seed,
                        )
                    per_seed.append(retrain_cache[ck])
                stacked = np.stack([p.reshape(-1) for p in per_seed])
                yhat = stacked.mean(axis=0) if len(seeds) > 1 else stacked[0]

                name = f"{route}_{family}_{track}"
                pred_frame[name] = yhat
                scored = score_predictions(test_times, y_true, yhat)
                daily = scored.pop("daily").copy()
                daily.insert(0, "model", name)
                daily_frames.append(daily)

                # per-seed realized spread, track B only
                seed_realized = []
                if len(seeds) > 1:
                    for p in per_seed:
                        s = score_predictions(test_times, y_true, p.reshape(-1))
                        seed_realized.append(s["realized_profit_mean"])

                leaderboard.append(
                    {
                        "model": name,
                        "route": route,
                        "family": family,
                        "track": track,
                        "config": config_key(params),
                        "n_seeds": len(seeds),
                        "final_epochs": epochs,
                        "train_days": DEV_DAYS,
                        "test_days": HOLD_OUT_DAYS,
                        "RMSE": float(mean_squared_error(y_true, yhat) ** 0.5),
                        "MAE": float(mean_absolute_error(y_true, yhat)),
                        "n_points": int(y_true.size),
                        "pred_profit_mean": scored["pred_profit_mean"],
                        "realized_profit_mean": scored["realized_profit_mean"],
                        "profit_gap": scored["profit_gap"],
                        "oracle_profit_mean": scored["oracle_profit_mean"],
                        "capture": scored["capture"],
                        "seed_realized_min": float(np.min(seed_realized)) if seed_realized else np.nan,
                        "seed_realized_max": float(np.max(seed_realized)) if seed_realized else np.nan,
                        "beats_money_line": bool(scored["realized_profit_mean"] > MONEY_LINE),
                        "status": "ok",
                    }
                )
                print(
                    f"[dfl_ltr] >>> {name}: realized={scored['realized_profit_mean']:.1f} "
                    f"capture={scored['capture']:.4f} rmse={leaderboard[-1]['RMSE']:.3f}",
                    flush=True,
                )

            pd.DataFrame(leaderboard).to_csv(output_dir / "leaderboard.csv", index=False)
            (output_dir / "selection.json").write_text(
                json.dumps(selection, indent=2, default=str), encoding="utf-8"
            )

    board = pd.DataFrame(leaderboard).sort_values("realized_profit_mean", ascending=False)
    board.to_csv(output_dir / "leaderboard.csv", index=False)
    pd.concat(daily_frames, ignore_index=True).to_csv(
        output_dir / "dispatch_daily.csv", index=False
    )
    pred_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "grid_runs.json").write_text(
        json.dumps(grid_records, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "protocol": "select_on_271_30_retrain_301_score_59",
                "holdout_not_used_for_selection": True,
                "tracks": {
                    "A": "nb12-identical: seed 42, config+epoch by 30-day realized mean",
                    "B": "robust: config by mean 30-day daily-mean capture over seeds 42-43, "
                         "scored as the mean prediction over retrain seeds 42-46",
                },
                "grid_seeds": list(GRID_SEEDS),
                "final_seeds": list(FINAL_SEEDS),
                "max_epochs": SEQ_MAX_EPOCHS,
                "patience": SEQ_PATIENCE,
                "epoch_transfer": "min(80, round(best_epoch * 301/271))",
                "feature_set": SEQUENCE_FEATURE_SET,
                "families": families,
                "routes": {r: ROUTES[r] for r in routes},
                "money_line": MONEY_LINE,
                "money_line_source": "nb12 lstm_dispatch realized 6301.331965",
                "phase_b_metrics_untouched": True,
                "v1_v2_rmse_columns_untouched": True,
                "not_a_new_champion_on_59": True,
                "seconds_total": round(time.time() - t_start, 1),
                "feature_audit": data["audit"],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"[dfl_ltr] done in {(time.time()-t_start)/60:.1f} min", flush=True)
    print(board[["model", "realized_profit_mean", "capture", "RMSE", "beats_money_line"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
