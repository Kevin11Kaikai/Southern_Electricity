"""S1 / R5: decision-weighted convex combination of heterogeneous curves.

Greedy forward selection with replacement (Caruana et al. 2004), the ensembling
recipe that wins tabular competitions, but scored with the *task* metric rather
than RMSE: every candidate step is judged by the 30-day daily-mean capture that
``optimize_day`` produces after locking the window.

Why the base curves are refit here rather than read from the frozen v1/v2
parquets: those artefacts only contain the 59 holdout days, so they carry no
30-day validation predictions and cannot be weighted without touching the
holdout.  The anchors are therefore refit under this campaign's own protocol
(271 fit -> weights on the last 30 -> 301 retrain -> score 59 once).  This is
stated in the report; the ensemble is over refits, not over the frozen curves.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.holdout_dfl_ltr_v1.run import (
    DEV_DAYS,
    FIT_DAYS,
    FINAL_SEEDS,
    GRID_SEEDS,
    REPORT_DIR,
    day_oracles,
    load_arrays,
    retrain_fixed,
    score_predictions,
    train_grid_run,
)
from src.phase_b.advanced_models import make_ridge_pipeline
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.dispatch import optimize_day
from src.phase_b.evaluation import score_day

GREEDY_STEPS = 60


def daily_mean_capture(y_days: np.ndarray, yhat_days: np.ndarray, oracle: np.ndarray) -> float:
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
    return float(np.nanmean(realized / safe))


# ------------------------------------------------------------- base learners


def _flat(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1, x.shape[-1])


def fit_ridge(x_train, y_train, x_apply, *, alpha: float = 1.0) -> np.ndarray:
    pipeline = make_ridge_pipeline(alpha=alpha)
    pipeline.fit(_flat(x_train), np.asarray(y_train, dtype=float).reshape(-1))
    return np.asarray(pipeline.predict(_flat(x_apply)), dtype=float).reshape(
        -1, STEPS_PER_DAY
    )


def fit_hgb(x_train, y_train, x_apply, *, seed: int = 42) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6, random_state=seed
    )
    model.fit(_flat(x_train), np.asarray(y_train, dtype=float).reshape(-1))
    return np.asarray(model.predict(_flat(x_apply)), dtype=float).reshape(
        -1, STEPS_PER_DAY
    )


def fit_lgb(x_train, y_train, x_apply, *, seed: int = 42) -> np.ndarray | None:
    try:
        import lightgbm as lgb
    except ImportError:
        return None

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(_flat(x_train), np.asarray(y_train, dtype=float).reshape(-1))
    return np.asarray(model.predict(_flat(x_apply)), dtype=float).reshape(
        -1, STEPS_PER_DAY
    )


# ----------------------------------------------------------------- ensembling


def greedy_forward(
    candidates: dict[str, np.ndarray],
    y_val: np.ndarray,
    oracle: np.ndarray,
    *,
    steps: int = GREEDY_STEPS,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Caruana-style greedy selection with replacement on 30-day capture."""

    names = list(candidates)
    running = np.zeros_like(next(iter(candidates.values())))
    counts = {name: 0 for name in names}
    trace: list[dict[str, Any]] = []
    total = 0
    best_overall, best_counts = -np.inf, dict(counts)

    for step in range(steps):
        scores = {}
        for name in names:
            blended = (running + candidates[name]) / (total + 1)
            scores[name] = daily_mean_capture(y_val, blended, oracle)
        pick = max(scores, key=scores.get)
        running = running + candidates[pick]
        counts[pick] += 1
        total += 1
        value = scores[pick]
        trace.append({"step": step + 1, "pick": pick, "val30_capture": value})
        if value > best_overall:
            best_overall, best_counts = value, dict(counts)

    weights = {k: v / max(1, sum(best_counts.values())) for k, v in best_counts.items()}
    return {k: v for k, v in weights.items() if v > 0}, trace


def main() -> int:
    t0 = time.time()
    data = load_arrays(str(ROOT / "configs" / "phase_b.toml"))
    x_dev, y_dev, x_test = data["x_dev"], data["y_dev"], data["x_test"]
    test_times = pd.Series(pd.to_datetime(data["test_times"]))
    y_true = np.asarray(data["test_y_true"], dtype=float)
    n_features = int(x_dev.shape[-1])
    x_fit, y_fit = x_dev[:FIT_DAYS], y_dev[:FIT_DAYS]
    x_val, y_val = x_dev[FIT_DAYS:], y_dev[FIT_DAYS:]
    val_oracle = day_oracles(y_val)

    val_curves: dict[str, np.ndarray] = {}
    test_curves: dict[str, np.ndarray] = {}

    print("[R5] refitting tabular anchors", flush=True)
    tabular: dict[str, Callable] = {"ridge_a1": fit_ridge, "hgb": fit_hgb, "lgb": fit_lgb}
    for name, fn in tabular.items():
        val_pred = fn(x_fit, y_fit, x_val)
        if val_pred is None:
            print(f"[R5]   {name} unavailable, skipped", flush=True)
            continue
        val_curves[name] = val_pred
        test_curves[name] = fn(x_dev, y_dev, x_test)
        print(
            f"[R5]   {name:10s} val30 capture {daily_mean_capture(y_val, val_curves[name], val_oracle):.4f}",
            flush=True,
        )

    selection_path = REPORT_DIR / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        best_per_family: dict[str, tuple[str, dict, int, float]] = {}
        for tag, payload in selection.items():
            route, family = tag.split("/")
            entry = payload["B_robust"]
            score = float(entry["val30_capture"])
            if family not in best_per_family or score > best_per_family[family][3]:
                best_per_family[family] = (
                    route,
                    entry["params"],
                    int(entry["final_epochs"]),
                    score,
                )
        print("[R5] adding the best sequence head per family (chosen on val30 capture)", flush=True)
        for family, (route, params, epochs, score) in best_per_family.items():
            name = f"{family}_{route.split('_')[0]}"
            # Fit on the 271 days only and predict the 30 selection days, so the
            # ensemble weights never see a curve that was trained on them.
            val_pred = np.mean(
                [
                    retrain_fixed(
                        family, n_features, x_fit, y_fit, x_val,
                        route=route, params=params,
                        num_epochs=max(1, int(round(epochs * FIT_DAYS / DEV_DAYS))),
                        seed=seed,
                    )
                    for seed in GRID_SEEDS
                ],
                axis=0,
            )
            test_pred = np.mean(
                [
                    retrain_fixed(
                        family, n_features, x_dev, y_dev, x_test,
                        route=route, params=params, num_epochs=epochs, seed=seed,
                    )
                    for seed in FINAL_SEEDS
                ],
                axis=0,
            )
            val_curves[name] = val_pred.reshape(-1, STEPS_PER_DAY)
            test_curves[name] = test_pred.reshape(-1, STEPS_PER_DAY)
            print(
                f"[R5]   {name:16s} ({route}) val30 capture "
                f"{daily_mean_capture(y_val, val_curves[name], val_oracle):.4f}",
                flush=True,
            )
    else:
        print("[R5] selection.json not found; ensembling tabular anchors only", flush=True)

    print(f"[R5] greedy forward selection over {len(val_curves)} curves", flush=True)
    weights, trace = greedy_forward(val_curves, y_val, val_oracle)
    print(f"[R5] weights {json.dumps({k: round(v, 3) for k, v in weights.items()})}", flush=True)

    blended_test = sum(weights[k] * test_curves[k] for k in weights)
    yhat = np.asarray(blended_test, dtype=float).reshape(-1)
    scored = score_predictions(test_times, y_true, yhat)
    daily = scored.pop("daily").copy()
    daily.insert(0, "model", "r5_ensemble")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "ensemble_weights.json").write_text(
        json.dumps(
            {
                "weights": weights,
                "selection_metric": "30-day daily-mean capture",
                "method": "Caruana greedy forward selection with replacement",
                "steps": GREEDY_STEPS,
                "base_curves": sorted(val_curves),
                "val30_capture_per_curve": {
                    k: daily_mean_capture(y_val, v, val_oracle) for k, v in val_curves.items()
                },
                "note": (
                    "base curves are refit under this campaign's 271/30/301/59 protocol; "
                    "the frozen v1/v2 parquets hold holdout days only and carry no "
                    "validation predictions"
                ),
                "trace": trace,
                "holdout": {k: v for k, v in scored.items()},
                "seconds": round(time.time() - t0, 1),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    daily.to_csv(REPORT_DIR / "ensemble_daily.csv", index=False)
    pd.DataFrame({"times": test_times, "A": y_true, "r5_ensemble": yhat}).to_parquet(
        REPORT_DIR / "ensemble_predictions.parquet", index=False
    )
    print(
        f"[R5] holdout realized {scored['realized_profit_mean']:.1f} "
        f"capture {scored['capture']:.4f} ({(time.time()-t0)/60:.1f} min)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
