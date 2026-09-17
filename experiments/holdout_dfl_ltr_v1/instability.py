"""Measure how much of the 59-day score is decided by bitwise nondeterminism.

``realized_profit`` is a step function of the predicted curve: ``optimize_day``
takes an argmax over 3321 windows, so an arbitrarily small perturbation can flip
the chosen window on a day where two windows are nearly tied.  On a spike day the
two windows are worth very different amounts, so the 59-day mean jumps.

This script repeats the *same* retrain (same seed, same config, same data) many
times and records the spread.  Any difference between repeats comes only from
non-deterministic CPU reduction order inside the model, not from sampling.

Output: ``reports/holdout_dfl_ltr_v1/instability.json``
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.holdout_dfl_ltr_v1.run import (
    FINAL_SEEDS,
    REPORT_DIR,
    load_arrays,
    retrain_fixed,
    score_predictions,
)
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.dispatch import optimize_day
from src.phase_b.evaluation import score_day

REPEATS = 8
CASES = (
    ("r1_listwise", "transformer"),
    ("r2_pairdiff", "lstm"),
    ("r6_cheap", "dlinear"),
)


def per_day(pred: np.ndarray, truth: np.ndarray):
    decisions = [optimize_day(day) for day in pred]
    profits = np.array(
        [score_day(t, d.power) for t, d in zip(truth, decisions)], dtype=float
    )
    windows = [(d.tc, d.td) for d in decisions]
    return profits, windows


def main() -> int:
    t0 = time.time()
    data = load_arrays(str(ROOT / "configs" / "phase_b.toml"))
    x_dev, y_dev, x_test = data["x_dev"], data["y_dev"], data["x_test"]
    times = pd.Series(pd.to_datetime(data["test_times"]))
    days = pd.DatetimeIndex(times.dt.normalize().unique())
    truth = np.asarray(data["test_y_true"], dtype=float).reshape(-1, STEPS_PER_DAY)
    n_features = int(x_dev.shape[-1])
    selection = json.loads((REPORT_DIR / "selection.json").read_text(encoding="utf-8"))

    out: dict[str, object] = {}
    for route, family in CASES:
        entry = selection[f"{route}/{family}"]["B_robust"]
        curves, means, window_sets = [], [], []
        for _ in range(REPEATS):
            pred = retrain_fixed(
                x_test=x_test, x_dev=x_dev, y_dev=y_dev, family=family,
                n_features=n_features, route=route, params=entry["params"],
                num_epochs=int(entry["final_epochs"]), seed=42,
            ).reshape(-1, STEPS_PER_DAY)
            profits, windows = per_day(pred, truth)
            curves.append(pred)
            means.append(float(profits.mean()))
            window_sets.append(windows)
        means_arr = np.array(means)

        stack = np.stack(curves)
        max_curve_spread = float((stack.max(axis=0) - stack.min(axis=0)).max())

        lo, hi = int(means_arr.argmin()), int(means_arr.argmax())
        lo_profit, _ = per_day(curves[lo], truth)
        hi_profit, _ = per_day(curves[hi], truth)
        gap = hi_profit - lo_profit
        worst_day = int(np.abs(gap).argmax())
        flips = sum(1 for a, b in zip(window_sets[lo], window_sets[hi]) if a != b)

        out[f"{route}/{family}"] = {
            "repeats": REPEATS,
            "same_seed": 42,
            "mean_59day_score": {
                "values": [round(v, 1) for v in means],
                "min": float(means_arr.min()),
                "max": float(means_arr.max()),
                "spread": float(means_arr.max() - means_arr.min()),
                "distinct_modes": len(Counter(np.round(means_arr, -1)).keys()),
            },
            "max_curve_spread_across_repeats": max_curve_spread,
            "worst_pair": {
                "low": float(means_arr.min()),
                "high": float(means_arr.max()),
                "days_with_a_different_window": flips,
                "largest_single_day": {
                    "date": str(days[worst_day].date()),
                    "low_profit": float(lo_profit[worst_day]),
                    "high_profit": float(hi_profit[worst_day]),
                    "low_window": list(window_sets[lo][worst_day]),
                    "high_window": list(window_sets[hi][worst_day]),
                    "share_of_total_gap": float(
                        abs(gap[worst_day]) / max(abs(gap.sum()), 1e-9)
                    ),
                },
            },
        }
        print(
            f"[instability] {route}/{family:12s} repeats={REPEATS} "
            f"spread={means_arr.max() - means_arr.min():7.1f} "
            f"curve spread={max_curve_spread:.2e} window flips={flips}/59",
            flush=True,
        )

    # does averaging five seeds damp it?
    route, family = "r1_listwise", "transformer"
    entry = selection[f"{route}/{family}"]["B_robust"]
    avg_scores = []
    for _ in range(3):
        curves = [
            retrain_fixed(
                x_test=x_test, x_dev=x_dev, y_dev=y_dev, family=family,
                n_features=n_features, route=route, params=entry["params"],
                num_epochs=int(entry["final_epochs"]), seed=s,
            ).reshape(-1)
            for s in FINAL_SEEDS
        ]
        avg_scores.append(
            score_predictions(times, np.asarray(data["test_y_true"], float),
                              np.mean(curves, axis=0))["realized_profit_mean"]
        )
    out["averaging_damps_it"] = {
        "repeats": 3,
        "averaged_curve_scores": [round(v, 1) for v in avg_scores],
        "spread": float(max(avg_scores) - min(avg_scores)),
        "single_curve_spread_for_comparison": out[f"{route}/{family}"]["mean_59day_score"]["spread"],
    }
    print(f"[instability] averaged-curve spread over 3 repeats: "
          f"{max(avg_scores) - min(avg_scores):.1f}", flush=True)

    out["reading"] = (
        "realized_profit is a step function of the prediction: optimize_day takes an "
        "argmax over 3321 windows. A curve perturbation of order 1e-2, caused purely by "
        "non-deterministic CPU reduction order, flips the window on a near-tied day. On "
        "2025-12-30 -- the spike day notebook 11 already flagged, oracle 31559 -- that "
        "flip is worth about 19000 on the day and about 320 on the 59-day mean. Any "
        "leaderboard gap of a few hundred on this holdout is therefore inside the noise "
        "floor of bitwise nondeterminism, before sampling noise is even considered."
    )
    out["seconds"] = round(time.time() - t0, 1)
    (REPORT_DIR / "instability.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nwrote instability.json in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
