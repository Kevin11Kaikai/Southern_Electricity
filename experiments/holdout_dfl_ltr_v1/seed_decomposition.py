"""Separate two things the B track bundles together: the loss, and curve averaging.

A B-track row is the score of the *pointwise mean of five predicted curves*.  That
conflates two effects:

  1. how good one curve from this loss is        -> mean of the five single-seed scores
  2. how much averaging five curves is worth     -> averaged-curve score minus that mean

Only (1) is a property of the loss.  Effect (2) is available to any recipe,
including the nb12 incumbent, which was scored as a single curve.  Comparing a
5-seed average against a single-curve incumbent therefore overstates the loss.

Also records run-to-run reproducibility: the compact transformer is not bitwise
deterministic on CPU even at a fixed seed, so exact figures carry a small jitter.

Output: ``reports/holdout_dfl_ltr_v1/seed_decomposition.json``
"""

from __future__ import annotations

import json
import sys
import time
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

MONEY_LINE = 6301.331965
INCUMBENT_IS_A_SINGLE_CURVE = True


def main() -> int:
    t0 = time.time()
    data = load_arrays(str(ROOT / "configs" / "phase_b.toml"))
    x_dev, y_dev, x_test = data["x_dev"], data["y_dev"], data["x_test"]
    times = pd.Series(pd.to_datetime(data["test_times"]))
    y_true = np.asarray(data["test_y_true"], dtype=float)
    n_features = int(x_dev.shape[-1])

    selection = json.loads((REPORT_DIR / "selection.json").read_text(encoding="utf-8"))
    rows = []
    for tag, payload in selection.items():
        route, family = tag.split("/")
        entry = payload["B_robust"]
        params, epochs = entry["params"], int(entry["final_epochs"])
        curves = [
            retrain_fixed(
                family, n_features, x_dev, y_dev, x_test,
                route=route, params=params, num_epochs=epochs, seed=seed,
            ).reshape(-1)
            for seed in FINAL_SEEDS
        ]
        singles = np.array(
            [score_predictions(times, y_true, c)["realized_profit_mean"] for c in curves]
        )
        averaged = score_predictions(times, y_true, np.mean(curves, axis=0))
        rows.append(
            {
                "route": route,
                "family": family,
                "config": entry["config_key"],
                "epochs": epochs,
                "single_curve_mean": float(singles.mean()),
                "single_curve_std": float(singles.std(ddof=1)),
                "single_curve_min": float(singles.min()),
                "single_curve_max": float(singles.max()),
                "averaged_curve": float(averaged["realized_profit_mean"]),
                "averaging_gain": float(averaged["realized_profit_mean"] - singles.mean()),
                "averaged_beats_best_seed_by": float(
                    averaged["realized_profit_mean"] - singles.max()
                ),
                "single_curve_vs_incumbent": float(singles.mean() - MONEY_LINE),
                "per_seed": dict(zip(map(str, FINAL_SEEDS), singles.round(2).tolist())),
            }
        )
        print(
            f"[decomp] {tag:26s} single {singles.mean():7.1f} (sd {singles.std(ddof=1):5.1f}) "
            f"averaged {averaged['realized_profit_mean']:7.1f} "
            f"gain {averaged['realized_profit_mean'] - singles.mean():+7.1f}",
            flush=True,
        )

    # run-to-run reproducibility at a fixed seed
    repro = {}
    for family, route in (("transformer", "r1_listwise"), ("lstm", "r2_pairdiff"), ("dlinear", "r6_cheap")):
        entry = selection[f"{route}/{family}"]["B_robust"]
        twice = [
            retrain_fixed(
                family, n_features, x_dev, y_dev, x_test,
                route=route, params=entry["params"],
                num_epochs=int(entry["final_epochs"]), seed=42,
            ).reshape(-1)
            for _ in range(2)
        ]
        scores = [score_predictions(times, y_true, c)["realized_profit_mean"] for c in twice]
        repro[family] = {
            "max_abs_curve_diff": float(np.abs(twice[0] - twice[1]).max()),
            "realized_run1": scores[0],
            "realized_run2": scores[1],
            "realized_diff": float(scores[0] - scores[1]),
        }
        print(f"[repro] {family}: curve diff {repro[family]['max_abs_curve_diff']:.2e} "
              f"realized diff {repro[family]['realized_diff']:+.1f}", flush=True)

    frame = pd.DataFrame(rows).sort_values("single_curve_mean", ascending=False)
    payload = {
        "note": (
            "single_curve_mean isolates the loss; averaging_gain is the ensembling "
            "effect, which the nb12 incumbent (a single curve) never received"
        ),
        "money_line": MONEY_LINE,
        "incumbent_is_a_single_curve": INCUMBENT_IS_A_SINGLE_CURVE,
        "rows": frame.to_dict("records"),
        "reproducibility_same_seed_two_runs": repro,
        "seconds": round(time.time() - t0, 1),
    }
    (REPORT_DIR / "seed_decomposition.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print()
    print(frame[["route", "family", "single_curve_mean", "averaged_curve",
                 "averaging_gain", "single_curve_vs_incumbent"]].round(1).to_string(index=False))
    print(f"\nwrote seed_decomposition.json in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
