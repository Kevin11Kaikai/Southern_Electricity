"""Independently recompute the report's headline numbers from frozen predictions.

    python tools/independent_audit.py [--repo-root .]

This tool deliberately does not import report/shared/derive_records.py. It
rebuilds the block sums with np.convolve, re-settles every saved action against
the project solver in src/phase_b/dispatch.py, and compares the result with the
packaged records in report/shared/data.

It needs reports/ -- the frozen experiment outputs, which are NOT distributed
with this repository because they embed the evaluation-period target prices.
See DATA.md for how to obtain the source data and regenerate them. Without
reports/ the tool prints that guidance and exits 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _common import DATA, ROOT

SOURCES = ("holdout_rmse_v1", "holdout_rmse_v2", "holdout_seed_bag_v1",
           "holdout_dispatch_loss_v1", "holdout_dfl_bakeoff_v1")
HEAD = "listwise_transformer_mean"
REFERENCE = ("holdout_rmse_v2", "transformer_contextual")
BASELINE = ("holdout_rmse_v2", "lgb_baseline")
WINDOWS = np.array([(c, d) for c in range(81) for d in range(c + 8, 89)], dtype=int)


def block_sums(day_matrix: np.ndarray) -> np.ndarray:
    """Eight-slot rolling sums, computed independently of the report generator."""
    return np.array([np.convolve(row, np.ones(8), "valid") for row in day_matrix])


def window_values(day_matrix: np.ndarray) -> np.ndarray:
    blocks = block_sums(day_matrix)
    return blocks[:, WINDOWS[:, 1]] - blocks[:, WINDOWS[:, 0]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.repo_root.resolve()
    reports = root / "reports"

    if not reports.is_dir():
        print(f"reports/ not found under {root}.\n")
        print("The frozen experiment outputs are not distributed with this repository:")
        print("they embed the 59-day evaluation target prices, which belong to the")
        print("competition dataset. To run this audit:\n")
        print("  1. Obtain the source data (see DATA.md)")
        print("  2. Run the experiment pipeline described in DATA.md to produce reports/")
        print("  3. Re-run this tool\n")
        print("Everything else in this repository -- the figures, tables and both PDFs --")
        print("rebuilds without reports/ via: python tools/build_report.py")
        return 0

    sys.path.insert(0, str(root))
    from src.phase_b.dispatch import LEGAL_WINDOWS, optimize_day

    assert tuple(map(tuple, WINDOWS)) == tuple(LEGAL_WINDOWS) and len(WINDOWS) == 3321
    print(f"Action set verified: {len(WINDOWS)} active windows plus idle\n")

    rows, series, canonical = [], {}, None
    for name in SOURCES:
        folder = reports / name
        predictions = pd.read_parquet(folder / "predictions.parquet").sort_values("times")
        daily = pd.read_csv(folder / "dispatch_daily.csv")
        board = pd.read_csv(folder / "leaderboard.csv").set_index("model")
        times = pd.to_datetime(predictions["times"])
        assert len(predictions) == 5664 and times.groupby(times.dt.date).size().eq(96).all()
        truth = predictions["A"].to_numpy(float).reshape(59, 96)
        dates = times.iloc[::96].dt.strftime("%Y-%m-%d").tolist()
        if canonical is None:
            canonical = truth
        else:
            np.testing.assert_allclose(truth, canonical, rtol=0, atol=1e-12)
        oracle = np.maximum(window_values(truth).max(axis=1), 0) * 1000

        for model in predictions.columns.drop(["times", "A"]):
            curve = predictions[model].to_numpy(float).reshape(59, 96)
            actions = daily[daily["model"].eq(model)].set_index("date").reindex(dates)
            # Re-solve every day from the saved curve and settle against the true prices.
            realized = np.empty(59)
            mismatch = 0
            for i in range(59):
                decision = optimize_day(curve[i])
                realized[i] = float((truth[i] * decision.power).sum())
                saved = (actions["tc"].iloc[i], actions["td"].iloc[i])
                if not actions["idle"].astype(str).str.lower().iloc[i] in ("true", "1"):
                    if (decision.tc, decision.td) != (int(saved[0]), int(saved[1])):
                        mismatch += 1
            recorded = float(board.loc[model, "realized_profit_mean"])
            rows.append({"source": name, "model": model,
                         "independent_realized": float(realized.mean()),
                         "recorded_realized": recorded,
                         "abs_difference": abs(float(realized.mean()) - recorded),
                         "action_mismatch_days": mismatch})
            series[(name, model)] = pd.Series(realized, index=dates)

    table = pd.DataFrame(rows)
    print(f"Re-solved and re-settled {len(table)} saved prediction curves "
          f"across {len(SOURCES)} source directories.")
    print(f"  max |independent - recorded| realized mean : {table.abs_difference.max():.3e}")
    print(f"  days where the re-solved action differs    : {int(table.action_mismatch_days.sum())}\n")

    head = series[("holdout_seed_bag_v1", HEAD)]
    reference = series[REFERENCE]
    baseline = series[BASELINE]
    seeds = np.column_stack([series[("holdout_seed_bag_v1", f"listwise_transformer_seed{s}")]
                             for s in range(42, 47)])
    gain = head - reference
    computed = {
        "headline": head.mean(), "reference": reference.mean(), "baseline": baseline.mean(),
        "paired_gain": gain.mean(),
        "relative_gain_pct": 100 * gain.mean() / reference.mean(),
        "single_seed_mean": seeds.mean(),
        "ensemble_increment": (head.to_numpy() - seeds.mean(axis=1)).mean(),
        "nonensemble_remainder": seeds.mean() - reference.mean(),
        "worst_day_headline": head.min(), "worst_day_reference": reference.min(),
        "gross_winning_day_total": gain[gain > 0].sum(),
        "gross_losing_day_total": gain[gain < 0].sum(),
    }
    packaged = json.loads((DATA / "summary.json").read_text(encoding="utf-8"))

    print(f"{'quantity':<26}{'independent':>18}{'packaged':>18}{'difference':>14}")
    worst = 0.0
    for key, value in computed.items():
        other = packaged.get(key)
        if other is None:
            continue
        delta = float(value) - float(other)
        worst = max(worst, abs(delta))
        print(f"{key:<26}{float(value):>18.6f}{float(other):>18.6f}{delta:>14.2e}")
    print(f"\nLargest disagreement with the packaged records: {worst:.3e}")
    ok = worst < 1e-6 and table.abs_difference.max() < 1e-4 and table.action_mismatch_days.sum() == 0
    print("OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
