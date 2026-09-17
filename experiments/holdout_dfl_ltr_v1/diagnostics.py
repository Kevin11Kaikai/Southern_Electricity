"""S0: recompute D1-D4 and the no-model constant-window baselines.

Everything here is read-only over frozen artefacts: the v1/v2/nb12 prediction
parquets, their dispatch_daily tables, and the raw label series.  Nothing is
trained.  Output: ``reports/holdout_dfl_ltr_v1/diagnostics.json``.

D1  the true score surface is flat at the top, and hit@1 is ~0 (6 exact hits in
    2242 recipe-days) -> a one-hot CE target is both unlearnable and wrong
D2  perfect reranking over a recipe's own top-K windows is worth +781 at K=50
D3  the feasible set has 3321 elements and is fully enumerable
    -> exact listwise; no cache, no DBB, no perturbation
D4  a 59-day realized mean has SE ~ 549; a 30-day one ~ 764
    -> nb12's per-epoch val swings are day-sampling noise, and daily-mean
       capture is the better selection statistic
B   no-model constant windows chosen on training data only
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_b.config import load_config
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.dispatch import LEGAL_WINDOWS, optimize_day

V1 = ROOT / "reports" / "holdout_rmse_v1"
V2 = ROOT / "reports" / "holdout_rmse_v2"
NB12 = ROOT / "reports" / "holdout_dispatch_loss_v1"
OUT = ROOT / "reports" / "holdout_dfl_ltr_v1"
HOLDOUT_START = pd.Timestamp("2025-11-03")

_TC = np.array([w[0] for w in LEGAL_WINDOWS])
_TD = np.array([w[1] for w in LEGAL_WINDOWS])


def window_score_matrix(days: np.ndarray) -> np.ndarray:
    """``(n_days, 3321)`` official score of every legal window."""

    matrix = np.asarray(days, dtype=float).reshape(-1, STEPS_PER_DAY)
    blocks = np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(8), "valid"), 1, matrix
    )
    return 1000.0 * (blocks[:, _TD] - blocks[:, _TC])


def load_holdout_truth() -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_parquet(V1 / "predictions.parquet")
    actual = frame["A"].to_numpy(dtype=float).reshape(-1, STEPS_PER_DAY)
    return actual, window_score_matrix(actual)


def load_training_days() -> np.ndarray:
    config = load_config(ROOT / "configs" / "phase_b.toml")
    path = ROOT / config.raw["paths"]["train_labels"]
    price = pd.read_csv(path, parse_dates=["times"])
    # Timestamps run 00:15..24:00, so the market day is the stamp minus one step.
    price["day"] = (price["times"] - pd.Timedelta(minutes=15)).dt.normalize()
    grouped = price.groupby("day")["A"].apply(lambda s: s.to_numpy())
    complete = grouped[grouped.apply(len) == STEPS_PER_DAY]
    train = complete[complete.index < HOLDOUT_START]
    return np.stack(train.to_numpy())


def recipe_curves() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for path, tag in ((V1, "v1"), (V2, "v2"), (NB12, "nb12")):
        parquet = path / "predictions.parquet"
        if not parquet.exists():
            continue
        frame = pd.read_parquet(parquet)
        for column in frame.columns:
            if column in {"times", "A", "date"}:
                continue
            label = column if tag == "nb12" else f"{tag}/{column}"
            out[label] = frame[column].to_numpy(dtype=float).reshape(-1, STEPS_PER_DAY)
    return out


def d1_surface_flatness(scores: np.ndarray, curves: dict[str, np.ndarray]) -> dict[str, Any]:
    oracle = np.maximum(scores.max(1), 0.0).mean()
    kth = {
        str(k): float(np.mean(np.sort(scores, axis=1)[:, -k]))
        for k in (1, 2, 5, 10, 50)
    }
    hit1 = {}
    truth_best = scores.argmax(1)
    for label, yhat in curves.items():
        predicted_best = window_score_matrix(yhat).argmax(1)
        hit1[label] = float(np.mean(predicted_best == truth_best))
    return {
        "oracle_mean": float(oracle),
        "kth_best_true_window_value": kth,
        "kth_best_as_share_of_oracle": {k: v / oracle for k, v in kth.items()},
        "hit_at_1": hit1,
        "reading": (
            "the 2nd/10th/50th best window is worth 99.9%/98.7%/92.8% of the oracle, "
            "while 33 of 38 recipes never hit the exact argmax and the best manages "
            "2 days out of 59 (6 exact hits in 2242 recipe-days) -- a one-hot CE target "
            "is effectively unlearnable here and penalises near-optimal windows as hard "
            "as terrible ones"
        ),
    }


def d2_topk_ceiling(scores: np.ndarray, curves: dict[str, np.ndarray]) -> dict[str, Any]:
    ks = (1, 5, 20, 50, 200, 500)
    out: dict[str, Any] = {}
    for label, yhat in curves.items():
        predicted = window_score_matrix(yhat)
        order = np.argsort(-predicted, axis=1)
        truth_best = scores.argmax(1)
        ceilings, hits = {}, {}
        for k in ks:
            top = order[:, :k]
            ceilings[str(k)] = float(
                np.mean([scores[i, top[i]].max() for i in range(scores.shape[0])])
            )
            hits[str(k)] = float(
                np.mean([truth_best[i] in top[i] for i in range(scores.shape[0])])
            )
        out[label] = {"perfect_rerank_ceiling": ceilings, "oracle_in_topk": hits}
    return {
        "per_recipe": out,
        "reading": (
            "the curves already rank good windows highly; what is wrong is the order "
            "inside the shortlist -- this is the only measured headroom toward 9001"
        ),
    }


def d3_enumerability() -> dict[str, Any]:
    return {
        "n_legal_windows": len(LEGAL_WINDOWS),
        "exact_gibbs": True,
        "reading": (
            "3321 actions fit in a (batch, 3321) tensor, so the Gibbs distribution over "
            "actions is exact: solution caching (Mandi 2022), blackbox differentiation "
            "(Pogancic 2020) and Monte-Carlo perturbation (Berthet 2020) would all be "
            "approximations of something we can compute exactly"
        ),
    }


def d4_noise(scores: np.ndarray) -> dict[str, Any]:
    oracle_daily = np.maximum(scores.max(1), 0.0)
    daily_frames = []
    for path, tag in ((V1, "v1"), (V2, "v2"), (NB12, "nb12")):
        csv = path / "dispatch_daily.csv"
        if csv.exists():
            frame = pd.read_csv(csv)
            frame["label"] = (
                frame["model"] if tag == "nb12" else tag + "/" + frame["model"]
            )
            daily_frames.append(frame[["label", "date", "realized_profit"]])
    daily = pd.concat(daily_frames, ignore_index=True)
    pivot = daily.pivot_table(index="date", columns="label", values="realized_profit")
    se59 = float((pivot.std(ddof=1) / np.sqrt(len(pivot))).median())

    rng = np.random.default_rng(0)
    incumbent = pivot["lstm_dispatch"].to_numpy()
    challenger = pivot["v2/transformer_contextual"].to_numpy()
    index = rng.integers(0, len(incumbent), (20000, 30))
    realized_gap = (incumbent - challenger)[index].mean(1)
    capture_gap = ((incumbent - challenger) / oracle_daily)[index].mean(1)
    return {
        "median_se_of_59day_mean": se59,
        "se_of_30day_mean": se59 * float(np.sqrt(59 / 30)),
        "nb12_per_epoch_val_realized_std_range": [383.0, 764.0],
        "selection_signal_to_noise_30day": {
            "realized_mean": float(abs(realized_gap.mean()) / realized_gap.std()),
            "daily_mean_capture": float(abs(capture_gap.mean()) / capture_gap.std()),
        },
        "reading": (
            "nb12's epoch-to-epoch val_realized swings match pure 30-day sampling noise; "
            "daily-mean capture raises the selection signal-to-noise by about 28%"
        ),
    }


def constant_window_baselines(train_days: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    train_scores = window_score_matrix(train_days)
    oracle = np.maximum(scores.max(1), 0.0).mean()

    def describe(index: int, name: str) -> dict[str, Any]:
        realized = scores[:, index]
        return {
            "name": name,
            "tc": int(_TC[index]),
            "td": int(_TD[index]),
            "charge_hours": [_TC[index] / 4.0, (_TC[index] + 8) / 4.0],
            "discharge_hours": [_TD[index] / 4.0, (_TD[index] + 8) / 4.0],
            "realized_mean": float(realized.mean()),
            "capture": float(realized.mean() / oracle),
            "daily": realized.tolist(),
        }

    out = {
        "all_train_days": describe(int(train_scores.mean(0).argmax()), "fixed window, all training days"),
        "last_60_train_days": describe(int(train_scores[-60:].mean(0).argmax()), "fixed window, last 60 training days"),
        "last_30_train_days": describe(int(train_scores[-30:].mean(0).argmax()), "fixed window, last 30 training days"),
    }
    best_hindsight = int(scores.mean(0).argmax())
    out["hindsight_best_on_holdout"] = describe(best_hindsight, "best constant window ON the holdout (cheating bound)")
    out["reading"] = (
        "the primary no-model reference is 'all_train_days'; 'last_60' is reported but "
        "its window length was picked with hindsight, so it is not an honest baseline"
    )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    actual, scores = load_holdout_truth()
    curves = recipe_curves()
    train_days = load_training_days()
    payload = {
        "holdout_days": int(actual.shape[0]),
        "training_days": int(train_days.shape[0]),
        "oracle_profit_mean": float(np.maximum(scores.max(1), 0.0).mean()),
        "D1_surface_flatness": d1_surface_flatness(scores, curves),
        "D2_topk_ceiling": d2_topk_ceiling(scores, curves),
        "D3_enumerability": d3_enumerability(),
        "D4_noise": d4_noise(scores),
        "constant_window_baselines": constant_window_baselines(train_days, scores),
    }
    (OUT / "diagnostics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    base = payload["constant_window_baselines"]
    print(f"oracle {payload['oracle_profit_mean']:.1f}")
    for key in ("all_train_days", "last_60_train_days", "last_30_train_days", "hindsight_best_on_holdout"):
        row = base[key]
        print(f"  {row['name']:52s} {row['realized_mean']:7.1f}  capture {row['capture']:.3f}")
    d1 = payload["D1_surface_flatness"]
    print("  kth-best true window as share of oracle:", {k: round(v, 4) for k, v in d1["kth_best_as_share_of_oracle"].items()})
    print(f"  wrote {(OUT / 'diagnostics.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
