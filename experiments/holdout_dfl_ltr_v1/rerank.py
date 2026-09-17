"""S6 / R4: two-stage dispatch -- strong curve, then a shortlist reranker.

Motivated by D2: ``hit@1`` against the true oracle window is ~0 across the
campaign (6 exact hits in 2242 recipe-days), yet the best true window inside a
recipe's own top-50 predicted windows is worth 7082 on average against 6301
realized.  The curve already puts good windows near the front; the order inside
the shortlist is what is wrong.

Stage 1  a curve scores all 3321 windows and keeps the top K.
Stage 2  a ridge model rescores those K candidates from level-free features and
         the argmax of the rescore is dispatched (idle still applies when the
         stage-1 score of the pick is non-positive).

Stacking discipline: the reranker is trained on **out-of-fold** stage-1 curves
over the 271 fit days (5 contiguous day blocks), so its shortlists at training
time are as noisy as the ones it will face.  Every hyper-parameter -- stage-1
source, K, ridge alpha -- is chosen on the 30 selection days by daily-mean
capture.  The holdout is scored once.
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

from experiments.holdout_dfl_ltr_v1.ensemble import fit_hgb, fit_ridge
from experiments.holdout_dfl_ltr_v1.run import (
    DEV_DAYS,
    FIT_DAYS,
    FINAL_SEEDS,
    REPORT_DIR,
    day_oracles,
    load_arrays,
    retrain_fixed,
    score_predictions,
)
from src.phase_b.contracts import BLOCK_STEPS, STEPS_PER_DAY
from src.phase_b.dispatch import LEGAL_WINDOWS

_TC = np.array([w[0] for w in LEGAL_WINDOWS])
_TD = np.array([w[1] for w in LEGAL_WINDOWS])
K_GRID = (20, 50, 200)
ALPHA_GRID = (1.0, 10.0, 100.0)
# How far the rescore is allowed to move away from stage 1.  ``blend = 0`` is
# "decline to rerank" -- the grid must be able to choose it, otherwise a
# reranker that is worse than its own stage 1 still gets deployed.
BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
N_FOLDS = 5


def block_sums(days: np.ndarray) -> np.ndarray:
    matrix = np.asarray(days, dtype=float).reshape(-1, STEPS_PER_DAY)
    return np.apply_along_axis(
        lambda row: np.convolve(row, np.ones(BLOCK_STEPS), "valid"), 1, matrix
    )


def window_scores(days: np.ndarray) -> np.ndarray:
    blocks = block_sums(days)
    return 1000.0 * (blocks[:, _TD] - blocks[:, _TC])


def candidate_features(pred_days: np.ndarray, shortlist: np.ndarray) -> np.ndarray:
    """Level-free features for each (day, candidate window) pair.

    The official score is invariant to ``p -> a*p + b`` for ``a > 0``, so the
    daily mean and the absolute amplitude carry no decision information.  Block
    sums are therefore centred and divided by their own daily spread.
    """

    pred = np.asarray(pred_days, dtype=float).reshape(-1, STEPS_PER_DAY)
    blocks = block_sums(pred)
    centred = blocks - blocks.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    z = centred / np.where(spread > 1e-9, spread, 1.0)

    n_days, k = shortlist.shape
    rows = np.arange(n_days)[:, None]
    tc = _TC[shortlist].astype(float)
    td = _TD[shortlist].astype(float)
    z_charge = z[rows, _TC[shortlist]]
    z_discharge = z[rows, _TD[shortlist]]
    raw_rank = np.argsort(np.argsort(-window_scores(pred), axis=1), axis=1)[rows, shortlist]

    return np.stack(
        [
            z_discharge - z_charge,   # normalised predicted spread of this window
            z_charge,                 # how deep the trough is, in daily units
            z_discharge,              # how high the peak is
            tc / 96.0,                # clock position of the charge block
            td / 96.0,                # clock position of the discharge block
            (td - tc) / 96.0,         # separation between them
            raw_rank / 3321.0,        # where stage 1 ranked this candidate
        ],
        axis=-1,
    )


def shortlist_of(pred_days: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-window_scores(pred_days), axis=1)[:, :k]


def training_pairs(pred_days: np.ndarray, true_days: np.ndarray, k: int):
    shortlist = shortlist_of(pred_days, k)
    rows = np.arange(shortlist.shape[0])[:, None]
    target = window_scores(true_days)[rows, shortlist]
    scale = np.abs(target).max(axis=1, keepdims=True)
    target = target / np.where(scale > 1e-9, scale, 1.0)
    return candidate_features(pred_days, shortlist), target, shortlist


def fit_reranker(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    x = features.reshape(-1, features.shape[-1])
    x = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)
    y = target.reshape(-1)
    gram = x.T @ x + alpha * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def _zrows(values: np.ndarray) -> np.ndarray:
    centred = values - values.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 1e-12, spread, 1.0)


def apply_reranker(
    coef: np.ndarray,
    pred_days: np.ndarray,
    true_days: np.ndarray,
    k: int,
    blend: float = 1.0,
) -> np.ndarray:
    """Per-day official score after rescoring the top-k shortlist.

    The deployed score is ``blend * z(rerank) + (1 - blend) * z(-stage1 rank)``,
    so ``blend = 0`` reproduces stage 1 exactly and the hyper-parameter search can
    decline to rerank.
    """

    shortlist = shortlist_of(pred_days, k)
    rows = np.arange(shortlist.shape[0])
    features = candidate_features(pred_days, shortlist)
    flat = features.reshape(-1, features.shape[-1])
    flat = np.concatenate([flat, np.ones((flat.shape[0], 1))], axis=1)
    rescore = _zrows((flat @ coef).reshape(shortlist.shape))
    # shortlist is already ordered by stage-1 score, so position == stage-1 rank
    stage1_pref = _zrows(
        -np.tile(np.arange(shortlist.shape[1], dtype=float), (shortlist.shape[0], 1))
    )
    combined = float(blend) * rescore + (1.0 - float(blend)) * stage1_pref
    picked = shortlist[rows, combined.argmax(axis=1)]
    realized = window_scores(true_days)[rows, picked]
    stage1 = window_scores(pred_days)[rows, picked]
    return np.where(stage1 > 0, realized, 0.0)


def capture_of(realized: np.ndarray, oracle: np.ndarray) -> float:
    safe = np.where(np.abs(oracle) > 1e-9, oracle, np.nan)
    return float(np.nanmean(realized / safe))


def oof_curves(fn: Callable, x_fit: np.ndarray, y_fit: np.ndarray) -> np.ndarray:
    """Out-of-fold stage-1 predictions over the 271 fit days, 5 contiguous blocks."""

    n = x_fit.shape[0]
    out = np.zeros((n, STEPS_PER_DAY), dtype=float)
    bounds = np.linspace(0, n, N_FOLDS + 1).astype(int)
    for i in range(N_FOLDS):
        lo, hi = bounds[i], bounds[i + 1]
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        out[lo:hi] = np.asarray(
            fn(x_fit[mask], y_fit[mask], x_fit[lo:hi]), dtype=float
        ).reshape(-1, STEPS_PER_DAY)
    return out


def main() -> int:
    t0 = time.time()
    data = load_arrays(str(ROOT / "configs" / "phase_b.toml"))
    x_dev, y_dev, x_test = data["x_dev"], data["y_dev"], data["x_test"]
    test_times = pd.Series(pd.to_datetime(data["test_times"]))
    y_true = np.asarray(data["test_y_true"], dtype=float)
    y_test_days = y_true.reshape(-1, STEPS_PER_DAY)
    n_features = int(x_dev.shape[-1])
    x_fit, y_fit = x_dev[:FIT_DAYS], y_dev[:FIT_DAYS]
    x_val, y_val = x_dev[FIT_DAYS:], y_dev[FIT_DAYS:]
    val_oracle = day_oracles(y_val)
    test_oracle = day_oracles(y_test_days)

    sources: dict[str, dict[str, np.ndarray]] = {}
    print("[R4] building stage-1 sources (oof / val30 / holdout)", flush=True)
    for name, fn in (("ridge_a1", fit_ridge), ("hgb", fit_hgb)):
        sources[name] = {
            "oof": oof_curves(fn, x_fit, y_fit),
            "val": np.asarray(fn(x_fit, y_fit, x_val), dtype=float).reshape(-1, STEPS_PER_DAY),
            "test": np.asarray(fn(x_dev, y_dev, x_test), dtype=float).reshape(-1, STEPS_PER_DAY),
        }
        print(f"[R4]   {name} ready", flush=True)

    selection_path = REPORT_DIR / "selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        best: dict[str, tuple[str, dict, int, float]] = {}
        for tag, payload in selection.items():
            route, family = tag.split("/")
            entry = payload["B_robust"]
            score = float(entry["val30_capture"])
            if family not in best or score > best[family][3]:
                best[family] = (route, entry["params"], int(entry["final_epochs"]), score)
        for family, (route, params, epochs, _s) in best.items():
            fit_epochs = max(1, int(round(epochs * FIT_DAYS / DEV_DAYS)))
            seq = lambda xa, ya, xb, _f=family, _r=route, _p=params, _e=fit_epochs: retrain_fixed(
                _f, n_features, xa, ya, xb, route=_r, params=_p, num_epochs=_e, seed=42
            )
            name = f"{family}_{route.split('_')[0]}"
            sources[name] = {
                "oof": oof_curves(seq, x_fit, y_fit),
                "val": np.mean(
                    [
                        retrain_fixed(family, n_features, x_fit, y_fit, x_val,
                                      route=route, params=params,
                                      num_epochs=fit_epochs, seed=s)
                        for s in FINAL_SEEDS[:2]
                    ],
                    axis=0,
                ).reshape(-1, STEPS_PER_DAY),
                "test": np.mean(
                    [
                        retrain_fixed(family, n_features, x_dev, y_dev, x_test,
                                      route=route, params=params,
                                      num_epochs=epochs, seed=s)
                        for s in FINAL_SEEDS
                    ],
                    axis=0,
                ).reshape(-1, STEPS_PER_DAY),
            }
            print(f"[R4]   {name} ({route}, {epochs} ep) ready", flush=True)
    else:
        print("[R4] selection.json missing; tabular stage-1 sources only", flush=True)

    print(
        f"[R4] grid: {len(sources)} sources x {len(K_GRID)} K x {len(ALPHA_GRID)} alpha "
        f"x {len(BLEND_GRID)} blend",
        flush=True,
    )
    grid: list[dict[str, Any]] = []
    fitted: dict[tuple, np.ndarray] = {}
    for name, curves in sources.items():
        base_val = apply_reranker(np.zeros(8), curves["val"], y_val, 1, blend=0.0)
        base_capture = capture_of(base_val, val_oracle)
        for k in K_GRID:
            features_tr, target_tr, _ = training_pairs(curves["oof"], y_fit, k)
            for alpha in ALPHA_GRID:
                coef = fit_reranker(features_tr, target_tr, alpha)
                fitted[(name, k, alpha)] = coef
                for blend in BLEND_GRID:
                    realized = apply_reranker(coef, curves["val"], y_val, k, blend=blend)
                    grid.append(
                        {
                            "source": name,
                            "k": k,
                            "alpha": alpha,
                            "blend": blend,
                            "val30_capture": capture_of(realized, val_oracle),
                            "val30_realized": float(realized.mean()),
                            "stage1_only_val30_capture": base_capture,
                        }
                    )
    # ties (every blend=0 row is the same model) resolve toward less reranking
    frame = pd.DataFrame(grid).sort_values(
        ["val30_capture", "blend"], ascending=[False, True]
    )
    best_row = frame.iloc[0]
    name = str(best_row["source"])
    k, alpha, blend = int(best_row["k"]), float(best_row["alpha"]), float(best_row["blend"])
    print(
        f"[R4] selected source={name} K={k} alpha={alpha} blend={blend} "
        f"val30 capture {best_row['val30_capture']:.4f} "
        f"(stage-1 only {best_row['stage1_only_val30_capture']:.4f})"
        + ("   [declined to rerank]" if blend == 0.0 else ""),
        flush=True,
    )

    coef = fitted[(name, k, alpha)]
    curves = sources[name]
    realized_test = apply_reranker(coef, curves["test"], y_test_days, k, blend=blend)
    stage1_test = apply_reranker(np.zeros(8), curves["test"], y_test_days, 1, blend=0.0)

    payload = {
        "selected": {"source": name, "k": k, "alpha": alpha, "blend": blend,
                     "declined_to_rerank": blend == 0.0},
        "coefficients": dict(
            zip(
                [
                    "z_spread", "z_charge", "z_discharge",
                    "tc", "td", "td_minus_tc", "stage1_rank", "bias",
                ],
                coef.tolist(),
            )
        ),
        "val30": {
            "capture": float(best_row["val30_capture"]),
            "realized": float(best_row["val30_realized"]),
            "stage1_only_capture": float(best_row["stage1_only_val30_capture"]),
        },
        "holdout": {
            "realized_profit_mean": float(realized_test.mean()),
            "capture": float(realized_test.mean() / test_oracle.mean()),
            "stage1_only_realized": float(stage1_test.mean()),
            "lift_over_stage1": float(realized_test.mean() - stage1_test.mean()),
            "oracle_profit_mean": float(test_oracle.mean()),
        },
        "grid": frame.to_dict("records"),
        "protocol": "reranker trained on 5-fold OOF stage-1 curves over the 271 fit days",
        "seconds": round(time.time() - t0, 1),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "rerank.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "model": "r4_rerank",
            "date": pd.to_datetime(test_times).dt.normalize().unique(),
            "realized_profit": realized_test,
            "oracle_profit": test_oracle,
            "stage1_realized": stage1_test,
        }
    ).to_csv(REPORT_DIR / "rerank_daily.csv", index=False)
    print(
        f"[R4] holdout realized {realized_test.mean():.1f} "
        f"capture {realized_test.mean()/test_oracle.mean():.4f} "
        f"(stage-1 alone {stage1_test.mean():.1f}) in {(time.time()-t0)/60:.1f} min",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
