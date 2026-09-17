"""Diagnose RMSE vs realized dispatch profit on the frozen 59-day holdout."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase_b.contracts import BLOCK_STEPS, STEPS_PER_DAY
from src.phase_b.dispatch import optimize_day

V1_REPORT = ROOT / "reports" / "holdout_rmse_v1"
V2_REPORT = ROOT / "reports" / "holdout_rmse_v2"
OUT_DIR = ROOT / "reports" / "holdout_dispatch_replay"
ORACLE = 9000.886096


def _load_board(path: Path, version: str) -> pd.DataFrame:
    board = pd.read_csv(path)
    board["version"] = version
    board["label"] = version + "/" + board["model"]
    return board


def _day_spread(curve: np.ndarray) -> float:
    return float(np.max(curve) - np.min(curve))


def _spearman(actual: np.ndarray, predicted: np.ndarray) -> float:
    left = pd.Series(actual).rank().to_numpy()
    right = pd.Series(predicted).rank().to_numpy()
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _block_mean(curve: np.ndarray, start: int) -> float:
    return float(np.mean(curve[start : start + BLOCK_STEPS]))


def analyze_predictions(pred: pd.DataFrame, models: list[str]) -> dict[str, dict[str, float]]:
    actual = pred["A"].to_numpy(dtype=float).reshape(-1, STEPS_PER_DAY)
    n_days = actual.shape[0]
    oracle_windows = [optimize_day(day) for day in actual]
    stats: dict[str, dict[str, float]] = {}
    for name in models:
        yhat = pred[name].to_numpy(dtype=float).reshape(-1, STEPS_PER_DAY)
        exact = 0
        charge_err = []
        discharge_err = []
        spearman = []
        spread_pred = []
        spread_true = []
        rmse_all = []
        rmse_oracle_slots = []
        rmse_other = []
        pred_block_spread = []
        true_oracle_spread = []
        for i, oracle in enumerate(oracle_windows):
            chosen = optimize_day(yhat[i])
            if chosen.idle and oracle.idle:
                exact += 1
            elif not chosen.idle and not oracle.idle and chosen.tc == oracle.tc and chosen.td == oracle.td:
                exact += 1
            if chosen.idle or oracle.idle:
                charge_err.append(float("nan"))
                discharge_err.append(float("nan"))
            else:
                charge_err.append(abs(int(chosen.tc) - int(oracle.tc)))
                discharge_err.append(abs(int(chosen.td) - int(oracle.td)))
            rho = _spearman(actual[i], yhat[i])
            spearman.append(float(rho) if np.isfinite(rho) else 0.0)
            spread_pred.append(_day_spread(yhat[i]))
            spread_true.append(_day_spread(actual[i]))
            err = yhat[i] - actual[i]
            rmse_all.append(float(np.sqrt(np.mean(err**2))))
            if oracle.idle:
                rmse_oracle_slots.append(float("nan"))
                rmse_other.append(float(np.sqrt(np.mean(err**2))))
            else:
                mask = np.zeros(STEPS_PER_DAY, dtype=bool)
                mask[oracle.tc : oracle.tc + BLOCK_STEPS] = True
                mask[oracle.td : oracle.td + BLOCK_STEPS] = True
                rmse_oracle_slots.append(float(np.sqrt(np.mean(err[mask] ** 2))))
                rmse_other.append(float(np.sqrt(np.mean(err[~mask] ** 2))))
            if chosen.idle:
                pred_block_spread.append(0.0)
            else:
                pred_block_spread.append(_block_mean(yhat[i], chosen.td) - _block_mean(yhat[i], chosen.tc))
            if oracle.idle:
                true_oracle_spread.append(0.0)
            else:
                true_oracle_spread.append(_block_mean(actual[i], oracle.td) - _block_mean(actual[i], oracle.tc))
        stats[name] = {
            "exact_window_match": exact / n_days,
            "mean_abs_charge_slots": float(np.nanmean(charge_err)),
            "mean_abs_discharge_slots": float(np.nanmean(discharge_err)),
            "mean_spearman": float(np.mean(spearman)),
            "mean_pred_spread": float(np.mean(spread_pred)),
            "mean_true_spread": float(np.mean(spread_true)),
            "spread_ratio": float(np.mean(spread_pred) / np.mean(spread_true)),
            "mean_day_rmse": float(np.mean(rmse_all)),
            "rmse_oracle_16_slots": float(np.nanmean(rmse_oracle_slots)),
            "rmse_other_80_slots": float(np.mean(rmse_other)),
            "mean_pred_chosen_block_spread": float(np.mean(pred_block_spread)),
            "mean_true_oracle_block_spread": float(np.mean(true_oracle_spread)),
        }
    return stats


def daily_failure(daily: pd.DataFrame, name: str) -> dict[str, float]:
    slice_ = daily.loc[daily["model"] == name].copy()
    return {
        "negative_days": int((slice_["realized_profit"] < 0).sum()),
        "idle_days": int(slice_["idle"].sum()) if "idle" in slice_.columns else 0,
        "days_capture_below_0_5": int((slice_["capture_rate"] < 0.5).sum()),
        "days_window_exact": int(
            ((slice_["tc"] == slice_["oracle_tc"]) & (slice_["td"] == slice_["oracle_td"])).sum()
        ),
        "mean_regret": float(slice_["regret"].mean()),
        "worst_day": str(pd.to_datetime(slice_.loc[slice_["realized_profit"].idxmin(), "date"]).date()),
        "worst_realized": float(slice_["realized_profit"].min()),
        "best_realized": float(slice_["realized_profit"].max()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v1 = _load_board(V1_REPORT / "leaderboard.csv", "v1")
    v2 = _load_board(V2_REPORT / "leaderboard.csv", "v2")
    all_rows = pd.concat([v1, v2], ignore_index=True)
    all_rows["rmse_rank"] = all_rows.groupby("version")["RMSE"].rank(method="min")
    all_rows["realized_rank"] = all_rows.groupby("version")["realized_profit_mean"].rank(
        method="min", ascending=False
    )
    combined = all_rows.sort_values("realized_profit_mean", ascending=False).drop_duplicates("model")
    combined["source"] = combined["version"]

    pred_v1 = pd.read_parquet(V1_REPORT / "predictions.parquet")
    pred_v2 = pd.read_parquet(V2_REPORT / "predictions.parquet")
    daily_v1 = pd.read_csv(V1_REPORT / "dispatch_daily.csv")
    daily_v2 = pd.read_csv(V2_REPORT / "dispatch_daily.csv")
    v1_models = [c for c in pred_v1.columns if c not in {"times", "A"}]
    v2_models = [c for c in pred_v2.columns if c not in {"times", "A"}]
    shape_v1 = analyze_predictions(pred_v1, v1_models)
    shape_v2 = analyze_predictions(pred_v2, v2_models)

    focus = [
        ("v1", "transformer_contextual"),
        ("v1", "lgb_contextual"),
        ("v1", "catboost_contextual"),
        ("v1", "ridge_contextual_a1"),
        ("v1", "lgb_full"),
        ("v1", "lstm_contextual"),
        ("v2", "transformer_contextual"),
        ("v2", "lgb_contextual"),
        ("v2", "xgboost_contextual"),
        ("v2", "dlinear_contextual"),
        ("v2", "lstm_contextual"),
        ("v2", "ridge_contextual_a1"),
    ]
    daily_map = {"v1": daily_v1, "v2": daily_v2}
    shape_map = {"v1": shape_v1, "v2": shape_v2}
    focus_rows = []
    for version, name in focus:
        board = v1 if version == "v1" else v2
        row = board.loc[board["model"] == name].iloc[0]
        shape = shape_map[version][name]
        fail = daily_failure(daily_map[version], name)
        focus_rows.append(
            {
                "version": version,
                "model": name,
                "RMSE": float(row["RMSE"]),
                "realized_profit_mean": float(row["realized_profit_mean"]),
                "pred_profit_mean": float(row["pred_profit_mean"]),
                "profit_gap": float(row["profit_gap"]),
                "capture": float(row["capture"]),
                **shape,
                **fail,
            }
        )

    payload = {
        "oracle_profit_mean": ORACLE,
        "holdout": "2025-11-03 to 2025-12-31, 59 days",
        "n_v1": int(len(v1)),
        "n_v2": int(len(v2)),
        "all_rows": all_rows.sort_values("realized_profit_mean", ascending=False)[
            [
                "label",
                "version",
                "model",
                "RMSE",
                "MAE",
                "pred_profit_mean",
                "realized_profit_mean",
                "profit_gap",
                "capture",
                "rmse_rank",
                "realized_rank",
            ]
        ].to_dict(orient="records"),
        "best_per_name": combined.sort_values("realized_profit_mean", ascending=False)[
            ["model", "version", "RMSE", "realized_profit_mean", "profit_gap", "capture"]
        ].to_dict(orient="records"),
        "shape": {"v1": shape_v1, "v2": shape_v2},
        "focus": focus_rows,
        "corr_v1": {
            "rmse_vs_realized": float(v1["RMSE"].corr(v1["realized_profit_mean"])),
            "rmse_vs_capture": float(v1["RMSE"].corr(v1["capture"])),
            "gap_vs_realized": float(v1["profit_gap"].corr(v1["realized_profit_mean"])),
            "pred_vs_realized": float(v1["pred_profit_mean"].corr(v1["realized_profit_mean"])),
        },
        "corr_v2": {
            "rmse_vs_realized": float(v2["RMSE"].corr(v2["realized_profit_mean"])),
            "rmse_vs_capture": float(v2["RMSE"].corr(v2["capture"])),
            "gap_vs_realized": float(v2["profit_gap"].corr(v2["realized_profit_mean"])),
            "pred_vs_realized": float(v2["pred_profit_mean"].corr(v2["realized_profit_mean"])),
        },
    }
    out = OUT_DIR / "failure_analysis.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print("v1 corr rmse vs realized", payload["corr_v1"]["rmse_vs_realized"])
    print("v2 corr rmse vs realized", payload["corr_v2"]["rmse_vs_realized"])
    print("top realized")
    for row in payload["all_rows"][:8]:
        print(f"  {row['label']:40s} RMSE={row['RMSE']:.3f} realized={row['realized_profit_mean']:.1f} cap={row['capture']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
