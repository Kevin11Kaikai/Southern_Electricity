"""Single common holdout replay, only AFTER candidate freeze; no training."""

import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from .common import ROOT, PROTOCOL_PATH, Ledger, construct_model, check_protected, read_json, sha256, utcnow, write_json
from .run import run_directory, snapshot
from experiments.holdout_rmse_v1.sequence_models import apply_preprocess
from src.phase_b.evaluation import evaluate_price_predictions
from src.phase_b.dispatch import LEGAL_WINDOWS, optimize_day


def independent_power(prices):
    """Independent cumsum implementation, including original near-tie rule."""
    prices = np.asarray(prices, dtype=float)
    sums = np.r_[0., np.cumsum(prices)]
    blocks = sums[8:] - sums[:-8]
    scores = np.asarray([1000. * (blocks[d] - blocks[c]) for c, d in LEGAL_WINDOWS])
    tolerance = 64 * np.finfo(float).eps * np.max(np.abs(scores))
    best = np.flatnonzero(scores >= scores.max() - tolerance)[0]
    power = np.zeros(96)
    if scores[best] > 0:
        c, d = LEGAL_WINDOWS[best]
        power[c:c + 8], power[d:d + 8] = -1000., 1000.
    return power


def load_prediction(checkpoint, x, seed):
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = construct_model(data["family"], data["n_features"], seed)
    model.load_state_dict(data["state_dict"])
    model.eval()
    features = apply_preprocess(x, data["fill_values"], data["mean"], data["scale"])
    with torch.no_grad():
        return model(torch.from_numpy(features)).cpu().numpy().astype(float)


def paired_summary(daily, left, right):
    a = daily.loc[daily.candidate == left].set_index("date")["realized_profit"]
    b = daily.loc[daily.candidate == right].set_index("date")["realized_profit"]
    delta = (a - b).sort_index()
    if len(delta) != 59 or delta.isna().any():
        raise AssertionError("paired test dates mismatch")
    top = delta.idxmax()
    return {
        "candidate": left, "reference": right, "mean_difference": float(delta.mean()),
        "median_difference": float(delta.median()), "wins": int((delta > 1e-7).sum()),
        "losses": int((delta < -1e-7).sum()), "ties": int((delta.abs() <= 1e-7).sum()),
        "first_30_mean_difference": float(delta.iloc[:30].mean()),
        "last_29_mean_difference": float(delta.iloc[30:].mean()),
        "best_day": str(top.date()), "best_day_difference": float(delta.loc[top]),
        "without_best_day_mean_difference": float(delta.drop(top).mean()),
        "worst_day": str(delta.idxmin().date()), "worst_day_difference": float(delta.min()),
        "period": "2025-11-03 through 2025-12-31; 59-day holdout",
        "weights_fitted_on_holdout": False,
    }


def evaluate():
    protocol = read_json(PROTOCOL_PATH)
    run_dir = run_directory(protocol)
    if (run_dir / "evaluation_complete.json").exists():
        print("Evaluation already complete; reusing saved results, no test rerun.")
        return read_json(run_dir / "analysis.json")
    frozen = read_json(run_dir / "candidate_freeze.json")
    if not check_protected(run_dir)["unchanged"]:
        raise RuntimeError("reference inputs changed before evaluation")
    for c in frozen["candidates"]:
        if sha256(c["checkpoint"]) != c["checkpoint_sha256"]:
            raise RuntimeError("frozen checkpoint changed")
    ledger = Ledger(run_dir, protocol)
    if ledger.remaining_seconds() < 90:
        raise RuntimeError("insufficient remaining evaluation budget")
    ledger.add("holdout_evaluation_start", candidate_freeze_sha256=sha256(run_dir / "candidate_freeze.json"))
    started = time.perf_counter()
    torch.set_num_threads(protocol["torch_cpu_threads"])
    with np.load(run_dir / "data/holdout_inputs.npz", allow_pickle=False) as data:
        x, times = data["x"], pd.to_datetime(data["times"])
    with np.load(run_dir / "data/holdout_labels.npz", allow_pickle=False) as data:
        y = data["y"].reshape(-1)
    predictions = pd.DataFrame({"times": times, "A": y})
    rows, daily_frames = [], []
    checks = {"reload_max_abs_diff": 0., "independent_score_max_abs_diff": 0., "independent_action_mismatch_days": 0, "evaluated_candidate_days": 0}

    def score(candidate, values, group, family, metadata):
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != 5664 or not np.isfinite(values).all():
            raise ValueError(f"invalid predictions for {candidate}")
        predictions[candidate] = values
        report = evaluate_price_predictions(times, y, values, incomplete="raise")
        daily = report["daily"].copy()
        daily.insert(0, "candidate", candidate)
        daily_frames.append(daily)
        for i, (actual_day, pred_day) in enumerate(zip(y.reshape(59, 96), values.reshape(59, 96))):
            power = independent_power(pred_day)
            exact = optimize_day(pred_day).power
            checks["independent_action_mismatch_days"] += int(not np.array_equal(power, exact))
            checks["independent_score_max_abs_diff"] = max(checks["independent_score_max_abs_diff"], abs(float(actual_day @ power) - float(daily.iloc[i].realized_profit)))
            checks["evaluated_candidate_days"] += 1
        realized = float(daily.realized_profit.mean())
        oracle = float(daily.oracle_profit.mean())
        row = {
            "candidate": candidate, "group": group, "family": family, "days": 59,
            "test_start": "2025-11-03", "test_end": "2025-12-31", "is_59_day_holdout": True,
            "weights_fitted_on_holdout": False,
            "realized_profit_mean": realized, "pred_profit_mean": float(daily.predicted_profit.mean()),
            "capture": realized / oracle, "oracle_profit_mean": oracle,
            "RMSE": float(np.sqrt(np.mean((values - y) ** 2))), "MAE": float(np.mean(np.abs(values - y))),
            "negative_days": int((daily.realized_profit < 0).sum()), "idle_days": int(daily.idle.sum()),
            "worst_day_score": float(daily.realized_profit.min()), **metadata,
        }
        rows.append(row)

    # Recompute historical metrics from their actual saved predictions. Do not
    # pair old RMSE columns with later refitted dispatch scores.
    references = [
        ("reports/holdout_rmse_v2/predictions.parquet", [
            ("lgb_baseline", "lgb_baseline", "baseline", "lightgbm"),
            ("transformer_contextual", "nb11_v2_transformer", "nb11_v2", "transformer"),
            ("ridge_contextual_a1", "nb11_v2_ridge", "nb11_v2", "ridge"),
        ]),
        ("reports/holdout_rmse_v1/predictions.parquet", [
            ("transformer_contextual", "nb11_v1_transformer", "nb11_v1_context_only", "transformer"),
            ("lstm_contextual", "nb11_v1_lstm", "nb11_v1_context_only", "lstm"),
        ]),
        ("reports/holdout_dispatch_loss_v1/predictions.parquet", [
            ("transformer_dispatch", "cursor_original_transformer", "cursor_original", "transformer"),
            ("lstm_dispatch", "cursor_original_lstm", "cursor_original", "lstm"),
            ("dlinear_dispatch", "cursor_original_dlinear", "cursor_original", "dlinear"),
        ]),
    ]
    for source, mappings in references:
        ref = pd.read_parquet(ROOT / source)
        if not pd.DatetimeIndex(pd.to_datetime(ref.times)).equals(pd.DatetimeIndex(times)) or not np.allclose(ref.A, y, rtol=0, atol=1e-12):
            raise AssertionError(f"reference target/date mismatch: {source}")
        for column, name, group, family in mappings:
            score(name, ref[column], group, family, {"source": source, "training_role": "historical_reference"})
    for c in frozen["candidates"]:
        values = load_prediction(c["checkpoint"], x, protocol["seed"])
        reloaded = load_prediction(c["checkpoint"], x, protocol["seed"])
        checks["reload_max_abs_diff"] = max(checks["reload_max_abs_diff"], float(np.max(np.abs(values - reloaded))))
        group = "codex_main" if c["method"] == "soft_regret" else ("matched_cursor_control" if c["method"] == "cursor_ce" else f"mse_{c['selection_policy']}_control")
        score(c["candidate"], values, group, c["family"], {
            "source": str(Path(c["checkpoint"]).relative_to(ROOT)), "training_role": "new_frozen_candidate",
            "validation_realized": c["validation_realized"], "validation_rmse": c["validation_rmse"],
            "best_epoch": c["best_epoch"], "final_epochs": c["final_epochs"],
            "lambda": c["params"]["lam"], "tau": c["params"]["tau"],
            "selection_policy": c["selection_policy"], "preselected_primary": c["candidate"] == frozen["primary_candidate"],
        })
    comparison = pd.DataFrame(rows)
    daily = pd.concat(daily_frames, ignore_index=True)
    baseline = float(comparison.loc[comparison.candidate == "lgb_baseline", "realized_profit_mean"].iloc[0])
    cursor_best = comparison.loc[comparison.group == "cursor_original"].sort_values("realized_profit_mean", ascending=False).iloc[0]
    comparison["difference_vs_lgb"] = comparison.realized_profit_mean - baseline
    comparison["improvement_vs_lgb_pct"] = 100 * comparison.difference_vs_lgb / baseline
    comparison["difference_vs_cursor_best"] = comparison.realized_profit_mean - float(cursor_best.realized_profit_mean)
    comparison["improvement_vs_cursor_best_pct"] = 100 * comparison.difference_vs_cursor_best / float(cursor_best.realized_profit_mean)
    comparison.to_csv(run_dir / "comparison.csv", index=False)
    daily.to_csv(run_dir / "dispatch_daily.csv", index=False)
    predictions.to_parquet(run_dir / "predictions.parquet", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    primary = frozen["primary_candidate"]
    primary_row = comparison.loc[comparison.candidate == primary].iloc[0]
    observed_best = comparison.loc[comparison.group == "codex_main"].sort_values("realized_profit_mean", ascending=False).iloc[0]
    controls = []
    for family in protocol["families"]:
        mse_money = f"codex_mse_{family}_realized"
        mse_rmse = f"codex_mse_{family}_rmse"
        soft = f"codex_soft_regret_{family}_realized"
        ce = f"codex_cursor_ce_{family}_realized"
        controls.append({
            "family": family,
            "selection_only": paired_summary(daily, mse_money, mse_rmse),
            "soft_vs_mse_money": paired_summary(daily, soft, mse_money),
            "soft_vs_matched_ce": paired_summary(daily, soft, ce),
            "soft_vs_cursor_original": paired_summary(daily, soft, f"cursor_original_{family}"),
        })
    selection = read_json(run_dir / "selection_frozen.json")
    initial_hashes = {}
    for c in frozen["candidates"]:
        initial_hashes.setdefault(c["family"], set()).update([c["search_initial_hash"], c["refit_initial_hash"]])
    checks["same_initialization_within_family"] = {family: len(values) == 1 for family, values in initial_hashes.items()}
    checks["checkpoint_hashes_match_freeze"] = True
    checks["same_59_dates_and_targets"] = True
    checks["model_action_constraints_checked_by_official_evaluator"] = True
    checks["nonnegative_regret_minimum"] = float(daily.regret.min())
    if checks["independent_action_mismatch_days"] or checks["independent_score_max_abs_diff"] > 1e-7 or checks["reload_max_abs_diff"] > 1e-7 or daily.regret.min() < -1e-7:
        raise AssertionError(f"evaluation verification failed: {checks}")
    pair = paired_summary(daily, primary, cursor_best.candidate)
    analysis = {
        "completed_utc": utcnow(), "protocol": "271/30 -> refit301 -> score59",
        "period": "2025-11-03 through 2025-12-31", "days": 59,
        "weight_selection_period": "2025-10-04 through 2025-11-02 (30 development days)", "weights_fitted_on_test_days": False,
        "primary_candidate": primary, "primary_score": float(primary_row.realized_profit_mean),
        "cursor_original_best_candidate": cursor_best.candidate, "cursor_original_best_score": float(cursor_best.realized_profit_mean),
        "primary_difference_vs_cursor_best": float(primary_row.realized_profit_mean - cursor_best.realized_profit_mean),
        "primary_improvement_vs_cursor_best_pct": float(primary_row.improvement_vs_cursor_best_pct),
        "baseline_score": baseline, "primary_improvement_vs_baseline_pct": float(primary_row.improvement_vs_lgb_pct),
        "observed_best_codex_candidate": observed_best.candidate, "observed_best_codex_score": float(observed_best.realized_profit_mean),
        "observed_best_is_preselected_primary": observed_best.candidate == primary,
        "observed_best_is_post_test_description_only": True,
        "primary_vs_cursor_daily": pair, "controls": controls, "checks": checks,
        "cost": snapshot(run_dir, protocol, "HOLDOUT_EVALUATED; building notebook and figures"),
        "limits": [
            "Single seed; same local test period; no stable-generalization or SOTA claim.",
            "soft_regret and inclusion of idle are a combined mechanism; no separate idle ablation.",
            "Soft vs MSE includes loss-specific three-point search vs fixed MSE; matched CE has the same three-point search budget.",
            "Historical Cursor seed-before-constructor and CPU runtime were not matched; use new matched CE for causal comparison.",
            "No actual agent research sessions, RL, new data, or real-money deployment in this run.",
            "Historical notebook 11 v1 models are context only, not matched 301-day retraining controls.",
        ],
    }
    write_json(run_dir / "analysis.json", analysis)
    write_json(run_dir / "evaluation_checks.json", checks)
    ledger.add("holdout_evaluation_end", seconds=time.perf_counter() - started, candidates=len(comparison), status="ok")
    write_json(run_dir / "evaluation_complete.json", {"completed_utc": utcnow(), "analysis_sha256": sha256(run_dir / "analysis.json"), "comparison_sha256": sha256(run_dir / "comparison.csv")})
    print(comparison[["candidate", "realized_profit_mean", "difference_vs_cursor_best", "RMSE"]].to_string(index=False))
    print("PRIMARY", primary, "vs Cursor", analysis["primary_difference_vs_cursor_best"], flush=True)
    return analysis


if __name__ == "__main__":
    evaluate()
