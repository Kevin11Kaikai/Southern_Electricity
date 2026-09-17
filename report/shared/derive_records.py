"""Read frozen repository evidence and export auditable report records.

This program does not fit or load any predictive model. It never writes to its
input directories. Figures are drawn separately by build_figures.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata, spearmanr

SEED = 20260915
N_BOOT = 10000
BLOCKS = (7, 1, 3)
WINDOWS = np.array([(c, d) for c in range(81) for d in range(c + 8, 89)])
HEAD = "listwise_transformer_mean"
SOURCE_DIRS = ["holdout_rmse_v1", "holdout_rmse_v2", "holdout_seed_bag_v1",
               "holdout_dispatch_loss_v1", "holdout_dfl_bakeoff_v1"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def blocks(q: np.ndarray, centered: bool = False) -> np.ndarray:
    if centered:
        q = q - q.mean(axis=1, keepdims=True)
    return np.lib.stride_tricks.sliding_window_view(q, 8, axis=1).sum(axis=-1)


def scores(q: np.ndarray) -> np.ndarray:
    b = blocks(q)
    return b[:, WINDOWS[:, 1]] - b[:, WINDOWS[:, 0]]


def pearson_rows(x, y):
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt((xc * xc).sum(axis=-1) * (yc * yc).sum(axis=-1))
    return np.divide((xc * yc).sum(axis=-1), den, out=np.full_like(den, np.nan), where=den > 0)


def bootstrap_counts(n: int, length: int, repetitions: int, seed: int):
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n, size=(repetitions, math.ceil(n / length)))
    ids = ((start[..., None] + np.arange(length)) % n).reshape(repetitions, -1)[:, :n]
    counts = np.zeros((repetitions, n), dtype=float)
    np.add.at(counts, (np.repeat(np.arange(repetitions), n), ids.ravel()), 1)
    return counts / n


def ci(a):
    z = np.asarray(a, dtype=float)
    z = z[np.isfinite(z)]
    return [float(v) for v in np.quantile(z, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs = []

    def source(relative):
        p = root / relative
        if not p.is_file():
            raise FileNotFoundError(p)
        if relative not in [v["path"] for v in inputs]:
            inputs.append({"path": relative, "sha256": sha(p), "bytes": p.stat().st_size})
        return p

    # Persist analysis choices before inspecting derived correlations/intervals.
    save_json(out / "analysis_plan.json", {
        "operation": "posthoc_analysis_of_fixed_predictions_no_training",
        "candidate_panels": {"v1": 16, "v2": 19},
        "exclusions": [], "pool_versions": False,
        "loss": "centered_block_sum_mse + 2 * listwise_cross_entropy",
        "tau_target": 0.05, "tau_prediction": 0.1,
        "window_rank_ties": "average ranks; float64 raw 8-slot sums",
        "bootstrap": {"method": "paired circular day blocks", "primary_block_length": 7,
                      "sensitivity_block_lengths": [1, 3], "replicates": N_BOOT, "seed": SEED},
        "inference_scope": "conditional on the observed window and fixed candidates; historical selection uncorrected",
        "float_tolerance_score": {"rtol": 1e-9, "atol": 1e-6}
    })

    metric_rows, score_rows, drift_rows, model_rows = [], [], [], []
    canonical_times, canonical_a = None, None
    for directory in SOURCE_DIRS:
        pred = pd.read_parquet(source(f"reports/{directory}/predictions.parquet")).sort_values("times").reset_index(drop=True)
        daily = pd.read_csv(source(f"reports/{directory}/dispatch_daily.csv"))
        board = pd.read_csv(source(f"reports/{directory}/leaderboard.csv")).set_index("model")
        times = pd.to_datetime(pred["times"])
        assert len(pred) == 5664 and not times.duplicated().any()
        assert times.groupby(times.dt.date).size().eq(96).all()
        a = pred["A"].to_numpy(float).reshape(59, 96)
        dates = times.iloc[::96].dt.strftime("%Y-%m-%d").tolist()
        if canonical_times is None:
            canonical_times, canonical_a = times, a
        else:
            assert times.equals(canonical_times)
            np.testing.assert_allclose(a, canonical_a, rtol=0, atol=1e-12)
        ba = blocks(a, centered=True)
        sa = scores(a)
        rank_a = rankdata(sa, axis=1, method="average")
        target = softmax((ba[:, WINDOWS[:, 1]] - ba[:, WINDOWS[:, 0]]) / .05, axis=1)
        oracle = np.maximum(sa.max(axis=1), 0) * 1000
        for model in pred.columns.drop(["times", "A"]):
            q = pred[model].to_numpy(float).reshape(59, 96)
            assert np.isfinite(q).all()
            d = daily[daily["model"].eq(model)].set_index("date").reindex(dates)
            if d["realized_profit"].isna().any():
                raise AssertionError((directory, model, "missing dates"))
            sq, bq = scores(q), blocks(q, centered=True)
            idle = d["idle"].astype(str).str.lower().isin(["true", "1"]).to_numpy()
            c = d["tc"].fillna(0).to_numpy(int)
            z = d["td"].fillna(8).to_numpy(int)
            active = ~idle
            assert ((c[active] >= 0) & (c[active] <= 80) & (z[active] >= c[active] + 8) & (z[active] <= 88)).all()
            b_raw = blocks(a)
            truth_score = np.where(idle, 0, 1000 * (b_raw[np.arange(59), z] - b_raw[np.arange(59), c]))
            np.testing.assert_allclose(truth_score, d.realized_profit, atol=1e-6, rtol=1e-9)
            np.testing.assert_allclose(oracle, d.oracle_profit, atol=1e-6, rtol=1e-9)
            q_raw = blocks(q)
            pred_score = np.where(idle, 0, 1000 * (q_raw[np.arange(59), z] - q_raw[np.arange(59), c]))
            np.testing.assert_allclose(pred_score, d.predicted_profit, atol=1e-3, rtol=1e-7)
            # Check objective optimality; use saved solver tie-breaking for the actual action.
            np.testing.assert_allclose(pred_score, np.maximum(sq.max(axis=1), 0) * 1000, atol=1e-3, rtol=1e-7)
            rp = truth_score
            sq_centered = bq[:, WINDOWS[:, 1]] - bq[:, WINDOWS[:, 0]]
            logits = sq_centered / .1
            log_pi = logits - logsumexp(logits, axis=1, keepdims=True)
            ce = -(target * log_pi).sum(axis=1)
            block_mse = ((bq - ba) ** 2).mean(axis=1)
            mse = ((q - a) ** 2).mean(axis=1)
            mae = np.abs(q - a).mean(axis=1)
            cq, ca = q - q.mean(axis=1, keepdims=True), a - a.mean(axis=1, keepdims=True)
            cmse = ((cq - ca) ** 2).mean(axis=1)
            wrank = pearson_rows(rankdata(sq, axis=1, method="average"), rank_a)
            assert np.isfinite(wrank).all()
            actual_rmse = float(np.sqrt(mse.mean()))
            saved_rmse = float(board.loc[model, "RMSE"])
            np.testing.assert_allclose(rp.mean(), float(board.loc[model, "realized_profit_mean"]), atol=2e-5, rtol=0)
            np.testing.assert_allclose(rp.mean() / oracle.mean(), float(board.loc[model, "capture"]), atol=1e-6, rtol=0)
            family = (str(board.loc[model, "family"]) if "family" in board.columns else model.split("_")[0])
            model_rows.append({"source": directory, "model": model, "family": family,
                "n_days": 59, "rmse_same_prediction": actual_rmse,
                "rmse_original_table": saved_rmse, "mae_same_prediction": float(mae.mean()),
                "centered_rmse": float(np.sqrt(cmse.mean())),
                "realized_mean": float(rp.mean()), "predicted_mean": float(pred_score.mean()),
                "oracle_mean": float(oracle.mean()), "capture_aggregate": float(rp.sum() / oracle.sum()),
                "capture_daily_mean": float(np.mean(rp / oracle)),
                "listwise_proxy": float((block_mse + 2 * ce).mean()),
                "amplitude_ratio": float(np.sqrt((cq*cq).mean()/(ca*ca).mean())),
                "window_spearman": float(wrank.mean())})
            if abs(actual_rmse - saved_rmse) > 1e-5:
                drift_rows.append({"source": directory, "model": model,
                                  "original_rmse": saved_rmse, "same_prediction_rmse": actual_rmse,
                                  "realized_mean": float(rp.mean())})
            for i, date in enumerate(dates):
                record = {"source": directory, "model": model, "date": date,
                    "tc": int(c[i]), "td": int(z[i]), "idle": bool(idle[i]),
                    "realized": float(rp[i]), "predicted": float(pred_score[i]), "oracle": float(oracle[i]),
                    "capture_daily": float(rp[i] / oracle[i])}
                score_rows.append(record)
                metric_rows.append({**record, "point_mse": float(mse[i]), "point_mae": float(mae[i]),
                    "centered_mse": float(cmse[i]), "block_mse": float(block_mse[i]),
                    "listwise_ce": float(ce[i]), "listwise_proxy": float(block_mse[i] + 2 * ce[i]),
                    "window_spearman": float(wrank[i]),
                    "pred_centered_ms": float((cq[i] ** 2).mean()), "true_centered_ms": float((ca[i] ** 2).mean())})
        # Window-vote outputs lack their own price curves, but their saved actions can be settled exactly.
        for model in [v for v in daily.model.unique() if v not in pred.columns]:
            d = daily[daily.model.eq(model)].set_index("date").reindex(dates)
            for i, date in enumerate(dates):
                row = d.iloc[i]
                score_rows.append({"source": directory, "model": model, "date": date,
                                  "tc": int(row.tc), "td": int(row.td), "idle": bool(row.idle),
                                  "realized": float(row.realized_profit), "predicted": float(row.predicted_profit),
                                  "oracle": float(row.oracle_profit), "capture_daily": float(row.capture_rate)})
        print("Validated", directory, len(pred.columns)-2, "frozen curves", flush=True)

    metrics, all_scores = pd.DataFrame(metric_rows), pd.DataFrame(score_rows)
    models = pd.DataFrame(model_rows)
    metrics.to_csv(out / "daily_metrics.csv", index=False)
    all_scores.to_csv(out / "daily_scores.csv", index=False)
    models.to_csv(out / "model_metrics.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(out / "rmse_provenance.csv", index=False)
    weights = {k: bootstrap_counts(59, k, N_BOOT, SEED+k) for k in BLOCKS}
    corr_rows = []
    partial_rows = []
    for version in ["v1", "v2"]:
        d = metrics[metrics.source.eq("holdout_rmse_" + version)]
        y = d.pivot(index="date", columns="model", values="realized")
        assert y.shape == (59, 16 if version == "v1" else 19)
        ym = y.mean(axis=0).to_numpy()
        for column, root_mean in [("point_mse", True), ("centered_mse", True),
                                  ("listwise_proxy", False), ("window_spearman", False)]:
            x = d.pivot(index="date", columns="model", values=column).reindex(columns=y.columns).to_numpy()
            xm = x.mean(axis=0)
            if root_mean:
                xm = np.sqrt(xm)
            for k, w in weights.items():
                xb, yb = w @ x, w @ y.to_numpy()
                if root_mean:
                    xb = np.sqrt(xb)
                pear = pearson_rows(xb, yb)
                rank = pearson_rows(rankdata(xb, axis=1), rankdata(yb, axis=1))
                lo, hi = ci(pear); rlo, rhi = ci(rank)
                corr_rows.append({"version": version, "metric": column, "n_candidates": len(xm),
                    "n_days": 59, "pearson": float(np.corrcoef(xm, ym)[0, 1]),
                    "spearman": float(spearmanr(xm, ym).statistic),
                    "block_length": k, "pearson_ci_low": lo, "pearson_ci_high": hi,
                    "spearman_ci_low": rlo, "spearman_ci_high": rhi})
                if column in ("window_spearman", "listwise_proxy"):
                    pq = d.pivot(index="date", columns="model", values="pred_centered_ms").reindex(columns=y.columns).to_numpy()
                    pa = d.pivot(index="date", columns="model", values="true_centered_ms").reindex(columns=y.columns).to_numpy()
                    amp = np.sqrt(pq.mean(axis=0)/pa.mean(axis=0))
                    ab = np.sqrt((w @ pq)/(w @ pa))
                    def partial(xv, yv, av):
                        xy, xa, ya = pearson_rows(xv,yv), pearson_rows(xv,av), pearson_rows(yv,av)
                        return (xy-xa*ya)/np.sqrt((1-xa*xa)*(1-ya*ya))
                    plo, phi = ci(partial(xb,yb,ab))
                    partial_rows.append({"version": version, "metric": column, "control": "centered_amplitude_ratio",
                        "partial_pearson": float(partial(xm,ym,amp)), "block_length": k,
                        "ci_low": plo, "ci_high": phi,
                        "amplitude_realized_pearson": float(pearson_rows(amp,ym))})
    pd.DataFrame(corr_rows).to_csv(out / "correlations.csv", index=False)
    pd.DataFrame(partial_rows).to_csv(out / "amplitude_control.csv", index=False)

    def series(directory, model):
        return all_scores[(all_scores.source.eq(directory)) & (all_scores.model.eq(model))].set_index("date").sort_index()
    h = series("holdout_seed_bag_v1", HEAD)
    ref = series("holdout_rmse_v2", "transformer_contextual")
    baseline = series("holdout_rmse_v2", "lgb_baseline")
    assert h.index.is_unique and h.index.equals(ref.index) and h.index.equals(baseline.index)
    for seed in range(42, 47):
        assert h.index.equals(series("holdout_seed_bag_v1", f"listwise_transformer_seed{seed}").index)
    seed_day = np.column_stack([series("holdout_seed_bag_v1", f"listwise_transformer_seed{s}").realized for s in range(42,47)])
    paired = pd.DataFrame({"date": h.index, "reference": ref.realized.to_numpy(),
        "headline": h.realized.to_numpy(), "baseline": baseline.realized.to_numpy(),
        "single_seed_average": seed_day.mean(axis=1), "oracle": h.oracle.to_numpy()})
    paired["paired_gain"] = paired.headline - paired.reference
    paired["ensemble_increment"] = paired.headline - paired.single_seed_average
    paired["cumulative_gain"] = paired.paired_gain.cumsum()
    paired["same_window_as_reference"] = ((h.idle.to_numpy() & ref.idle.to_numpy()) |
        ((~h.idle.to_numpy()) & (~ref.idle.to_numpy()) &
         (h.tc.to_numpy() == ref.tc.to_numpy()) & (h.td.to_numpy() == ref.td.to_numpy())))
    paired.to_csv(out / "paired_daily.csv", index=False)
    intervals = []
    for comparison, dif in [("headline_vs_reference", paired.paired_gain),
                            ("headline_vs_single_seed_mean", paired.ensemble_increment)]:
        for k, w in weights.items():
            lo, hi = ci(w @ dif.to_numpy())
            intervals.append({"comparison": comparison, "block_length": k, "n_days": 59,
                "difference_mean": float(dif.mean()), "paired_iid_se": float(dif.std(ddof=1)/np.sqrt(59)),
                "ci_low": lo, "ci_high": hi, "replicates": N_BOOT, "seed": SEED+k})
    pd.DataFrame(intervals).to_csv(out / "paired_intervals.csv", index=False)
    special = paired[paired.date.eq("2025-12-30")].iloc[0]
    keep = paired.date.ne("2025-12-30")
    summary = {
        "headline": float(paired.headline.mean()), "reference": float(paired.reference.mean()),
        "baseline": float(paired.baseline.mean()), "oracle": float(paired.oracle.mean()),
        "capture_headline": float(paired.headline.sum()/paired.oracle.sum()),
        "capture_reference": float(paired.reference.sum()/paired.oracle.sum()),
        "paired_gain": float(paired.paired_gain.mean()),
        "relative_gain_pct": float(100 * paired.paired_gain.mean()/paired.reference.mean()),
        "total_gain": float(paired.paired_gain.sum()),
        "single_seed_mean": float(seed_day.mean()), "ensemble_increment": float(paired.ensemble_increment.mean()),
        "nonensemble_remainder": float(seed_day.mean() - paired.reference.mean()),
        "paired_iid_se": float(paired.paired_gain.std(ddof=1)/np.sqrt(59)),
        "wins": int((paired.paired_gain > 1e-6).sum()),
        "ties": int((paired.paired_gain.abs() <= 1e-6).sum()),
        "losses": int((paired.paired_gain < -1e-6).sum()),
        "paired_median": float(paired.paired_gain.median()),
        "dec30_headline": float(special.headline), "dec30_reference": float(special.reference),
        "dec30_single_seed_mean": float(special.single_seed_average),
        "dec30_share_total_gain_pct": float(100*special.paired_gain/paired.paired_gain.sum()),
        "dec30_share_ensemble_gain_pct": float(100*special.ensemble_increment/paired.ensemble_increment.sum()),
        "without_dec30_gain": float(paired.loc[keep, "paired_gain"].mean()),
        "without_dec30_ensemble_increment": float(paired.loc[keep,"ensemble_increment"].mean()),
        "oracle_value_days_headline": int(np.isclose(paired.headline,paired.oracle,rtol=1e-10,atol=1e-6).sum()),
        "oracle_value_days_reference": int(np.isclose(paired.reference,paired.oracle,rtol=1e-10,atol=1e-6).sum()),
        # Downside of the settled action: the mean alone hides the loss tail a dispatcher carries.
        "worst_day_headline": float(paired.headline.min()), "worst_day_reference": float(paired.reference.min()),
        "worst_day_baseline": float(paired.baseline.min()),
        "negative_days_headline": int((paired.headline < 0).sum()),
        "negative_days_reference": int((paired.reference < 0).sum()),
        "negative_days_baseline": int((paired.baseline < 0).sum()),
        "p05_headline": float(paired.headline.quantile(0.05)), "p05_reference": float(paired.reference.quantile(0.05)),
        "median_headline": float(paired.headline.median()), "median_reference": float(paired.reference.median()),
        "gross_winning_day_total": float(paired.paired_gain[paired.paired_gain > 1e-6].sum()),
        "gross_losing_day_total": float(paired.paired_gain[paired.paired_gain < -1e-6].sum()),
        "tie_day_total_residual": float(paired.paired_gain[paired.paired_gain.abs() <= 1e-6].sum()),
        "worst_paired_day": str(paired.loc[paired.paired_gain.idxmin(), "date"]),
        "worst_paired_dates_within_tolerance": paired.loc[
            np.isclose(paired.paired_gain, paired.paired_gain.min(), rtol=0, atol=1e-6), "date"].tolist(),
        "worst_paired_tie_rule": "idxmin selects the first exact minimum in chronological order; near ties are listed separately",
        "worst_paired_difference": float(paired.paired_gain.min()),
        # Tie structure: the target series is piecewise constant, so distinct windows can settle identically.
        "tie_days_same_window": int((paired.same_window_as_reference & (paired.paired_gain.abs() <= 1e-6)).sum()),
        "tie_days_other_window": int((~paired.same_window_as_reference & (paired.paired_gain.abs() <= 1e-6)).sum()),
        "equal_adjacent_slot_share": float(np.mean(np.diff(canonical_a, axis=1) == 0)),
        "median_distinct_prices_per_day": float(np.median([len(np.unique(row)) for row in canonical_a])),
        "min_distinct_prices_per_day": int(min(len(np.unique(row)) for row in canonical_a)),
    }
    np.testing.assert_allclose(summary["headline"], 6693.875025301353, rtol=0, atol=1e-6)
    np.testing.assert_allclose(summary["reference"], 6214.361781989831, rtol=0, atol=1e-6)
    np.testing.assert_allclose(summary["paired_gain"], summary["nonensemble_remainder"]+summary["ensemble_increment"], atol=1e-9)
    assert summary["tie_days_same_window"] + summary["tie_days_other_window"] == summary["ties"]
    assert summary["wins"] + summary["ties"] + summary["losses"] == 59
    np.testing.assert_allclose(summary["gross_winning_day_total"] + summary["gross_losing_day_total"] +
        summary["tie_day_total_residual"], summary["total_gain"], rtol=0, atol=1e-6)
    save_json(out / "summary.json", summary)

    # The two point-loss panels were not produced under one fitting rule; record the difference
    # so the R1 text can state it instead of implying a single controlled candidate process.
    v1m = json.loads(source("reports/holdout_rmse_v1/manifest.json").read_text(encoding="utf-8"))
    v2m = json.loads(source("reports/holdout_rmse_v2/manifest.json").read_text(encoding="utf-8"))
    save_json(out / "candidate_panel_rules.json", {
        "v1": {"n_candidates": int(v1m["n_models"]), "protocol": v1m["protocol"],
               "tree_early_stopping": bool(v1m["tree_early_stopping"]),
               "lgb_xgb_catboost_rounds": v1m["tree_num_boost_round"],
               "histgb_rounds": 300,
               "early_stopping_used_for_ranking": bool(v1m["early_stopping_used_for_ranking"]),
               "ridge_alpha_fixed": v1m["ridge_alpha"],
               "sequence_inner_val_days": v1m["sequence_inner_val_days"],
               "sequence_fit_days": v1m["sequence"]["transformer"]["fit_days"],
               "hyperparameter_search": "fixed settings for trees/ridge; sequence models fit 287 days and use 14 days for early stopping"},
        "v2": {"n_candidates": int(v2m["n_models"]), "protocol": v2m["protocol"],
               "selection_split": v2m["selection_split"],
               "tree_early_stopping_rounds": v2m["tree_early_stopping_rounds"],
               "tree_round_cap": v2m["tree_round_cap"],
               "sequence_patience": v2m["sequence_patience"],
               "sequence_max_epochs": v2m["sequence_max_epochs"],
               "hyperparameter_search": "tuned candidates use val30 then refit on 301; fixed LightGBM baseline and weekday-slot mean are exceptions",
               "histgb_selection": "20 to 300 iterations in steps of 20; stop after 3 unimproved validation checks; final cap 300",
               "blend": v2m["selection"]["blend_lgb_transformer"],
               "blend_weight_selection": "validation RMSE grid; selected weights are fixed at evaluation"},
        "note": ("Both panels are scored on the same 59 days with the same solver, but they were not "
                 "produced by one selection process. They are reported separately and never pooled.")})
    bag_rows = []
    for bag in ["window_ce_lstm", "listwise_transformer", "spo_lstm"]:
        singles = np.array([series("holdout_seed_bag_v1", f"{bag}_seed{s}").realized.mean() for s in range(42,47)])
        result = series("holdout_seed_bag_v1", bag+"_mean")
        bag_rows.append({"bag": bag, "single_seed_mean": float(singles.mean()),
            "single_seed_sd": float(singles.std(ddof=1)), "single_seed_min": float(singles.min()),
            "single_seed_max": float(singles.max()), "curve_mean_score": float(result.realized.mean()),
            "ensemble_increment": float(result.realized.mean()-singles.mean()),
            "capture": float(result.realized.sum()/result.oracle.sum())})
    pd.DataFrame(bag_rows).to_csv(out / "ensemble_decomposition.csv", index=False)
    cp = source("reports/holdout_dispatch_loss_codex_v1/runs/20260914T221004Z/comparison_zh.csv")
    pd.read_csv(cp).to_csv(out / "controlled_comparisons.csv", index=False)
    cp_machine = source("reports/holdout_dispatch_loss_codex_v1/runs/20260914T221004Z/comparison.csv")
    pd.read_csv(cp_machine).to_csv(out / "controlled_comparisons_machine.csv", index=False)
    grid = json.loads(source("reports/holdout_dfl_ltr_v1/grid_runs.json").read_text(encoding="utf-8"))
    chosen = [r for r in grid if r["route"] == "r1_listwise" and r["family"] == "transformer" and
              r["params"] == {"lam": 2.0, "tau_tgt": .05, "tau_pred": .1}]
    val_rows = []
    for run in chosen:
        cap = max(run["history"], key=lambda r: r["val_capture"])
        val = max(run["history"], key=lambda r: r["val_realized"])
        val_rows.append({"seed": run["seed"], "capture_epoch": int(cap["epoch"]),
            "best_capture": cap["val_capture"], "realized_epoch": int(val["epoch"]),
            "best_realized": val["val_realized"], "realized_at_capture_epoch": cap["val_realized"]})
    save_json(out / "validation_selection_detail.json", {"runs": val_rows,
        "mean_best_capture": float(np.mean([r["best_capture"] for r in val_rows])),
        "mean_best_realized": float(np.mean([r["best_realized"] for r in val_rows])),
        "scope": "epoch-log aggregation, not independent replay of missing validation prediction curves"})
    for p in ["reports/holdout_seed_bag_v1/manifest.json", "reports/holdout_dfl_ltr_v1/selection.json",
              "reports/holdout_rmse_v1/manifest.json", "reports/holdout_rmse_v2/manifest.json",
              "reports/holdout_rmse_v1/dispatch_replay_manifest.json", "reports/holdout_rmse_v2/dispatch_replay_manifest.json",
              "reports/phase_b/final_metrics.json", "reports/phase_b/MODEL_CARD.md",
              "experiments/holdout_dispatch_loss_v1/loss.py", "experiments/holdout_dfl_ltr_v1/loss.py",
              "experiments/holdout_seed_bag_v1/run.py", "experiments/holdout_seed_bag_v1/combine.py",
              "experiments/holdout_dispatch_replay/run.py", "experiments/holdout_dispatch_replay/fit.py",
              "src/phase_b/dispatch.py", "src/phase_b/contracts.py", "src/phase_b/features.py",
              "src/phase_b/experiments.py", "experiments/holdout_rmse_v1/sequence_models.py"]:
        source(p)
    for p in ["experiments/holdout_rmse_v1/run.py", "experiments/holdout_rmse_v2/run.py",
              "experiments/holdout_dfl_ltr_v1/run.py", "experiments/holdout_dispatch_replay/recipes.py"]:
        source(p)
    for p in inputs:
        if sha(root / p["path"]) != p["sha256"]:
            raise RuntimeError("Input changed during analysis: " + p["path"])
    save_json(out / "source_manifest.json", {"sources": inputs, "all_inputs_unchanged_during_run": True,
        "packages": {k: importlib.metadata.version(k) for k in ["numpy","pandas","scipy","pyarrow"]},
        "notes": ["No fitting, checkpoint inference, or changes to historical result columns.",
                  "Packaged records are derived diagnostics, not raw price or feature data.",
                  "Historical leaderboard RMSE retained separately from same-prediction recomputation."]})
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
