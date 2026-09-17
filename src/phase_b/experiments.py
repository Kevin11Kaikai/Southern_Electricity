"""Reproducible, profit-oriented Phase B experiment runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import platform
import random
import subprocess
import sys
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn

from .config import PhaseBConfig
from .contracts import assert_valid_dispatch
from .data import (
    TARGET_COL,
    TIME_COL,
    audit_data,
    build_forecast_timeline,
    complete_day_index,
    frame_fingerprint,
    join_labels,
    load_raw_frames,
    supervised_complete_days,
)
from .dispatch import optimize_day
from .evaluation import contiguous_block_starts, evaluate_price_predictions
from .features import FeatureSets, build_features
from .modeling import (
    FoldModelResult,
    ModelSpec,
    deterministic_params,
    load_model_bundle,
    save_model_bundle,
    train_final_model,
    train_fold_model,
)
from .splits import DaySplitPlan, make_day_split_plan, mask_for_dates
from .weather import expand_hourly_to_quarters, extract_weather_directory


@dataclass(frozen=True)
class CandidateDefinition:
    name: str
    feature_set: str
    target_mode: str
    params: dict[str, Any]
    num_boost_round: int = 1200
    early_stopping_rounds: int = 80


@dataclass
class CandidateResult:
    definition: CandidateDefinition
    fold_metrics: pd.DataFrame
    daily: pd.DataFrame
    oof: pd.DataFrame

    @property
    def mean_profit(self) -> float:
        return float(self.daily["realized_profit"].mean())

    @property
    def mean_model_rmse(self) -> float:
        return float(self.fold_metrics["model_rmse"].mean())

    @property
    def median_best_iteration(self) -> int:
        return max(1, int(round(float(self.fold_metrics["best_iteration"].median()))))


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _feature_columns(sets: FeatureSets, name: str) -> list[str]:
    try:
        return list(getattr(sets, name))
    except AttributeError as exc:
        raise ValueError(f"Unknown feature set: {name}") from exc


def _candidate_spec(definition: CandidateDefinition) -> ModelSpec:
    return ModelSpec(
        name=definition.name,
        params=definition.params,
        num_boost_round=definition.num_boost_round,
        early_stopping_rounds=definition.early_stopping_rounds,
        target_mode=definition.target_mode,
    )


def run_candidate_cv(
    frame: pd.DataFrame,
    sets: FeatureSets,
    split_plan: DaySplitPlan,
    definition: CandidateDefinition,
) -> CandidateResult:
    feature_columns = _feature_columns(sets, definition.feature_set)
    spec = _candidate_spec(definition)
    fold_rows: list[dict[str, object]] = []
    daily_frames: list[pd.DataFrame] = []
    oof_frames: list[pd.DataFrame] = []
    for fold in split_plan.folds:
        _, prediction, model_metrics = train_fold_model(frame, feature_columns, fold, spec)
        validation = frame.loc[mask_for_dates(frame[TIME_COL], fold.validation_dates)].copy()
        validation = validation.sort_values(TIME_COL).reset_index(drop=True)
        validation["prediction"] = prediction
        evaluation = evaluate_price_predictions(
            validation,
            true_col=TARGET_COL,
            pred_col="prediction",
            incomplete="raise",
        )
        daily = evaluation["daily"].copy()
        daily["fold"] = fold.fold
        daily["candidate"] = definition.name
        daily_frames.append(daily)
        summary = evaluation["summary"]
        fold_rows.append(
            {
                **model_metrics.to_dict(),
                "candidate": definition.name,
                "feature_set": definition.feature_set,
                "target_mode": definition.target_mode,
                "model_rmse": model_metrics.rmse,
                "model_mae": model_metrics.mae,
                "mean_realized_profit": summary["mean_realized_profit"],
                "mean_oracle_profit": summary["mean_oracle_profit"],
                "oracle_capture_rate": summary["oracle_capture_rate"],
                "negative_profit_day_rate": summary["negative_profit_day_rate"],
            }
        )
        oof_frames.append(
            validation[[TIME_COL, TARGET_COL, "prediction"]].assign(
                fold=fold.fold,
                candidate=definition.name,
            )
        )
    return CandidateResult(
        definition=definition,
        fold_metrics=pd.DataFrame(fold_rows),
        daily=pd.concat(daily_frames, ignore_index=True),
        oof=pd.concat(oof_frames, ignore_index=True),
    )


def moving_block_bootstrap_difference(
    candidate_daily: pd.DataFrame,
    baseline_daily: pd.DataFrame,
    *,
    block_days: int = 7,
    samples: int = 4000,
    seed: int = 42,
    require_calendar_contiguous: bool = True,
) -> dict[str, float]:
    """Paired moving-block bootstrap of candidate minus baseline daily profit.

    ``require_calendar_contiguous`` keeps only blocks that span consecutive
    calendar days.  Positional slicing across a calendar gap stitches
    non-adjacent days into one block and breaks the serial-dependence
    assumption.  Pass ``False`` only to reproduce a superseded historical run.
    """

    left = candidate_daily[["date", "realized_profit"]].rename(columns={"realized_profit": "candidate"})
    right = baseline_daily[["date", "realized_profit"]].rename(columns={"realized_profit": "baseline"})
    paired = left.merge(right, on="date", how="inner", validate="one_to_one").sort_values("date")
    difference = (paired["candidate"] - paired["baseline"]).to_numpy(dtype=float)
    if len(difference) < block_days:
        raise ValueError("Not enough paired days for moving-block bootstrap")
    rng = np.random.default_rng(seed)
    positional = np.arange(len(difference) - block_days + 1)
    if require_calendar_contiguous:
        starts = contiguous_block_starts(paired["date"], block_days)
        if starts.size == 0:
            raise ValueError(
                "no calendar-contiguous block is available at "
                f"block_days={block_days}; the evaluated days are too fragmented"
            )
    else:
        starts = positional
    draws = np.empty(samples, dtype=float)
    blocks_needed = int(np.ceil(len(difference) / block_days))
    for index in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([difference[start : start + block_days] for start in chosen])[: len(difference)]
        draws[index] = sample.mean()
    return {
        "mean_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "paired_days": int(len(difference)),
        "block_days": int(block_days),
        "samples": int(samples),
        "block_start_rule": (
            "calendar_contiguous" if require_calendar_contiguous else "positional_legacy"
        ),
        "block_starts_used": int(starts.size),
        "block_starts_dropped_for_calendar_gap": int(positional.size - starts.size),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def prepare_feature_frame(
    config: PhaseBConfig,
    *,
    rebuild_weather: bool = False,
) -> tuple[pd.DataFrame, FeatureSets, dict[str, object]]:
    train_features, labels, test_features = load_raw_frames(config)
    timeline = build_forecast_timeline(train_features, test_features)
    output_dir = config.path("output_dir")
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    weather_cache = cache_dir / "weather_hourly.csv.gz"
    weather_audit_path = cache_dir / "weather_audit.json"
    weather_config = config.section("weather")
    if rebuild_weather or not weather_cache.exists():
        hourly, weather_audit = extract_weather_directory(
            config.path("weather_dir"),
            tile_rows=int(weather_config["tile_rows"]),
            tile_cols=int(weather_config["tile_cols"]),
        )
        hourly.to_csv(weather_cache, index=False, compression="gzip")
        weather_audit_path.write_text(
            json.dumps(weather_audit.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        hourly = pd.read_csv(weather_cache, parse_dates=[TIME_COL, "weather_issue_time_utc"])
        weather_audit = json.loads(weather_audit_path.read_text(encoding="utf-8"))
    quarters = expand_hourly_to_quarters(hourly)
    feature_config = config.section("features")
    feature_frame, sets = build_features(
        timeline,
        lags=feature_config["lags"],
        rolling_windows=feature_config["rolling_windows"],
        weather=quarters,
    )
    feature_frame = join_labels(feature_frame, labels)
    audit = {
        "data": audit_data(config).to_dict(),
        "weather": weather_audit.to_dict() if hasattr(weather_audit, "to_dict") else weather_audit,
        "feature_frame_rows": int(len(feature_frame)),
        "feature_set_sizes": {key: len(getattr(sets, key)) for key in ("baseline", "contextual", "temporal", "full")},
        "feature_fingerprint": frame_fingerprint(feature_frame, [TIME_COL, *list(sets.baseline)]),
    }
    return feature_frame, sets, audit


def _default_candidates(seed: int, n_jobs: int) -> list[CandidateDefinition]:
    baseline = deterministic_params(
        seed=seed,
        n_jobs=n_jobs,
        learning_rate=0.05,
        num_leaves=63,
        feature_fraction=1.0,
        bagging_fraction=1.0,
        bagging_freq=0,
    )
    regularized = deterministic_params(seed=seed, n_jobs=n_jobs)
    return [
        CandidateDefinition("baseline_level", "baseline", "level", baseline, 1000, 50),
        CandidateDefinition("contextual_level", "contextual", "level", regularized),
        CandidateDefinition("temporal_level", "temporal", "level", regularized),
        CandidateDefinition("temporal_shape", "temporal", "day_centered", regularized),
        CandidateDefinition("weather_level", "full", "level", regularized),
        CandidateDefinition("weather_shape", "full", "day_centered", regularized),
    ]


def _search_definitions(base: CandidateDefinition, seed: int, n_jobs: int) -> list[CandidateDefinition]:
    grid = [
        {"num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.03},
        {"num_leaves": 63, "min_child_samples": 10, "learning_rate": 0.03},
        {"num_leaves": 63, "min_child_samples": 40, "learning_rate": 0.03},
        {"num_leaves": 127, "min_child_samples": 20, "learning_rate": 0.02},
        {"num_leaves": 31, "min_child_samples": 40, "learning_rate": 0.05},
        {"num_leaves": 63, "min_child_samples": 20, "learning_rate": 0.05, "lambda_l2": 1.0},
    ]
    definitions: list[CandidateDefinition] = []
    for index, overrides in enumerate(grid, start=1):
        params = deterministic_params(seed=seed, n_jobs=n_jobs, **overrides)
        definitions.append(
            CandidateDefinition(
                name=f"search_{index:02d}_{base.feature_set}_{base.target_mode}",
                feature_set=base.feature_set,
                target_mode=base.target_mode,
                params=params,
            )
        )
    return definitions


def _summary_row(result: CandidateResult) -> dict[str, object]:
    return {
        "candidate": result.definition.name,
        "feature_set": result.definition.feature_set,
        "target_mode": result.definition.target_mode,
        "feature_count": None,
        "mean_oof_profit": result.mean_profit,
        "mean_oracle_profit": float(result.daily["oracle_profit"].mean()),
        "oracle_capture_rate": float(result.daily["realized_profit"].sum() / result.daily["oracle_profit"].sum()),
        "negative_profit_day_rate": float((result.daily["realized_profit"] < 0).mean()),
        "mean_model_rmse": result.mean_model_rmse,
        "median_best_iteration": result.median_best_iteration,
    }


def run_phase_b_experiments(
    config: PhaseBConfig,
    *,
    rebuild_weather: bool = False,
    skip_search: bool = False,
) -> dict[str, object]:
    set_global_determinism(config.seed)
    output_dir = config.path("output_dir")
    report_dir = config.path("report_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    protected = [
        config.path("train_features"),
        config.path("train_labels"),
        config.path("test_features"),
        config.path("submission_template"),
        config.path("train_features").parents[3] / "lgb_baseline.py",
        config.path("train_features").parents[3] / "analyze_baseline.py",
    ]
    before_hashes = {str(path.relative_to(config.path("train_features").parents[3])): _sha256(path) for path in protected}

    frame, sets, audit = prepare_feature_frame(config, rebuild_weather=rebuild_weather)
    complete_training = supervised_complete_days(frame.loc[frame["dataset_split"] == "train"])
    complete_dates = complete_day_index(complete_training)
    validation_config = config.section("validation")
    split_plan = make_day_split_plan(
        complete_dates,
        n_splits=int(validation_config["n_splits"]),
        validation_days=int(validation_config["validation_days"]),
        holdout_days=int(validation_config["holdout_days"]),
    )
    n_jobs = int(config.section("model")["n_jobs"])

    results: list[CandidateResult] = []
    for definition in _default_candidates(config.seed, n_jobs):
        print(f"CV candidate: {definition.name}", flush=True)
        results.append(run_candidate_cv(complete_training, sets, split_plan, definition))

    baseline_result = next(result for result in results if result.definition.name == "baseline_level")
    selected = max(results, key=lambda result: result.mean_profit)
    search_results: list[CandidateResult] = []
    if not skip_search:
        for definition in _search_definitions(selected.definition, config.seed, n_jobs):
            print(f"CV search trial: {definition.name}", flush=True)
            search_results.append(run_candidate_cv(complete_training, sets, split_plan, definition))
        results.extend(search_results)
        selected = max(results, key=lambda result: result.mean_profit)

    level_results = [result for result in results if result.definition.target_mode == "level"]
    price_result = min(level_results, key=lambda result: result.mean_model_rmse)

    # The holdout is revealed once, after all feature/model choices are frozen.
    dev = complete_training.loc[mask_for_dates(complete_training[TIME_COL], split_plan.development_dates)]
    holdout = complete_training.loc[mask_for_dates(complete_training[TIME_COL], split_plan.holdout_dates)].copy()
    baseline_model = train_final_model(
        dev,
        _feature_columns(sets, baseline_result.definition.feature_set),
        _candidate_spec(baseline_result.definition),
        num_boost_round=baseline_result.median_best_iteration,
    )
    selected_model = train_final_model(
        dev,
        _feature_columns(sets, selected.definition.feature_set),
        _candidate_spec(selected.definition),
        num_boost_round=selected.median_best_iteration,
    )
    holdout["baseline_prediction"] = baseline_model.predict(
        holdout[_feature_columns(sets, baseline_result.definition.feature_set)]
    )
    holdout["selected_prediction"] = selected_model.predict(
        holdout[_feature_columns(sets, selected.definition.feature_set)]
    )
    baseline_holdout = evaluate_price_predictions(
        holdout, true_col=TARGET_COL, pred_col="baseline_prediction", incomplete="raise"
    )
    selected_holdout = evaluate_price_predictions(
        holdout, true_col=TARGET_COL, pred_col="selected_prediction", incomplete="raise"
    )

    # Refit frozen models on every complete training day for test inference.
    decision_features = _feature_columns(sets, selected.definition.feature_set)
    price_features = _feature_columns(sets, price_result.definition.feature_set)
    final_decision_model = train_final_model(
        complete_training,
        decision_features,
        _candidate_spec(selected.definition),
        num_boost_round=selected.median_best_iteration,
    )
    final_price_model = train_final_model(
        complete_training,
        price_features,
        _candidate_spec(price_result.definition),
        num_boost_round=price_result.median_best_iteration,
    )
    models_dir = output_dir / "models"
    decision_model_path, decision_manifest_path = save_model_bundle(
        final_decision_model,
        models_dir,
        feature_columns=decision_features,
        spec=_candidate_spec(selected.definition),
        metadata={"role": "dispatch_decision_signal", "best_iteration": selected.median_best_iteration},
    )
    price_model_path, price_manifest_path = save_model_bundle(
        final_price_model,
        models_dir,
        feature_columns=price_features,
        spec=_candidate_spec(price_result.definition),
        metadata={"role": "submission_price", "best_iteration": price_result.median_best_iteration},
    )

    test = frame.loc[frame["dataset_split"] == "test"].sort_values(TIME_COL).copy()
    test["decision_signal"] = final_decision_model.predict(test[decision_features])
    test["pred_price"] = final_price_model.predict(test[price_features])
    power_parts: list[np.ndarray] = []
    for _, day in test.groupby(test[TIME_COL].dt.normalize(), sort=True):
        if len(day) != 96:
            raise ValueError(f"Test day {day[TIME_COL].iloc[0].date()} has {len(day)} rows, expected 96")
        power_parts.append(optimize_day(day["decision_signal"].to_numpy()).power)
    test["power"] = np.concatenate(power_parts)
    assert_valid_dispatch(test[TIME_COL], test["power"])
    submission = test[[TIME_COL, "pred_price", "power"]].rename(columns={"pred_price": "实时价格"})
    template = pd.read_csv(config.path("submission_template"), parse_dates=[TIME_COL])
    if list(submission.columns) != [TIME_COL, "实时价格", "power"]:
        raise AssertionError("Submission columns do not match the official template")
    if len(submission) != len(template) or not submission[TIME_COL].reset_index(drop=True).equals(
        template[TIME_COL].reset_index(drop=True)
    ):
        raise ValueError("Submission timestamps do not match output_demo.csv")
    submission_path = output_dir / "submission_phase_b.csv"
    submission.to_csv(submission_path, index=False, encoding="utf-8-sig")

    loaded_decision, loaded_manifest = load_model_bundle(decision_model_path, decision_manifest_path)
    reload_prediction = loaded_decision.predict(test[loaded_manifest["feature_columns"]])
    reload_abs_diff = float(np.max(np.abs(reload_prediction - test["decision_signal"].to_numpy())))

    all_fold_metrics = pd.concat([result.fold_metrics for result in results], ignore_index=True)
    all_daily = pd.concat([result.daily for result in results], ignore_index=True)
    candidate_summary = pd.DataFrame([_summary_row(result) for result in results])
    feature_sizes = {key: len(getattr(sets, key)) for key in ("baseline", "contextual", "temporal", "full")}
    candidate_summary["feature_count"] = candidate_summary["feature_set"].map(feature_sizes)
    paired_bootstrap = moving_block_bootstrap_difference(selected.daily, baseline_result.daily, seed=config.seed)
    fold_comparison = (
        selected.fold_metrics[["fold", "mean_realized_profit"]]
        .rename(columns={"mean_realized_profit": "selected_profit"})
        .merge(
            baseline_result.fold_metrics[["fold", "mean_realized_profit"]].rename(
                columns={"mean_realized_profit": "baseline_profit"}
            ),
            on="fold",
            validate="one_to_one",
        )
    )
    fold_wins = int((fold_comparison["selected_profit"] > fold_comparison["baseline_profit"]).sum())
    baseline_mean = baseline_result.mean_profit
    improvement_pct = float((selected.mean_profit - baseline_mean) / abs(baseline_mean) * 100) if baseline_mean else float("nan")
    holdout_base_mean = float(baseline_holdout["summary"]["mean_realized_profit"])
    holdout_selected_mean = float(selected_holdout["summary"]["mean_realized_profit"])
    holdout_improvement_pct = (
        float((holdout_selected_mean - holdout_base_mean) / abs(holdout_base_mean) * 100)
        if holdout_base_mean
        else float("nan")
    )
    acceptance = config.section("acceptance")
    strong_gate = bool(
        improvement_pct >= float(acceptance["min_profit_improvement_pct"])
        and fold_wins >= int(acceptance["min_fold_wins"])
        and paired_bootstrap["ci95_low"] > 0
        and holdout_selected_mean > holdout_base_mean
        and reload_abs_diff <= float(acceptance["max_reload_abs_diff"])
    )

    before_after_hashes = {
        path: {"before": digest, "after": _sha256(config.path("train_features").parents[3] / path)}
        for path, digest in before_hashes.items()
    }
    raw_metrics = {
        "status": "verified" if strong_gate else "experimental",
        "selected_decision_candidate": selected.definition.name,
        "selected_price_candidate": price_result.definition.name,
        "oof": {
            "baseline_mean_daily_profit": baseline_mean,
            "selected_mean_daily_profit": selected.mean_profit,
            "profit_improvement_pct": improvement_pct,
            "fold_wins": fold_wins,
            "bootstrap": paired_bootstrap,
        },
        "holdout": {
            "days": len(split_plan.holdout_dates),
            "baseline": baseline_holdout["summary"],
            "selected": selected_holdout["summary"],
            "profit_improvement_pct": holdout_improvement_pct,
        },
        "reproducibility": {"model_reload_max_abs_diff": reload_abs_diff},
        "submission": {
            "path": str(submission_path.relative_to(config.path("train_features").parents[3])),
            "rows": len(submission),
            "days": int(submission[TIME_COL].dt.normalize().nunique()),
            "constraint_violations": 0,
        },
        "strong_hr_gate_passed": strong_gate,
        "data_audit": audit,
    }
    run_manifest = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
        "seed": config.seed,
        "config": str(config.source.relative_to(config.path("train_features").parents[3])),
        "git_commit_before_changes": _git_commit(config.path("train_features").parents[3]),
        "protected_hashes": before_after_hashes,
    }

    all_fold_metrics.to_csv(report_dir / "fold_metrics.csv", index=False)
    all_daily.to_csv(report_dir / "daily_candidate_metrics.csv", index=False)
    candidate_summary.sort_values("mean_oof_profit", ascending=False).to_csv(
        report_dir / "candidate_summary.csv", index=False
    )
    fold_comparison.to_csv(report_dir / "selected_vs_baseline_folds.csv", index=False)
    baseline_holdout["daily"].assign(candidate="baseline_level").to_csv(
        report_dir / "holdout_baseline_daily.csv", index=False
    )
    selected_holdout["daily"].assign(candidate=selected.definition.name).to_csv(
        report_dir / "holdout_selected_daily.csv", index=False
    )
    selected.oof.to_csv(output_dir / "selected_oof_predictions.csv", index=False)
    price_result.oof.to_csv(output_dir / "price_oof_predictions.csv", index=False)
    (report_dir / "metrics.json").write_text(
        json.dumps(raw_metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (report_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return raw_metrics

