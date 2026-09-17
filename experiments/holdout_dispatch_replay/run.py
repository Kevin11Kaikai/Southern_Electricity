"""Train once, freeze artifacts, and append 59-day dispatch profits to v1/v2 leaderboards."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[2]
TORCH_RUNTIME = ROOT / ".torch_runtime"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
if TORCH_RUNTIME.exists() and str(TORCH_RUNTIME) not in sys.path:
    sys.path.insert(0, str(TORCH_RUNTIME))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.holdout_dispatch_replay import artifacts  # noqa: E402
from experiments.holdout_dispatch_replay.fit import fit_and_save, load_yhat_from_checkpoint  # noqa: E402
from experiments.holdout_dispatch_replay.recipes import RMSE_DRIFT_TOL, parse_model_name, v1_recipe, v2_recipe  # noqa: E402
from src.phase_b.config import load_config  # noqa: E402
from src.phase_b.data import TARGET_COL, TIME_COL, complete_day_index  # noqa: E402
from src.phase_b.evaluation import evaluate_price_predictions  # noqa: E402
from src.phase_b.experiments import prepare_feature_frame  # noqa: E402
from src.phase_b.features import FeatureSets  # noqa: E402
from src.phase_b.splits import make_day_split_plan, mask_for_dates  # noqa: E402

SEED = 42
HOLD_OUT_DAYS = 59
DEV_DAYS = 301
VAL_DAYS = 30
FIT_DAYS = DEV_DAYS - VAL_DAYS
V1_REPORT = ROOT / "reports" / "holdout_rmse_v1"
V2_REPORT = ROOT / "reports" / "holdout_rmse_v2"
PHASE_B_METRICS = ROOT / "reports" / "phase_b" / "final_metrics.json"
PROFIT_COLUMNS = [
    "pred_profit_mean",
    "realized_profit_mean",
    "profit_gap",
    "oracle_profit_mean",
    "capture",
]


def _feature_columns(sets: FeatureSets, name: str | None) -> list[str]:
    if name is None:
        return []
    return list(getattr(sets, name))


def holdout_rmse(y_true: np.ndarray, yhat: np.ndarray) -> float:
    return float(mean_squared_error(np.asarray(y_true, dtype=float), np.asarray(yhat, dtype=float)) ** 0.5)


def holdout_mae(y_true: np.ndarray, yhat: np.ndarray) -> float:
    return float(mean_absolute_error(np.asarray(y_true, dtype=float), np.asarray(yhat, dtype=float)))


def score_predictions(
    times: pd.Series,
    y_true: np.ndarray,
    yhat: np.ndarray,
) -> dict[str, Any]:
    report = evaluate_price_predictions(times, y_true, yhat, incomplete="raise")
    daily = report["daily"]
    summary = report["summary"]
    pred_mean = float(daily["predicted_profit"].mean())
    realized = float(summary["mean_realized_profit"])
    oracle = float(summary["mean_oracle_profit"])
    capture = float(realized / oracle) if oracle else float("nan")
    return {
        "pred_profit_mean": pred_mean,
        "realized_profit_mean": realized,
        "profit_gap": pred_mean - realized,
        "oracle_profit_mean": oracle,
        "capture": capture,
        "days_evaluated": int(summary["days_evaluated"]),
        "daily": daily,
    }


def _format_float(value: object, digits: int = 6) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return f"{float(value):.{digits}f}"


def _format_int(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(int(value))


def write_leaderboard(path: Path, frame: pd.DataFrame, original_columns: list[str]) -> None:
    display = frame.copy()
    float_cols = [col for col in ("RMSE", "MAE", "val30_RMSE", *PROFIT_COLUMNS) if col in display.columns]
    int_cols = [col for col in ("n_points", "best_rounds_or_epochs") if col in display.columns]
    for column in float_cols:
        display[column] = display[column].map(_format_float)
    for column in int_cols:
        display[column] = display[column].map(_format_int)
    ordered = original_columns + [col for col in PROFIT_COLUMNS if col not in original_columns]
    display[ordered].to_csv(path, index=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def fit_order(names: list[str]) -> list[str]:
    blend = [name for name in names if name == "blend_lgb_transformer"]
    rest = [name for name in names if name != "blend_lgb_transformer"]
    head = [name for name in ("lgb_contextual", "transformer_contextual") if name in rest]
    tail = [name for name in rest if name not in head]
    return head + tail + blend


def ok_rank_order(frame: pd.DataFrame, rmse_col: str) -> list[str]:
    ok = frame.loc[frame["status"] == "ok"].copy()
    ok["_rank"] = ok[rmse_col].rank(method="min")
    return ok.sort_values(["_rank", "model"], kind="mergesort")["model"].tolist()


def load_split(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    seed = int(config.seed)
    n_jobs = int(config.section("model")["n_jobs"])
    print("[holdout_dispatch_replay] preparing feature frame", flush=True)
    feature_frame, sets, audit = prepare_feature_frame(config)
    labeled = feature_frame.loc[feature_frame[TARGET_COL].notna()].copy()
    complete = complete_day_index(labeled)
    plan = make_day_split_plan(
        complete,
        n_splits=int(config.section("validation")["n_splits"]),
        validation_days=int(config.section("validation")["validation_days"]),
        holdout_days=HOLD_OUT_DAYS,
    )
    development_dates = plan.development_dates
    if int(development_dates.size) != DEV_DAYS or int(plan.holdout_dates.size) != HOLD_OUT_DAYS:
        raise AssertionError(
            f"expected {DEV_DAYS}/{HOLD_OUT_DAYS} split, got {development_dates.size}/{plan.holdout_dates.size}"
        )
    fit_dates = development_dates[:FIT_DAYS]
    val_dates = development_dates[-VAL_DAYS:]
    development = labeled.loc[mask_for_dates(labeled[TIME_COL], development_dates)].sort_values(TIME_COL).reset_index(drop=True)
    fit = labeled.loc[mask_for_dates(labeled[TIME_COL], fit_dates)].sort_values(TIME_COL).reset_index(drop=True)
    val = labeled.loc[mask_for_dates(labeled[TIME_COL], val_dates)].sort_values(TIME_COL).reset_index(drop=True)
    holdout = labeled.loc[mask_for_dates(labeled[TIME_COL], plan.holdout_dates)].sort_values(TIME_COL).reset_index(drop=True)
    return {
        "seed": seed,
        "n_jobs": n_jobs,
        "sets": sets,
        "audit": audit,
        "development": development,
        "fit": fit,
        "val": val,
        "holdout": holdout,
        "plan": plan,
    }


def _prediction_store(report_dir: Path) -> pd.DataFrame | None:
    parquet = report_dir / "predictions.parquet"
    csv_path = report_dir / "predictions.csv"
    if parquet.exists() or csv_path.exists():
        return artifacts.read_predictions(parquet if parquet.exists() else csv_path)
    return None


def obtain_predictions(
    *,
    version: str,
    name: str,
    recipe: dict[str, Any],
    split: dict[str, Any],
    report_dir: Path,
    cached: dict[str, np.ndarray],
) -> tuple[np.ndarray, str, dict[str, Any]]:
    holdout = split["holdout"]
    store = _prediction_store(report_dir)
    if store is not None and name in store.columns:
        print(f"[holdout_dispatch_replay] {version}/{name} loaded predictions parquet", flush=True)
        return store[name].to_numpy(dtype=float), "predictions", {}

    directory = artifacts.checkpoint_dir(version, name)
    loaded = load_yhat_from_checkpoint(
        directory,
        train=split["development"],
        test=holdout,
        cached=cached,
    )
    if loaded is not None:
        print(f"[holdout_dispatch_replay] {version}/{name} loaded checkpoint", flush=True)
        return loaded, "checkpoint", {}

    family, feature_set = parse_model_name(name)
    columns = _feature_columns(split["sets"], feature_set if family not in {"climatology", "blend", "dlinear_prevday"} else None)
    if family in {"lgb", "xgboost", "catboost", "hgb", "ridge", "lstm", "transformer", "dlinear"}:
        columns = _feature_columns(split["sets"], recipe.get("feature_set") or feature_set)

    print(f"[holdout_dispatch_replay] fitting {version}/{name}", flush=True)
    yhat, extra = fit_and_save(
        recipe,
        train=split["development"],
        test=holdout,
        columns=columns or None,
        seed=split["seed"],
        n_jobs=split["n_jobs"],
        directory=directory,
        fit_frame=split["fit"],
        val_frame=split["val"],
        cached=cached,
    )
    return yhat, "fit", extra


def replay_version(
    version: str,
    report_dir: Path,
    split: dict[str, Any],
    *,
    selection: dict[str, Any] | None,
) -> dict[str, Any]:
    leaderboard_path = report_dir / "leaderboard.csv"
    board = pd.read_csv(leaderboard_path)
    original_columns = list(board.columns)
    holdout = split["holdout"]
    y_true = holdout[TARGET_COL].astype(float).to_numpy()
    times = holdout[TIME_COL]
    cached: dict[str, np.ndarray] = {}
    drift: dict[str, Any] = {}
    sources: dict[str, str] = {}
    extras: dict[str, Any] = {}
    daily_parts: list[pd.DataFrame] = []
    refit_rmse: dict[str, float] = {}

    for column in PROFIT_COLUMNS:
        if column not in board.columns:
            board[column] = np.nan

    rows_by_name = {str(row["model"]): (index, row) for index, row in board.iterrows()}
    names = [str(row["model"]) for _, row in board.iterrows() if row["status"] == "ok"]
    for name in fit_order(names):
        index, row = rows_by_name[name]
        recipe = v1_recipe(name) if version == "v1" else v2_recipe(name, leaderboard_row=row, selection=selection or {})
        try:
            yhat, source, extra = obtain_predictions(
                version=version,
                name=name,
                recipe=recipe,
                split=split,
                report_dir=report_dir,
                cached=cached,
            )
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            extras[name] = {"status": f"failed: {type(exc).__name__}: {exc}"}
            print(f"[holdout_dispatch_replay] {version}/{name} failed: {exc}", flush=True)
            continue

        if yhat.shape[0] != y_true.shape[0]:
            raise ValueError(f"{name} prediction length {yhat.shape[0]} != holdout {y_true.shape[0]}")
        cached[name] = yhat
        sources[name] = source
        if extra:
            extras[name] = extra
        artifacts.upsert_prediction_column(report_dir / "predictions.parquet", times, y_true, name, yhat)

        rmse = holdout_rmse(y_true, yhat)
        mae = holdout_mae(y_true, yhat)
        refit_rmse[name] = rmse
        saved_rmse = float(row["RMSE"])
        delta = rmse - saved_rmse
        if abs(delta) > RMSE_DRIFT_TOL:
            drift[name] = {"saved_rmse": saved_rmse, "refit_rmse": rmse, "delta": delta, "refit_mae": mae}
            print(
                f"[holdout_dispatch_replay] {version}/{name} rmse_drift saved={saved_rmse:.6f} "
                f"refit={rmse:.6f} delta={delta:+.6f}",
                flush=True,
            )

        scored = score_predictions(times, y_true, yhat)
        if scored["days_evaluated"] != HOLD_OUT_DAYS:
            raise AssertionError(f"{name} evaluated {scored['days_evaluated']} days, expected {HOLD_OUT_DAYS}")
        for column in PROFIT_COLUMNS:
            board.at[index, column] = scored[column]
        daily = scored["daily"].copy()
        daily.insert(0, "model", name)
        daily_parts.append(daily)
        print(
            f"[holdout_dispatch_replay] {version}/{name} realized={scored['realized_profit_mean']:.2f} "
            f"pred={scored['pred_profit_mean']:.2f} capture={scored['capture']:.4f} via {source}",
            flush=True,
        )

    saved_order = ok_rank_order(pd.read_csv(leaderboard_path), "RMSE")
    refit_frame = pd.DataFrame({"model": list(refit_rmse), "RMSE": list(refit_rmse.values()), "status": "ok"})
    refit_order = ok_rank_order(refit_frame, "RMSE") if not refit_frame.empty else []
    saved_sub = [name for name in saved_order if name in refit_rmse]
    rank_would_change = saved_sub != refit_order
    if rank_would_change:
        print(
            f"[holdout_dispatch_replay] WARNING {version} refit RMSE rank would change: "
            f"saved={saved_order} refit={refit_order}",
            flush=True,
        )

    write_leaderboard(leaderboard_path, board, original_columns)
    if daily_parts:
        daily_all = pd.concat(daily_parts, ignore_index=True)
        daily_all.to_csv(report_dir / "dispatch_daily.csv", index=False)

    predictions_path = report_dir / "predictions.parquet"
    if not predictions_path.exists():
        predictions_path = report_dir / "predictions.csv"

    manifest = {
        "experiment": f"holdout_dispatch_replay_{version}",
        "holdout_not_used_for_selection": True,
        "online_replay": True,
        "rl": False,
        "optimize_day_unchanged": True,
        "phase_b_final_metrics_untouched": True,
        "seed": split["seed"],
        "profit_units": "official_score_sum_price_times_power",
        "predictions_path": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
        "checkpoint_dir": str((artifacts.CHECKPOINT_ROOT / version).relative_to(ROOT)).replace("\\", "/"),
        "sources": sources,
        "rmse_drift": drift,
        "rank_would_change": rank_would_change,
        "saved_rmse_rank": saved_order,
        "refit_rmse_rank": refit_order,
        "extras": extras,
        "n_scored": len(daily_parts),
    }
    artifacts.write_json(report_dir / "dispatch_replay_manifest.json", json_safe(manifest))
    return manifest


def assert_phase_b_untouched() -> None:
    if not PHASE_B_METRICS.exists():
        raise FileNotFoundError(PHASE_B_METRICS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Attach 59-day dispatch profits and freeze artifacts")
    parser.add_argument("--config", default=str(ROOT / "configs" / "phase_b.toml"))
    parser.add_argument("--only", choices=("v1", "v2", "both"), default="both")
    args = parser.parse_args(argv)

    assert_phase_b_untouched()
    split = load_split(args.config)
    v2_manifest = json.loads((V2_REPORT / "manifest.json").read_text(encoding="utf-8"))
    selection = v2_manifest.get("selection") or {}

    manifests: dict[str, Any] = {}
    if args.only in {"v1", "both"}:
        manifests["v1"] = replay_version("v1", V1_REPORT, split, selection=None)
    if args.only in {"v2", "both"}:
        manifests["v2"] = replay_version("v2", V2_REPORT, split, selection=selection)

    if "v1" in manifests:
        print(f"wrote {V1_REPORT / 'leaderboard.csv'}")
        print(f"wrote {V1_REPORT / 'dispatch_replay_manifest.json'}")
    if "v2" in manifests:
        print(f"wrote {V2_REPORT / 'leaderboard.csv'}")
        print(f"wrote {V2_REPORT / 'dispatch_replay_manifest.json'}")
    for version, manifest in manifests.items():
        print(f"{version} scored={manifest['n_scored']} rank_would_change={manifest['rank_would_change']}")
        if manifest["rmse_drift"]:
            print(f"{version} rmse_drift models={list(manifest['rmse_drift'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
