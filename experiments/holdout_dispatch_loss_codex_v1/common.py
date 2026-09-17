"""Small run ledger, deterministic construction, and read-only data adapter."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from . import ROOT
from experiments.holdout_rmse_v1.sequence_models import (
    CompactDayTransformer, CompactDayLSTM, CompactDLinear,
)
from src.phase_b.dispatch import optimize_day

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "protocol.json"


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def construct_model(family, n_features, seed=42):
    # In the original Cursor runner the caller constructed a model BEFORE
    # setting the seed. Seed before construction here, for every comparison.
    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = {"transformer": CompactDayTransformer, "lstm": CompactDayLSTM, "dlinear": CompactDLinear}
    return classes[family](n_features)


def mean_realized(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 2 or actual.shape[1] != 96:
        raise ValueError("expected matching (days,96) arrays")
    return float(np.mean([np.dot(y, optimize_day(p).power) for y, p in zip(actual, predicted)]))


def scaled_count(epoch):
    return min(80, max(1, int(round(epoch * 301 / 271))))


class Ledger:
    def __init__(self, run_dir, protocol):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / "ledger.jsonl"
        self.protocol = protocol

    def rows(self):
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]

    def add(self, event, **data):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"time_utc": utcnow(), "event": event, **data}, ensure_ascii=False, allow_nan=False) + "\n")

    def remaining_seconds(self):
        return (datetime.fromisoformat(self.protocol["deadline_utc"]) - datetime.now(timezone.utc)).total_seconds()

    def launches(self):
        return sum(row["event"] == "training_start" for row in self.rows())

    def check_start(self, expected_seconds=360):
        if self.launches() >= self.protocol["training_launch_limit"]:
            raise RuntimeError("training launch budget exhausted")
        if self.remaining_seconds() < expected_seconds + 120:
            raise RuntimeError("remaining wall-clock budget cannot cover this job and closeout")


def guard_paths():
    """An explicit allowlist only. Never enumerate the whole repository."""
    names = [
        "lgb_baseline.py", "analyze_baseline.py", "output_demo.csv", "configs/phase_b.toml",
        "to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv",
        "to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv",
        "to_sais_new/to_sais_new/test/test_in_feature_ori.csv",
        "src/phase_b/data.py", "src/phase_b/features.py", "src/phase_b/dispatch.py",
        "src/phase_b/contracts.py", "src/phase_b/evaluation.py", "src/phase_b/experiments.py",
        "experiments/holdout_rmse_v1/sequence_models.py",
        "experiments/holdout_dispatch_loss_v1/loss.py", "experiments/holdout_dispatch_loss_v1/run.py",
        "notebooks/11_failure_analysis.ipynb", "notebooks/12_dispatch_loss_trial.ipynb",
        "reports/phase_b/final_metrics.json", "reports/phase_b/final_run_manifest.json",
        "outputs/phase_b/cache/weather_hourly.csv.gz", "outputs/phase_b/cache/weather_audit.json",
    ]
    for folder in ["reports/holdout_rmse_v1", "reports/holdout_rmse_v2", "reports/holdout_dispatch_loss_v1"]:
        names.extend(f"{folder}/{name}" for name in ["leaderboard.csv", "predictions.parquet", "predictions.csv", "dispatch_daily.csv", "manifest.json", "selection.json", "dispatch_replay_manifest.json"])
    return [name for name in names if (ROOT / name).is_file()]


def prepare_data(run_dir):
    from src.phase_b.config import load_config
    from src.phase_b.experiments import prepare_feature_frame
    from src.phase_b.data import TIME_COL, TARGET_COL, complete_day_index, frame_fingerprint

    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if (data_dir / "manifest.json").exists():
        return read_json(data_dir / "manifest.json")
    guards = {name: sha256(ROOT / name) for name in guard_paths()}
    write_json(run_dir / "protected_hashes_before.json", guards)
    for name in ["weather_hourly.csv.gz", "weather_audit.json"]:
        if not (ROOT / "outputs/phase_b/cache" / name).exists():
            raise RuntimeError("existing weather cache missing; no download/rebuild permitted")
    started = time.perf_counter()
    frame, sets, audit = prepare_feature_frame(load_config(ROOT / "configs/phase_b.toml"))
    reference = read_json(ROOT / "reports/holdout_dispatch_loss_v1/manifest.json")
    if audit["feature_fingerprint"] != reference["feature_audit"]["feature_fingerprint"]:
        raise AssertionError("feature fingerprint differs from Cursor")
    complete = complete_day_index(frame)
    if len(complete) != 360:
        raise AssertionError(f"expected 360 complete label days, got {len(complete)}")
    development_dates, holdout_dates = complete[:301], complete[301:]
    expected = [(development_dates[0], "2025-01-02"), (development_dates[270], "2025-10-03"),
                (development_dates[271], "2025-10-04"), (development_dates[-1], "2025-11-02"),
                (holdout_dates[0], "2025-11-03"), (holdout_dates[-1], "2025-12-31")]
    for value, date in expected:
        if str(value.date()) != date:
            raise AssertionError(f"date mismatch: {value} vs {date}")
    columns = list(sets.contextual)
    if len(columns) != 67 or any("实际值" in col or col == "A" for col in columns):
        raise AssertionError("contextual feature contract failed")

    def subset(dates):
        return frame.loc[frame[TIME_COL].dt.normalize().isin(dates)].sort_values(TIME_COL).reset_index(drop=True)

    development, holdout = subset(development_dates), subset(holdout_dates)
    x_dev = development[columns].to_numpy(dtype=np.float32).reshape(301, 96, 67)
    y_dev = development[TARGET_COL].to_numpy(dtype=float).reshape(301, 96)
    x_test = holdout[columns].to_numpy(dtype=np.float32).reshape(59, 96, 67)
    y_test = holdout[TARGET_COL].to_numpy(dtype=float).reshape(59, 96)
    np.savez_compressed(data_dir / "development.npz", x=x_dev, y=y_dev, dates=development_dates.to_numpy(dtype="datetime64[ns]"))
    np.savez_compressed(data_dir / "holdout_inputs.npz", x=x_test, times=holdout[TIME_COL].to_numpy(dtype="datetime64[ns]"))
    np.savez_compressed(data_dir / "holdout_labels.npz", y=y_test)
    manifest = {
        "created_utc": utcnow(), "seconds": time.perf_counter() - started,
        "feature_columns": columns, "feature_audit": audit,
        "development_contextual_hash": frame_fingerprint(development, [TIME_COL, *columns]),
        "holdout_contextual_hash": frame_fingerprint(holdout, [TIME_COL, *columns]),
        "n_fit_days": 271, "n_val_days": 30, "n_development_days": 301, "n_test_days": 59,
        "training_file_contains_holdout": False,
        "data_files": {name: sha256(data_dir / name) for name in ["development.npz", "holdout_inputs.npz", "holdout_labels.npz"]},
    }
    write_json(data_dir / "manifest.json", manifest)
    return manifest


def check_protected(run_dir):
    original = read_json(Path(run_dir) / "protected_hashes_before.json")
    changed = [name for name, digest in original.items() if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    result = {"checked_utc": utcnow(), "files": len(original), "changed": changed, "unchanged": not changed}
    write_json(Path(run_dir) / "protected_hashes_after.json", result)
    return result
