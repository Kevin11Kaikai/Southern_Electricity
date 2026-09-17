"""Save and load holdout checkpoints and 59-day prediction tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_ROOT = ROOT / "outputs" / "holdout_checkpoints"


def checkpoint_dir(version: str, name: str) -> Path:
    return CHECKPOINT_ROOT / version / name


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_predictions_frame(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.sort_values("times").reset_index(drop=True)
    try:
        ordered.to_parquet(path, index=False)
        return path
    except (ImportError, ValueError):
        csv_path = path.with_suffix(".csv")
        ordered.to_csv(csv_path, index=False)
        return csv_path


def read_predictions(path: Path) -> pd.DataFrame:
    parquet = path if path.suffix == ".parquet" else path
    if parquet.exists():
        try:
            frame = pd.read_parquet(parquet)
            frame["times"] = pd.to_datetime(frame["times"])
            return frame.sort_values("times").reset_index(drop=True)
        except (ImportError, ValueError):
            pass
    csv_path = path.with_suffix(".csv") if path.suffix != ".csv" else path
    if not csv_path.exists() and path.suffix == ".parquet":
        csv_path = path.with_name("predictions.csv")
    frame = pd.read_csv(csv_path)
    frame["times"] = pd.to_datetime(frame["times"])
    return frame.sort_values("times").reset_index(drop=True)


def upsert_prediction_column(
    path: Path,
    times: pd.Series,
    y_true: np.ndarray,
    name: str,
    yhat: np.ndarray,
) -> Path:
    csv_fallback = path.with_suffix(".csv")
    clock = pd.to_datetime(pd.Series(times)).reset_index(drop=True)
    if path.exists() or csv_fallback.exists():
        existing_path = path if path.exists() else csv_fallback
        frame = read_predictions(existing_path)
        existing_clock = pd.to_datetime(frame["times"]).reset_index(drop=True)
        if not existing_clock.equals(clock):
            raise ValueError(f"prediction clock mismatch while writing {name}")
    else:
        frame = pd.DataFrame({"times": clock, "A": np.asarray(y_true, dtype=float)})
    frame[name] = np.asarray(yhat, dtype=float)
    return _write_predictions_frame(path, frame)


def save_meta(directory: Path, meta: dict[str, Any]) -> None:
    write_json(directory / "meta.json", meta)


def load_meta(directory: Path) -> dict[str, Any]:
    return read_json(directory / "meta.json")


def save_climatology(directory: Path, table: np.ndarray, meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "weekday_slot.npy", np.asarray(table, dtype=np.float32))
    save_meta(directory, {**meta, "family": "climatology"})


def load_climatology(directory: Path) -> np.ndarray:
    return np.load(directory / "weekday_slot.npy")


def save_lightgbm(directory: Path, booster: Any, feature_columns: list[str], meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(directory / "model.txt"))
    save_meta(directory, {**meta, "family": "lgb", "feature_columns": list(feature_columns)})


def load_lightgbm(directory: Path) -> tuple[Any, list[str]]:
    import lightgbm as lgb

    meta = load_meta(directory)
    booster = lgb.Booster(model_file=str(directory / "model.txt"))
    return booster, list(meta["feature_columns"])


def save_xgboost(directory: Path, model: Any, feature_columns: list[str], meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(directory / "model.json"))
    save_meta(directory, {**meta, "family": "xgboost", "feature_columns": list(feature_columns)})


def load_xgboost(directory: Path) -> tuple[Any, list[str]]:
    import xgboost as xgb

    meta = load_meta(directory)
    model = xgb.XGBRegressor()
    model.load_model(str(directory / "model.json"))
    return model, list(meta["feature_columns"])


def save_catboost(directory: Path, model: Any, feature_columns: list[str], meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save_model(str(directory / "model.cbm"))
    save_meta(directory, {**meta, "family": "catboost", "feature_columns": list(feature_columns)})


def load_catboost(directory: Path) -> tuple[Any, list[str]]:
    from catboost import CatBoostRegressor

    meta = load_meta(directory)
    model = CatBoostRegressor()
    model.load_model(str(directory / "model.cbm"))
    return model, list(meta["feature_columns"])


def save_sklearn(directory: Path, model: Any, feature_columns: list[str], meta: dict[str, Any]) -> None:
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, directory / "model.joblib")
    save_meta(directory, {**meta, "feature_columns": list(feature_columns)})


def load_sklearn(directory: Path) -> tuple[Any, list[str]]:
    import joblib

    meta = load_meta(directory)
    return joblib.load(directory / "model.joblib"), list(meta["feature_columns"])


def save_sequence(
    directory: Path,
    result: Any,
    *,
    family: str,
    n_features: int,
    meta: dict[str, Any],
) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": family,
        "n_features": int(n_features),
        "state_dict": {key: value.detach().cpu() for key, value in result.model.state_dict().items()},
        "fill_values": np.asarray(result.fill_values),
        "mean": np.asarray(result.mean),
        "scale": np.asarray(result.scale),
        "best_epoch": int(result.best_epoch),
        "epochs_run": int(result.epochs_run),
        "best_val_rmse": float(result.best_val_rmse) if np.isfinite(result.best_val_rmse) else None,
    }
    torch.save(payload, directory / "model.pt")
    merged = {**meta, "n_features": int(n_features), "architecture_family": family}
    merged.setdefault("family", family)
    save_meta(directory, merged)


def load_sequence(directory: Path) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(directory / "model.pt", map_location="cpu")
    payload["meta"] = load_meta(directory)
    return payload


def save_blend(directory: Path, meta: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_meta(directory, {**meta, "family": "blend"})


def has_checkpoint(directory: Path) -> bool:
    return (directory / "meta.json").exists()
