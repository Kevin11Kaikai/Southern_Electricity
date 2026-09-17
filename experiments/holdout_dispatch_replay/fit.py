"""Fit or reload one holdout recipe and return 59-day predictions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from experiments.holdout_dispatch_replay import artifacts
from experiments.holdout_dispatch_replay.recipes import catboost_depth, xgb_max_depth
from experiments.holdout_rmse_v1.run import (
    _climatology_day_matrix,
    _frame_to_day_arrays,
)
from experiments.holdout_rmse_v1.sequence_models import (
    CompactDLinear,
    CompactDayLSTM,
    CompactDayTransformer,
    SequenceTrainResult,
    predict_sequence,
    train_sequence_fixed_epochs,
    train_sequence_model,
)
from experiments.holdout_rmse_v2.run import LEAVES_GRID, LR_GRID, TREE_ES_ROUNDS, TREE_ROUND_CAP, _rmse
from src.phase_b.advanced_models import make_ridge_pipeline, quarter_slot
from src.phase_b.contracts import STEPS_PER_DAY
from src.phase_b.data import TARGET_COL, TIME_COL
from src.phase_b.modeling import ModelSpec, deterministic_params, train_final_model


def _sequence_ctor(family: str, n_features: int):
    if family == "lstm":
        return CompactDayLSTM(n_features)
    if family == "transformer":
        return CompactDayTransformer(n_features)
    if family == "dlinear":
        return CompactDLinear(n_features)
    raise ValueError(f"unknown sequence family: {family}")


def _result_from_payload(payload: dict[str, Any]) -> SequenceTrainResult:
    family = payload["family"]
    n_features = int(payload["n_features"])
    model = _sequence_ctor(family, n_features)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    best_val = payload.get("best_val_rmse")
    return SequenceTrainResult(
        model=model,
        best_val_rmse=float("nan") if best_val is None else float(best_val),
        epochs_run=int(payload["epochs_run"]),
        best_epoch=int(payload["best_epoch"]),
        fill_values=np.asarray(payload["fill_values"], dtype=np.float32),
        mean=np.asarray(payload["mean"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
    )


def predict_climatology_table(table: np.ndarray, test: pd.DataFrame) -> np.ndarray:
    dows = test[TIME_COL].dt.dayofweek.to_numpy()
    slots = quarter_slot(test[TIME_COL])
    return np.asarray(table[dows, slots], dtype=float)


def recover_xgboost_grid(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    feature_columns: list[str],
    published_val30: float,
    *,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    import xgboost as xgb

    y_fit = fit[TARGET_COL].astype(float).to_numpy()
    y_val = val[TARGET_COL].astype(float).to_numpy()
    best: dict[str, Any] | None = None
    closest: dict[str, Any] | None = None
    closest_gap = float("inf")
    for num_leaves in LEAVES_GRID:
        for learning_rate in LR_GRID:
            max_depth = xgb_max_depth(num_leaves)
            model = xgb.XGBRegressor(
                n_estimators=TREE_ROUND_CAP,
                learning_rate=learning_rate,
                max_depth=max_depth,
                max_leaves=num_leaves,
                min_child_weight=20,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=seed,
                n_jobs=n_jobs,
                tree_method="hist",
                objective="reg:squarederror",
                early_stopping_rounds=TREE_ES_ROUNDS,
            )
            model.fit(
                fit[feature_columns],
                y_fit,
                eval_set=[(val[feature_columns], y_val)],
                verbose=False,
            )
            val_pred = np.asarray(model.predict(val[feature_columns]), dtype=float)
            val_rmse = _rmse(y_val, val_pred)
            gap = abs(val_rmse - published_val30)
            candidate = {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "val_rmse": val_rmse,
            }
            print(
                f"[holdout_dispatch_replay]   recover xgboost leaves={num_leaves} "
                f"lr={learning_rate} val30={val_rmse:.6f} gap={gap:.6f}",
                flush=True,
            )
            if gap < closest_gap:
                closest_gap = gap
                closest = candidate
            if gap <= 5e-6:
                best = candidate
    chosen = best or closest
    assert chosen is not None
    chosen["recovered"] = best is not None
    chosen["val30_gap"] = closest_gap if best is None else abs(best["val_rmse"] - published_val30)
    return chosen


def recover_catboost_grid(
    fit: pd.DataFrame,
    val: pd.DataFrame,
    feature_columns: list[str],
    published_val30: float,
    *,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor

    y_fit = fit[TARGET_COL].astype(float).to_numpy()
    y_val = val[TARGET_COL].astype(float).to_numpy()
    best: dict[str, Any] | None = None
    closest: dict[str, Any] | None = None
    closest_gap = float("inf")
    for num_leaves in LEAVES_GRID:
        for learning_rate in LR_GRID:
            depth = catboost_depth(num_leaves)
            model = CatBoostRegressor(
                iterations=TREE_ROUND_CAP,
                learning_rate=learning_rate,
                depth=depth,
                random_seed=seed,
                loss_function="RMSE",
                verbose=False,
                allow_writing_files=False,
                thread_count=n_jobs,
                od_type="Iter",
                od_wait=TREE_ES_ROUNDS,
                use_best_model=True,
            )
            model.fit(fit[feature_columns], y_fit, eval_set=(val[feature_columns], y_val))
            val_pred = np.asarray(model.predict(val[feature_columns]), dtype=float)
            val_rmse = _rmse(y_val, val_pred)
            gap = abs(val_rmse - published_val30)
            candidate = {"depth": depth, "learning_rate": learning_rate, "val_rmse": val_rmse}
            print(
                f"[holdout_dispatch_replay]   recover catboost depth={depth} "
                f"lr={learning_rate} val30={val_rmse:.6f} gap={gap:.6f}",
                flush=True,
            )
            if gap < closest_gap:
                closest_gap = gap
                closest = candidate
            if gap <= 5e-6:
                best = candidate
    chosen = best or closest
    assert chosen is not None
    chosen["recovered"] = best is not None
    chosen["val30_gap"] = closest_gap if best is None else abs(best["val_rmse"] - published_val30)
    return chosen


def _fit_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    recipe: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, Any]:
    spec = ModelSpec(
        name=recipe["name"],
        params=deterministic_params(
            seed=seed,
            n_jobs=n_jobs,
            num_leaves=int(recipe.get("num_leaves", 63)),
            learning_rate=float(recipe.get("learning_rate", 0.03)),
        ),
        num_boost_round=int(recipe["num_boost_round"]),
        early_stopping_rounds=0,
        target_mode="level",
    )
    booster = train_final_model(train, columns, spec, num_boost_round=int(recipe["num_boost_round"]))
    yhat = np.asarray(booster.predict(test[columns]), dtype=float)
    return yhat, booster


def _fit_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    recipe: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, Any]:
    import xgboost as xgb

    kwargs: dict[str, Any] = {
        "n_estimators": int(recipe["n_estimators"]),
        "learning_rate": float(recipe.get("learning_rate", 0.03)),
        "max_depth": int(recipe.get("max_depth", 6)),
        "min_child_weight": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": seed,
        "n_jobs": n_jobs,
        "tree_method": "hist",
        "objective": "reg:squarederror",
    }
    if "num_leaves" in recipe:
        kwargs["max_leaves"] = int(recipe["num_leaves"])
    model = xgb.XGBRegressor(**kwargs)
    model.fit(train[columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(model.predict(test[columns]), dtype=float), model


def _fit_catboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    recipe: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> tuple[np.ndarray, Any]:
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=int(recipe["iterations"]),
        learning_rate=float(recipe.get("learning_rate", 0.03)),
        depth=int(recipe.get("depth", 6)),
        random_seed=seed,
        loss_function="RMSE",
        verbose=False,
        allow_writing_files=False,
        thread_count=n_jobs,
    )
    model.fit(train[columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(model.predict(test[columns]), dtype=float), model


def _fit_hgb(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    recipe: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, Any]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(recipe.get("learning_rate", 0.06)),
        max_iter=int(recipe["max_iter"]),
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=0.1,
        early_stopping=False,
        random_state=seed,
    )
    model.fit(train[columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(model.predict(test[columns]), dtype=float), model


def _fit_ridge(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    recipe: dict[str, Any],
) -> tuple[np.ndarray, Any]:
    pipeline = make_ridge_pipeline(alpha=float(recipe["alpha"]))
    pipeline.fit(train[columns], train[TARGET_COL].astype(float).to_numpy())
    return np.asarray(pipeline.predict(test[columns]), dtype=float), pipeline


def _fit_sequence_inner_val(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    recipe: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, SequenceTrainResult, int]:
    family = recipe["family"]
    x_all, y_all, train_dates = _frame_to_day_arrays(train, columns)
    x_test, _, test_dates = _frame_to_day_arrays(test, columns)
    inner_val_days = int(recipe["inner_val_days"])
    split = len(train_dates) - inner_val_days
    n_features = int(x_all.shape[-1])
    result = train_sequence_model(
        _sequence_ctor(family, n_features),
        x_all[:split],
        y_all[:split],
        x_all[split:],
        y_all[split:],
        seed=seed,
    )
    yhat_days = predict_sequence(result, x_test)
    if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
        raise ValueError(f"{family} prediction shape {yhat_days.shape} does not match holdout days")
    return yhat_days.reshape(-1).astype(float), result, n_features


def _fit_sequence_fixed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    recipe: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, SequenceTrainResult, int]:
    family = recipe["family"]
    x_all, y_all, _dates = _frame_to_day_arrays(train, columns)
    x_test, _, test_dates = _frame_to_day_arrays(test, columns)
    n_features = int(x_all.shape[-1])
    result = train_sequence_fixed_epochs(
        _sequence_ctor(family, n_features),
        x_all,
        y_all,
        num_epochs=int(recipe["num_epochs"]),
        seed=seed,
    )
    yhat_days = predict_sequence(result, x_test)
    if yhat_days.shape != (len(test_dates), STEPS_PER_DAY):
        raise ValueError(f"{family} prediction shape {yhat_days.shape} does not match holdout days")
    return yhat_days.reshape(-1).astype(float), result, n_features


def _prevday_arrays(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _, y_train, train_dates = _frame_to_day_arrays(train, [])
    _, y_test, test_dates = _frame_to_day_arrays(test, [])
    climatology = _climatology_day_matrix(train)
    train_lookback = np.concatenate([climatology[train_dates[0].dayofweek][None, :], y_train[:-1]], axis=0)[:, :, None]
    test_lookback = np.concatenate([y_train[-1][None, :], y_test[:-1]], axis=0)[:, :, None]
    return train_lookback, y_train, test_lookback, climatology, y_train[-1]


def _fit_dlinear_prevday_inner(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recipe: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, SequenceTrainResult, np.ndarray, np.ndarray]:
    train_lookback, y_train, test_lookback, climatology, last_train = _prevday_arrays(train, test)
    inner_val_days = int(recipe["inner_val_days"])
    split = len(y_train) - inner_val_days
    result = train_sequence_model(
        CompactDLinear(n_features=1),
        train_lookback[:split],
        y_train[:split],
        train_lookback[split:],
        y_train[split:],
        seed=seed,
    )
    yhat_days = predict_sequence(result, test_lookback)
    return yhat_days.reshape(-1).astype(float), result, climatology, last_train


def _fit_dlinear_prevday_fixed(
    train: pd.DataFrame,
    test: pd.DataFrame,
    recipe: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, SequenceTrainResult, np.ndarray, np.ndarray]:
    train_lookback, y_train, test_lookback, climatology, last_train = _prevday_arrays(train, test)
    result = train_sequence_fixed_epochs(
        CompactDLinear(n_features=1),
        train_lookback,
        y_train,
        num_epochs=int(recipe["num_epochs"]),
        seed=seed,
    )
    yhat_days = predict_sequence(result, test_lookback)
    return yhat_days.reshape(-1).astype(float), result, climatology, last_train


def load_yhat_from_checkpoint(
    directory: Any,
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    cached: dict[str, np.ndarray],
) -> np.ndarray | None:
    from pathlib import Path

    directory = Path(directory)
    if not artifacts.has_checkpoint(directory):
        return None
    meta = artifacts.load_meta(directory)
    family = meta["family"]
    if family == "climatology":
        return predict_climatology_table(artifacts.load_climatology(directory), test)
    if family == "lgb":
        booster, columns = artifacts.load_lightgbm(directory)
        return np.asarray(booster.predict(test[columns]), dtype=float)
    if family == "xgboost":
        model, columns = artifacts.load_xgboost(directory)
        return np.asarray(model.predict(test[columns]), dtype=float)
    if family == "catboost":
        model, columns = artifacts.load_catboost(directory)
        return np.asarray(model.predict(test[columns]), dtype=float)
    if family in {"ridge", "hgb"}:
        model, columns = artifacts.load_sklearn(directory)
        return np.asarray(model.predict(test[columns]), dtype=float)
    if family in {"lstm", "transformer", "dlinear"}:
        payload = artifacts.load_sequence(directory)
        result = _result_from_payload(payload)
        columns = list(payload["meta"]["feature_columns"])
        x_test, _, _dates = _frame_to_day_arrays(test, columns)
        return predict_sequence(result, x_test).reshape(-1).astype(float)
    if family == "dlinear_prevday":
        payload = artifacts.load_sequence(directory)
        result = _result_from_payload({**payload, "family": "dlinear", "n_features": 1})
        last_train = np.asarray(payload["meta"]["last_train_curve"], dtype=np.float32)
        _, y_test, _dates = _frame_to_day_arrays(test, [])
        test_lookback = np.concatenate([last_train[None, :], y_test[:-1]], axis=0)[:, :, None]
        return predict_sequence(result, test_lookback).reshape(-1).astype(float)
    if family == "blend":
        w_tr = float(meta["w_transformer"])
        w_lgb = float(meta["w_lgb"])
        if "lgb_contextual" in cached and "transformer_contextual" in cached:
            return w_tr * cached["transformer_contextual"] + w_lgb * cached["lgb_contextual"]
        return None
    raise ValueError(f"unknown checkpoint family: {family}")


def fit_and_save(
    recipe: dict[str, Any],
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str] | None,
    seed: int,
    n_jobs: int,
    directory: Any,
    fit_frame: pd.DataFrame | None = None,
    val_frame: pd.DataFrame | None = None,
    cached: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from pathlib import Path

    name = recipe["name"]
    family = recipe["family"]
    directory = Path(directory)
    extra: dict[str, Any] = {}
    cached = cached or {}

    if family == "climatology":
        table = _climatology_day_matrix(train)
        yhat = predict_climatology_table(table, test)
        artifacts.save_climatology(directory, table, {"name": name, "seed": seed})
        return yhat, extra

    if family == "blend":
        if "lgb_contextual" not in cached or "transformer_contextual" not in cached:
            raise RuntimeError("blend requires cached lgb_contextual and transformer_contextual predictions")
        yhat = recipe["w_transformer"] * cached["transformer_contextual"] + recipe["w_lgb"] * cached["lgb_contextual"]
        artifacts.save_blend(
            directory,
            {
                "name": name,
                "w_transformer": recipe["w_transformer"],
                "w_lgb": recipe["w_lgb"],
                "components": ["transformer_contextual", "lgb_contextual"],
            },
        )
        return yhat, extra

    if family == "lgb":
        yhat, booster = _fit_lightgbm(train, test, list(columns), recipe, seed=seed, n_jobs=n_jobs)
        artifacts.save_lightgbm(directory, booster, list(columns), {"name": name, "seed": seed, **{k: recipe[k] for k in recipe if k != "name"}})
        return yhat, extra

    if family == "xgboost":
        local = dict(recipe)
        if local.get("recover_grid"):
            if fit_frame is None or val_frame is None:
                raise RuntimeError("xgboost grid recovery needs the 271/30 frames")
            recovered = recover_xgboost_grid(
                fit_frame,
                val_frame,
                list(columns),
                float(local["published_val30_rmse"]),
                seed=seed,
                n_jobs=n_jobs,
            )
            local.update({k: recovered[k] for k in ("num_leaves", "learning_rate", "max_depth")})
            extra["grid_recovery"] = recovered
        yhat, model = _fit_xgboost(train, test, list(columns), local, seed=seed, n_jobs=n_jobs)
        artifacts.save_xgboost(directory, model, list(columns), {"name": name, "seed": seed, **{k: local[k] for k in local if k != "name"}})
        return yhat, extra

    if family == "catboost":
        local = dict(recipe)
        if local.get("recover_grid"):
            if fit_frame is None or val_frame is None:
                raise RuntimeError("catboost grid recovery needs the 271/30 frames")
            recovered = recover_catboost_grid(
                fit_frame,
                val_frame,
                list(columns),
                float(local["published_val30_rmse"]),
                seed=seed,
                n_jobs=n_jobs,
            )
            local.update({k: recovered[k] for k in ("depth", "learning_rate")})
            extra["grid_recovery"] = recovered
        yhat, model = _fit_catboost(train, test, list(columns), local, seed=seed, n_jobs=n_jobs)
        artifacts.save_catboost(directory, model, list(columns), {"name": name, "seed": seed, **{k: local[k] for k in local if k != "name"}})
        return yhat, extra

    if family == "hgb":
        yhat, model = _fit_hgb(train, test, list(columns), recipe, seed=seed)
        artifacts.save_sklearn(directory, model, list(columns), {"name": name, "family": "hgb", "seed": seed, **recipe})
        return yhat, extra

    if family == "ridge":
        yhat, model = _fit_ridge(train, test, list(columns), recipe)
        artifacts.save_sklearn(directory, model, list(columns), {"name": name, "family": "ridge", "seed": seed, **recipe})
        return yhat, extra

    if family in {"lstm", "transformer", "dlinear"}:
        if "inner_val_days" in recipe:
            yhat, result, n_features = _fit_sequence_inner_val(train, test, list(columns), recipe, seed=seed)
        else:
            yhat, result, n_features = _fit_sequence_fixed(train, test, list(columns), recipe, seed=seed)
        artifacts.save_sequence(
            directory,
            result,
            family=family,
            n_features=n_features,
            meta={"name": name, "seed": seed, "feature_columns": list(columns), **recipe},
        )
        extra["best_epoch"] = result.best_epoch
        extra["epochs_run"] = result.epochs_run
        return yhat, extra

    if family == "dlinear_prevday":
        if "inner_val_days" in recipe:
            yhat, result, climatology, last_train = _fit_dlinear_prevday_inner(train, test, recipe, seed=seed)
        else:
            yhat, result, climatology, last_train = _fit_dlinear_prevday_fixed(train, test, recipe, seed=seed)
        artifacts.save_sequence(
            directory,
            result,
            family="dlinear",
            n_features=1,
            meta={
                "name": name,
                "family": "dlinear_prevday",
                "seed": seed,
                "last_train_curve": np.asarray(last_train, dtype=float).tolist(),
                **recipe,
            },
        )
        np.save(directory / "climatology.npy", climatology)
        extra["best_epoch"] = result.best_epoch
        extra["epochs_run"] = result.epochs_run
        return yhat, extra

    raise ValueError(f"unsupported family: {family}")
