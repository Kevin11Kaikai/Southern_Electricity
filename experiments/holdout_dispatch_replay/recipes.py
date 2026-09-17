"""Frozen v1/v2 fit recipes. v2 does not re-run the 30-day selection grid."""

from __future__ import annotations

from typing import Any

import pandas as pd

from experiments.holdout_rmse_v1.run import RIDGE_ALPHA, TREE_ROUNDS
from experiments.holdout_rmse_v2.run import TREE_ROUND_CAP

V1_SEQUENCE_INNER_VAL_DAYS = 14
V1_HGB_MAX_ITER = 300
RMSE_DRIFT_TOL = 1e-4


def parse_model_name(name: str) -> tuple[str, str | None]:
    if name == "climatology_weekday_slot":
        return "climatology", None
    if name == "dlinear_prevday":
        return "dlinear_prevday", None
    if name == "blend_lgb_transformer":
        return "blend", None
    if name.startswith("ridge_"):
        feature_set = name[len("ridge_") :].rsplit("_", 1)[0]
        return "ridge", feature_set
    if name.startswith("hgb_"):
        return "hgb", name.split("_", 1)[1]
    for prefix in ("lgb_", "xgboost_", "catboost_", "lstm_", "transformer_", "dlinear_"):
        if name.startswith(prefix):
            return prefix[:-1], name[len(prefix) :]
    raise ValueError(f"unrecognized model name: {name}")


def _rounds_from_board(row: pd.Series, default: int | None = None) -> int | None:
    value = row.get("best_rounds_or_epochs")
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return default
    return int(value)


def xgb_max_depth(num_leaves: int) -> int:
    return 5 if num_leaves <= 31 else 6 if num_leaves <= 63 else 7


def catboost_depth(num_leaves: int) -> int:
    return 5 if num_leaves <= 31 else 6 if num_leaves <= 63 else 7


def v1_recipe(name: str) -> dict[str, Any]:
    family, feature_set = parse_model_name(name)
    recipe: dict[str, Any] = {"name": name, "family": family, "feature_set": feature_set, "version": "v1"}
    if family == "lgb":
        recipe["num_boost_round"] = TREE_ROUNDS
    elif family == "xgboost":
        recipe["n_estimators"] = TREE_ROUNDS
        recipe["learning_rate"] = 0.03
        recipe["max_depth"] = 6
        recipe["min_child_weight"] = 20
    elif family == "catboost":
        recipe["iterations"] = TREE_ROUNDS
        recipe["learning_rate"] = 0.03
        recipe["depth"] = 6
    elif family == "hgb":
        recipe["max_iter"] = V1_HGB_MAX_ITER
        recipe["learning_rate"] = 0.06
        recipe["early_stopping"] = False
    elif family == "ridge":
        recipe["alpha"] = RIDGE_ALPHA
    elif family in {"lstm", "transformer", "dlinear"}:
        recipe["inner_val_days"] = V1_SEQUENCE_INNER_VAL_DAYS
        recipe["feature_set"] = "contextual"
    elif family == "dlinear_prevday":
        recipe["inner_val_days"] = V1_SEQUENCE_INNER_VAL_DAYS
    return recipe


def v2_recipe(
    name: str,
    *,
    leaderboard_row: pd.Series,
    selection: dict[str, Any],
) -> dict[str, Any]:
    family, feature_set = parse_model_name(name)
    selected = selection.get(name) or {}
    recipe: dict[str, Any] = {
        "name": name,
        "family": family,
        "feature_set": feature_set,
        "version": "v2",
        "batch": leaderboard_row.get("batch"),
    }
    board_rounds = _rounds_from_board(leaderboard_row)

    if family == "lgb":
        if name == "lgb_baseline":
            recipe["num_boost_round"] = TREE_ROUNDS
            recipe["num_leaves"] = 63
            recipe["learning_rate"] = 0.03
        else:
            recipe["num_leaves"] = int(selected.get("num_leaves", 63))
            recipe["learning_rate"] = float(selected.get("learning_rate", 0.03))
            recipe["num_boost_round"] = int(selected.get("final_rounds", board_rounds or TREE_ROUND_CAP))
    elif family == "xgboost":
        recipe["n_estimators"] = int(board_rounds or TREE_ROUND_CAP)
        if name == "xgboost_contextual":
            recipe["recover_grid"] = True
            recipe["published_val30_rmse"] = float(leaderboard_row["val30_RMSE"])
        else:
            recipe["num_leaves"] = 63
            recipe["learning_rate"] = 0.03
            recipe["max_depth"] = 6
    elif family == "catboost":
        recipe["iterations"] = int(board_rounds or TREE_ROUND_CAP)
        if name == "catboost_contextual":
            recipe["recover_grid"] = True
            recipe["published_val30_rmse"] = float(leaderboard_row["val30_RMSE"])
        else:
            recipe["learning_rate"] = 0.03
            recipe["depth"] = 6
    elif family == "hgb":
        recipe["max_iter"] = int(board_rounds or 300)
        recipe["learning_rate"] = 0.06
        recipe["early_stopping"] = False
    elif family == "ridge":
        recipe["alpha"] = float(selected["alpha"])
    elif family in {"lstm", "transformer", "dlinear"}:
        recipe["feature_set"] = "contextual"
        recipe["num_epochs"] = int(selected.get("final_epochs", board_rounds or 1))
    elif family == "dlinear_prevday":
        recipe["num_epochs"] = int(board_rounds or 80)
    elif family == "blend":
        recipe["w_transformer"] = float(selected["w_transformer"])
        recipe["w_lgb"] = float(selected["w_lgb"])
    return recipe
