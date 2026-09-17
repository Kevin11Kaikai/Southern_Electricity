# Holdout RMSE bakeoff v1

Protocol: train every model on **all 301 development complete days**, then rank by **RMSE on the last 59 holdout days** (2025-11-03 to 2025-12-31). That 59-day RMSE is both the leaderboard and the reported score.

This experiment does **not** use nested 3×14 validation and does **not** use expanding 5-fold for model choice. `make_day_split_plan` is used only to recover `development_dates` and `holdout_dates`; fold objects are ignored.

## Split

- Universe: 360 complete labeled 15-minute days.
- Train: 301 days.
- Test: 59 days.
- Target: absolute price `A`.
- Metric: `RMSE = sqrt(mean((yhat - y)^2))` on all labeled holdout points (~5664). MAE is a second column only.
- No leakage: holdout rows never enter training.

## Early stopping

Trees use a **fixed** 1500 boosting rounds and seed 42. There is no fifth fold and no early stopping for ranking.

LSTM, Transformer, and DLinear may use the **last 14 days of the 301** as a training-time validation loader for early stop. That slice is never used to rank models. Ranking is always 59-day RMSE.

## Models

| Name | Notes |
| --- | --- |
| `climatology_weekday_slot` | Weekday × 15-minute slot mean of `A` |
| `lgb_baseline` / `contextual` / `temporal` / `full` | LightGBM, level target, 1500 rounds |
| `ridge_contextual_a1` / `ridge_temporal_a1` | Ridge α=1 with median impute + scale |
| `xgboost_contextual` / `xgboost_full` | XGBoost, 1500 rounds; skipped if the package is missing |
| `catboost_contextual` / `catboost_full` | CatBoost, 1500 rounds; skipped if the package is missing |
| `hgb_full` | sklearn HistGradientBoosting |
| `lstm_contextual` | Compact 96-step LSTM, contextual channels |
| `transformer_contextual` | Compact 96-step Transformer encoder, new module (not nested/action-head code) |
| `dlinear_contextual` | Compact DLinear on the same 96×contextual channels as LSTM/Transformer |
| `dlinear_prevday` | Compact DLinear: previous complete day's actual `A` → today (day-ahead lookback) |

DQN/PPO and ensemble-weight search are out of scope. This run does not pick charge/discharge windows.

## Outputs

- `reports/holdout_rmse_v1/leaderboard.csv` — model, train_days, test_days, RMSE, MAE, n_points, status
- `reports/holdout_rmse_v1/manifest.json` — seed, date ranges, `nested_cv: false`, `expanding_5fold: false`

## Run

```text
python experiments/holdout_rmse_v1/run.py
```

If a model fails or a package is missing, the leaderboard keeps a row with `status` and continues.
