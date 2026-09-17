# Holdout RMSE bakeoff v2 (271 / 30 / 59)

v1 trained on all 301 development days with a fixed 1500 rounds (trees) or a 14-day inner stop (sequences), then ranked on the last 59 days. That 59-day table is frozen in `reports/holdout_rmse_v1/leaderboard.csv`.

v2 keeps the same 59-day evaluation, but **selects only on the last 30 development days** (~October), then retrains on all 301 with the frozen config.

## Protocol

1. Fit on the first 271 complete development days.
2. Choose rounds / epochs / a small tree grid by **30-day RMSE**.
3. Retrain on all 301 days with `round(best * 301/271)` (caps: 1500 trees, 80 sequence epochs).
4. Score the last 59 holdout days **once**. No 59-day early stopping, no 59-day model picking.

Batch A runs the competitive contextual trees and Transformer first (plus climatology / `lgb_baseline` anchors and one 30-day LGB–Transformer blend). Batch B then reruns LSTM, DLinear, Ridge, temporal/full trees, and HGB under the same split so those families are not dropped.

## Outputs

- `reports/holdout_rmse_v2/leaderboard.csv`
- `reports/holdout_rmse_v2/comparison_0914.csv` (v1 vs v2 RMSE; negative delta is better)
- `reports/holdout_rmse_v2/manifest.json`

## Run

```text
python experiments/holdout_rmse_v2/run.py
```
