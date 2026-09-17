# Data: what is here, what is not, and how to regenerate the rest

*中文摘要见文末。*

## What this repository does not contain

No source dataset, no 96-point forecast curves, no per-day settlement tables, no
model weights. Specifically absent:

- the raw boundary, price and test feature CSVs
- the weather NetCDF archive
- `reports/**` — the frozen experiment outputs
- `outputs/holdout_checkpoints/**` — trained model binaries
- `_feature_cache.npz` — cached standardized feature arrays

`reports/**` is excluded because `predictions.parquet` carries a column named `A`:
the true evaluation-period prices, 5,664 values. Every realized score in the
report is `Aᵀu`, so recomputing from the curves requires those labels. They are
derivatives of the competition dataset, so they are not redistributed here.

## What this repository does contain

`report/shared/data/` holds sixteen **aggregated** records — per-day scores,
per-candidate summary metrics, correlations with resampling intervals, the
ensemble decomposition, the provenance table. These are the same quantities the
report prints in its figures and tables. None of them contains a price curve: the
finest granularity is one score per model per day.

That is enough to redraw every figure, regenerate every table and recompile both
PDFs. It is not enough to re-solve the scheduling problem from scratch.

## Obtaining the source data

The dataset comes from the AI4S competition:

> <https://competition.ai4s.com.cn/race/9/description>

Obtain it under the competition's own terms. This repository asserts no right to
redistribute it and makes no claim about what those terms permit.

Two caveats the report itself states and you should carry forward:

1. The target is an **anonymized, rescaled** series. Values are not in physical
   units and not in currency. The report does not identify a market and neither
   should any downstream use of these results.
2. Whole-day summaries of the supplied *forecast* boundary channels are only a
   legitimate input if the complete boundary curve was published before the
   decision. That publication contract is not established by the files.

## Where to put it

`configs/phase_b.toml` resolves every path relative to the repository root:

```
to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv    # 15 boundary channels, actual + forecast
to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv       # times, A  -- the target
to_sais_new/to_sais_new/test/test_in_feature_ori.csv               # forecast-only mirror, unlabeled period
to_sais_new/to_sais_new/all_nc/                                    # 424 NetCDF files, ~12.4 GiB
```

**The weather archive is optional for the headline result.** The 67-channel
`contextual` feature set is assembled in `src/phase_b/features.py` *before* the
weather merge; only the `full` set uses weather. If you skip `all_nc/`, the
headline listwise-Transformer recipe still trains. `netCDF4` and `xarray` in
`requirements-full.txt` are only needed for a cold weather build; once
`outputs/phase_b/cache/weather_hourly.csv.gz` exists it is reused.

## Regenerating the experiment outputs

```bash
pip install -r requirements-full.txt
```

Run in this order. The dependencies are real: v2 imports v1's module, the
bake-off imports the dispatch-loss module, the seed bag imports three earlier
experiments, and the replay step is what writes the prediction and settlement
files into the v1 and v2 report directories.

```bash
python experiments/holdout_rmse_v1/run.py             # point-loss panel 1 (16 candidates)
python experiments/holdout_rmse_v2/run.py             # point-loss panel 2 (19 candidates)
python experiments/holdout_dispatch_replay/run.py     # settles v1+v2, writes predictions & daily tables
python experiments/holdout_dispatch_loss_v1/run.py    # block MSE + hard window CE
python experiments/holdout_dfl_bakeoff_v1/run.py      # SPO+, pairwise, DBB, blend
python experiments/holdout_dfl_ltr_v1/run.py          # listwise grid; then diagnostics/ensemble/rerank/...
python experiments/holdout_seed_bag_v1/run.py         # the frozen 5-seed headline protocol
python experiments/holdout_dispatch_replay/failure_analysis.py
```

`experiments/holdout_dispatch_loss_codex_v1/` produces the matched-initialization
control table quoted in the report's Table 8; it is an independently scripted
run with its own protocol freeze.

### One precondition you cannot satisfy from this repository

`experiments/holdout_dispatch_replay/run.py` asserts that
`reports/phase_b/final_metrics.json` exists — a guard from the earlier Phase B
work, protecting that file from being overwritten. That file is **not shipped
here.** Either produce it from the Phase B pipeline (outside this repository's
scope) or remove the `assert_phase_b_untouched()` call before running the replay
step. The replay itself does not read the file's contents.

### Determinism

Seeds are fixed (42–46 for the headline bag, `torch.manual_seed` before model
construction). Tree and linear candidates reproduce exactly. Torch results are
reproducible on the same platform and version but are not guaranteed to be
bit-identical across CPU architectures or PyTorch releases, so the fifteen
sequence-model trainings may land a few score units away from the recorded
values. The report treats initialization convention as a first-order effect: a
matched-initialization control moved one LSTM by 137.79, comparable to an entire
stage-level gain.

## Verifying what you regenerated

```bash
python report/shared/derive_records.py --repo-root . --out /tmp/fresh_records
python tools/independent_audit.py
```

`derive_records.py` validates the 62 saved curves — shared timestamps, identical
target vector, legal saved actions, saved predicted score equal to the solver
optimum — then recomputes every record. Diff `/tmp/fresh_records` against
`report/shared/data/` to see whether your pipeline reproduced the frozen numbers.

`tools/independent_audit.py` is the stronger check: it never imports
`derive_records.py`, rebuilding block sums with `np.convolve` and re-solving each
day through the project solver.

---

## 中文摘要

**本仓库不含原始数据。** 没有原始价格与边界 CSV、没有天气 NetCDF、没有 `reports/**`
下的 96 点预测曲线与逐日结算表、没有模型权重。原因是 `predictions.parquet` 里包含
真实标签列 `A`（评价期 5,664 个值）——报告里每个实现分都是 `Aᵀu`，要从曲线重算就必须
附带这些标签，而它们属于竞赛数据的派生物。

**仓库包含的是聚合派生记录**（`report/shared/data/`，16 个文件），最细粒度为"每个模型
每天一个分数"，与报告正文图表所印的内容同级。凭它可以重画全部图、重建全部表、重新
编译两版 PDF；不足以从零重解调度问题。

**原始数据获取：** <https://competition.ai4s.com.cn/race/9/description>，请按竞赛自身
条款取得。本仓库不主张任何再分发权利。数据为匿名重标定序列，无物理单位、非货币；
报告不认定具体市场身份。

**放置路径与执行顺序见上文英文部分。** 两点要特别注意：天气档案（12.4 GiB）对主模型
**不是必需**的——67 维 `contextual` 特征在天气合并之前构建；以及
`holdout_dispatch_replay/run.py` 硬依赖本仓库不提供的
`reports/phase_b/final_metrics.json`，需自行生成或移除该断言。
