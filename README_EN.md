# Electricity Price Forecasting and Storage Dispatch Optimization

Yukai Song · Ph.D. candidate in Electrical and Computer Engineering, University of Pittsburgh · Independent research, May–September 2026

[中文首页](README.md) / [Published report snapshot (18 pages)](report/pdf/dispatch_value_report_en.pdf) / [Reproduction](#report-editions-and-reproduction)

## Aligning forecasts with the decisions they support

This project studies a practical mismatch: a forecast with lower pointwise error need not induce a storage schedule with higher realized value. It connects the geometry of a constrained action set to model training, forecast aggregation, exact dispatch, and replay against realized prices.

In one concrete historical comparison on the same 59 days, contextual Transformer v1 had lower RMSE than v2 (0.519558 vs 0.682788), but lower realized mean dispatch score (5,637.17 vs 6,214.36). The fitting recipes differed; this is a counterexample to candidate ranking by RMSE, not a controlled causal experiment.

## SCALE: learn window values, average forecasts, select a feasible schedule

[![SCALE method overview](docs/assets/scale_method_overview_20260917.png)](docs/assets/scale_method_overview_20260917.png)

SCALE stands for Storage-dispatch Curve Averaging and Listwise Estimation. It names the implemented combination of existing ideas, not a claim to new general algorithmic theory.

1. Map 96 price predictions to 89 overlapping eight-slot block sums and 3,321 charge/discharge spreads. Train with day-centered block regression and listwise soft-target supervision.
2. Average price curves from five fixed-seed Transformers (42–46), then apply the same exact scheduler. Averaging prices preserves a legal downstream decision; averaging actions need not.
3. Enumerate 3,321 active schedules plus idle. The action family implies capacity feasibility (0–8,000 in task accounting units), charge-before-discharge, and zero terminal storage. Evaluation prices enter settlement only after the action is locked.

The report derives price-transform invariance and a regret bound through feasible action differences. It also states the Bayes boundary: under ideal population MSE assumptions, the conditional mean remains optimal for risk-neutral linear value over a fixed feasible set.

## Recorded results and their scope

The 59-day evaluation period in November–December 2025 had already been inspected in early exploration. The final ensemble recipe and combination rule were frozen before ensemble scoring; the period is not an untouched confirmation set.

| Method | Mean raw dispatch score | Aggregate fraction of same-constraint oracle |
| :-- | --: | --: |
| Project LightGBM baseline | 5,786.10 | 64.28% |
| Project MSE Transformer reference | 6,214.36 | 69.04% |
| SCALE: five-seed listwise Transformer curve mean | 6,693.88 | 74.37% |
| Perfect-information oracle (upper bound only) | 9,000.89 | 100.00% |

The observed mean increase over the MSE Transformer reference is 479.51 (7.72%). Aggregate capture is total realized score divided by total oracle score, not the mean of daily ratios.

> 86.0% of the net increase occurs on one day. The paired seven-day-block 95% conditional resampling interval is [−227.87, 1,626.28]. The worst day is −5,922.66 for SCALE versus −2,340.48 for the reference. These results do not establish future-period superiority, improved tail risk, or an isolated causal benefit of the loss. Scores are not currency or net trading profit.

## Inspect the implementation

| Work | Entry point |
| :-- | :-- |
| 67-channel context and temporal splits | [Features](src/phase_b/features.py), [splits](src/phase_b/splits.py) |
| Block regression and listwise loss | [Loss implementation](experiments/holdout_dfl_ltr_v1/loss.py) |
| Fixed five-seed retraining and aggregation | [Experiment](experiments/holdout_seed_bag_v1/run.py), [combination](experiments/holdout_seed_bag_v1/combine.py) |
| Exact dispatch and action validation | [Solver](src/phase_b/dispatch.py), [contract](src/phase_b/contracts.py) |
| Independent replay of recorded predictions | [Audit script](tools/independent_audit.py), [replay records](audit_records/solver_replay.json) |

Recorded independent replay covers 62 prediction curves × 59 days (3,658 daily decisions), with no action mismatches and settlement agreement within 1.5 × 10⁻¹¹. It checks implementation consistency, not future performance.

## Report editions and reproduction

The method overview and constraint interpretation on this page reflect the 17 September revision. The currently published PDFs remain the earlier reproducibility snapshot: [English, 18 pages](report/pdf/dispatch_value_report_en.pdf) and [Chinese, 16 pages](report/pdf/dispatch_value_report_zh.pdf). The revised full PDFs and source packages have not yet been published here. The earlier audit’s broad statements about absent capacity constraints and unknown market origin are superseded by the more precise interpretation on this page. The original build tools reproduce the earlier snapshot.
