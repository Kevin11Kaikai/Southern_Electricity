# holdout_dfl_ltr_v1 · 计划（训练前，待批准）

**目标（唯一判定）：** 同一协议下 59 日 `realized_profit_mean` **> 6301.33**（capture > 0.700079）。
**incumbent：** `nb12 / lstm_dispatch` = 6301.33 / 0.700079。
**不改：** `reports/phase_b/final_metrics.json`、v1/v2 leaderboard 的 RMSE/MAE、`optimize_day` 规则、`holdout_dispatch_loss_v1/`。

---

## 0. 先量后设计：四个已算出的数字决定了路线

计划不是从论文排名推出来的，是从下面四个在本仓库数据上实测的量推出来的。复算脚本随实现一起提交（`diagnostics.py`）。

### D1 · one-hot CE 的目标在这个问题上不可学，而且不该学

真分数曲面顶部极平（59 日均，官方分数）：

| 真实第 K 好的窗 | 1 | 2 | 5 | 10 | 50 |
|---|---:|---:|---:|---:|---:|
| 价值 | 9001 | 8990 | 8937 | 8888 | 8356 |
| 占 oracle | 100% | **99.9%** | **99.3%** | 98.7% | **92.8%** |

而 **hit@1 ≈ 0**：全部 38 条配方 × 59 天 = 2242 个「配方-日」里只有 **6 次**命中真 argmax；38 条中 **33 条一次都没命中**，最好的一条也只有 2/59。（计划初稿写的「0/59」只查了 4 条配方，全量复算后修正为此，结论方向不变。）

> nb12 的 `CE(s/τ, oracle_window)` 把第 2 名（值 99.9%）和第 3000 名（值为负）同等惩罚，而它要学的那个 one-hot 目标经验命中率是 0。**这是 12 的损失最具体、可量化的缺陷**，也是 R1 的全部动机。

### D2 · 曲线已经把好窗排进前列，错的是前列内部的排序

对 incumbent 自己预测出的 top-K 窗，取其中真值最高的一个（= 完美 reranker 上界）：

| K | 1（现状） | 5 | 20 | 50 | 200 |
|---|---:|---:|---:|---:|---:|
| 上界 | 6301 | 6463 | 6841 | **7082** | 7872 |
| 相对 incumbent | — | +161 | +540 | **+781** | +1571 |

> 这是通往「逼近 9001」唯一有实测支撑的方向：**不需要更准的曲线，需要在自己的 top-K 里重排**。R4 由此而来。

### D3 · 3321 全集可枚举 → 用精确 listwise，不需要 solution cache / 扰动近似

Mandi 的 solution cache、Pogančić 的 DBB、Berthet 的扰动优化器，都是**在可行集无法枚举时**的近似手段。本任务 |W| = 3321，`(B, 3321)` 张量可以整张算，Gibbs 分布是精确的。

**结论：R1 就是带熵正则的精确 Fenchel–Young 损失；DBB / perturbation 在这里只会更差，直接排除。**

### D4 · 选模尺子必须换，否则再好的损失也被噪声淹没

- 59 日 realized 均值的 SE ≈ **549**；30 日 ≈ **764**。
- nb12 `selection.json` 里逐 epoch `val_realized` 的 std 是 **383–764** —— 与纯粹的「30 天抽样噪声」完全吻合。那些 epoch 间的起伏**不含模型信息**。
- val30（10/04–11/02）前 3 天占该窗 oracle 总量的 17.9%；在 val30 上选出的最优常数窗拿到 holdout 只值 5360（低于气候平均 5644）。
- 改用**日均 capture**（每天 realized/oracle 再等权平均）做选模统计量，同一对比较的信噪比从 0.203 → **0.260**（+28%，零成本）。

---

## 1. 路线表：论文 → 本仓库函数 → 为什么可能超过窗 CE

记号：真价 `p ∈ R^96`，预测 `p̂`；块和 `S(p) = conv(p, 1_8) ∈ R^89`；窗分 `s_w(p) = S[td] − S[tc]`，`s(p) ∈ R^3321`（沿用 12 的约定，不乘 1000）。

不变性（已实测 59/59 天）：`s(a·p + b) = a·s(p)`，`a > 0` 时窗与 idle 判定不变 —— 所有损失只能建在 `s` 上。

| # | 路线 | 外部依据 | 本仓库函数 | 为什么可能 > 窗 CE |
|---|---|---|---|---|
| **R1** | **Listwise（价值加权软标签）** | Mandi et al. ICML 2022 **Eq.16**：`L = −Σ_v p_τ(v│c)·log p_τ(v│ĉ)`，目标是**真目标值的 softmax**，不是 one-hot | `loss.window_scores`（已有）、`optimize_day` 校验 | D1。12 的 CE 是本式 `τ_tgt→0` 的退化特例；把值 92.8%–99.9% 的近优窗从「全错」改回「几乎全对」 |
| **R2** | **Pairwise difference** | Mandi **Eq.13**：`[(f(v_p,ĉ)−f(v_q,ĉ)) − (f(v_p,c)−f(v_q,c))]²` | `dispatch.LEGAL_WINDOWS` | 直接回归**决策相关的分数差**；天然满足平移不变性；对 hard negative（模型自己的 argmax）配对 = 最小可行 regret 目标 |
| **R3** | **SPO+ 作为训练损失** | Elmachtoub & Grigas, *Management Science* 2022；早停用 Sp-R-IP 的验证 regret 思路 | `optimize_day`（每样本 2 次调用）、`build_day_power` | 凸、上界 regret、次梯度 `2(z_{w*(2p̂−p)} − z_{w*(p)})`。**与仓库里失败的那次明确不同**：那次是 P01 之上的线性残差校准、90 个反复接触日、无 idle；这次是序列头的训练损失、271 个新日 |
| **R4** | **两段：强曲线 + top-K reranker** | Mandi 的 ranking 视角；`EXTERNAL_EVIDENCE_BANK.md` 的 top-K gate | `LEGAL_WINDOWS`、`block_model.optimize_from_block_values` | D2：上界 +781（K=50）。唯一有实测头寸指向 9001 的路线 |
| **R5** | **决策加权凸组合集成** | InCommodities CaseCrunch 2026 亚军 writeup（异质集成 + 用任务指标选权重）；Lago et al. 2021 强基线纪律 | `evaluation.evaluate_price_predictions` | 零新训练；11 的钱锚（v1 Ridge 6184 / v2 TF 6214 / v2 LGB 6145）失败模式互补 |
| **R6** | **廉价对照（必须先被打过）** | Smets et al., *Energy* 2025（loss function tuning）；MDPI 2024 峰谷选模 | `block_model.forward_block_means` | pinball + 只罚最便宜/最贵 8 格。若 R1–R4 打不过 R6，复杂 DFL 在本数据上存疑 |

### 明确排除（写进报告，不是遗漏）

| 排除 | 理由 |
|---|---|
| DBB（Pogančić et al., ICLR 2020）、Perturbed optimizers（Berthet et al., NeurIPS 2020） | D3：全集可枚举，精确解优于近似 |
| LODL（Shah et al., NeurIPS 2022） | 学代理任务损失是为了省求解器开销；本任务一次 `optimize_day` = 一次 convolve + argmax。列为 R1–R3 全败后的备选 |
| DFF（arXiv 2405.14719）、Perturbed DFL（arXiv 2406.17085）、Learning Reachability（arXiv 2512.06600，已读：连续 SOC + stopping-time） | 合同无跨日 SOC / 效率 / 连续功率，多阶段可微优化无处落脚 |
| RL、改 `optimize_day`、59 天调参、nested 5-fold 选模 | 用户合同禁止 |

### 文献阅读深度（按 `LITERATURE_ZH.md` 的诚实标准）

- **已取全文并核对公式**：Mandi et al., ICML 2022 / PMLR 162:14935-14947（Eq.5 NCE、Eq.11 pairwise、Eq.13 pairwise-difference、Eq.16 listwise；cache 以训练集全部真最优解初始化，按 `p_solve` 概率增长，每步用全 cache）。
- **已读正文方法概述**：Learning Reachability of Energy Storage Arbitrage（arXiv 2512.06600）—— 判定为不适用（连续 SOC）。
- **仅核对出处与摘要，未逐页精读**：Elmachtoub & Grigas 2022；DFL survey（JAIR 80 / arXiv 2307.13565，11 方法 × 7 问题）；Sp-R-IP（OpenReview `o0oroLuPLZ`）；Sang et al.（arXiv 2305.00362）；Berthet et al. 2020；Pogančić et al. 2020；Shah et al. 2022。
- **本轮新检索到、写进计划的 2025–2026 工作**：
  - Smets, Toubeau, Dolanyi, Bruninx, Delarue, *Value-oriented price forecasting for arbitrage strategies of Energy Storage Systems through loss function tuning*, **Energy 2025**（ScienceDirect S0360544225027549）—— 广义损失显式编码日内价差，直接对应我们的「块和差」目标，是 R2/R6 的最近邻。
  - *Probabilistic Forecasting for Day-ahead Electricity Prices, Battery Trading Strategies and the Economic Evaluation of Predictive Accuracy*, **arXiv 2604.19580 (2026)** —— 预测精度与交易收益的经济评价必须分轨报告，支撑第 4 节的双轨结论写法。
  - *Deriving Loss Function for Value-oriented Renewable Energy Forecasting*, **arXiv 2310.00571** —— 双层规划反推损失，是 R2 的理论背书。
  - *A Decision-Focused Predict-then-Bid Framework for Strategic Energy Storage*, **arXiv 2505.01551** —— 端到端 predict-then-bid；其市场出清层不适用本合同，仅确认路线族仍活跃。

---

## 2. 损失的确切定义

### R1 Listwise（主力）—— 两个温度，这是相对 Mandi 的适配

```
q  = softmax( s(p)  / τ_tgt )      # 目标：真价上 3321 窗的价值分布，detach
π̂  = softmax( s(p̂) / τ_pred )
L1 = CE(π̂, q) = − Σ_w q_w · log π̂_w
L  = MSE( S(p̂_centered), S(p_centered) ) + λ · L1
```

`τ_tgt` 由 D1 标定：取使目标分布有效支撑约为 top-20～top-50 的值（这些窗值 92.8%–98.7% oracle）。Mandi 用单一 τ；这里分离，因为**目标平坦度是数据性质、预测锐度是优化超参**，混成一个是 12 的 τ 难调的原因之一。

### R2 Pairwise difference

每样本取 (真最优窗 `w*`, 模型当前 argmax `ŵ`) 与若干随机窗构成对：

```
L2 = mean[ ( (s_u(p̂) − s_v(p̂)) − (s_u(p) − s_v(p)) )² ]
```

### R3 SPO+（最大化形式）

```
L3 = max_w (2p̂ − p)' z_w  −  2p̂' z_{w*(p)}  +  p' z_{w*(p)}
次梯度(∂/∂p̂) = 2 ( z_{w*(2p̂−p)} − z_{w*(p)} )
```

`z_w ∈ {−1000, 0, +1000}^96` 由 `build_day_power(tc, td)` 给出；每样本两次 `optimize_day`。

### R4 Reranker

一阶段冻结曲线给出 top-K 窗；二阶段小模型 `g(块和特征, tc, td, 日历)` 对 K 个候选出分，训练用 R1 的 listwise（K 维，目标 = 该日真值 softmax）。`K ∈ {20, 50}` 作为超参。

---

## 3. 协议

**冻结部分（完全遵守用户合同）：** 360 日 → 开发 301（2025-01-02 → 11-02，其中 271 训 / 最后 30 选）→ **holdout 59（11-03 → 12-31）只打一次分**。`holdout_not_used_for_selection: true` 写进 manifest。标签里的「当天最优窗」只来自当天**训练集**真 A。

**我要加的三条护栏（理由 = D4，目的是让「超过 6301」这件事成立，而不是撞运气）：**

1. **选模统计量改为 30 日「日均 capture」**（仍然是用钱选，不是 RMSE）。信噪比 +28%，且不被 val30 里那 3 个尖峰日绑架。
2. **每条配方 5 个 seed（42–46），报告均值。** 59 日单 seed 单点比较的 SE 是 549；nb12 的 transformer 只训了 2 个 epoch、单 seed，那个 6214 不是方法的性质。
3. **59 日结果一律带按天配对 bootstrap CI**，对照三条线：`6301.33`（incumbent）、`5946`（无模型固定窗，训练段选出）、同架构 MSE 版。

---

## 3a. 与 nb12 计划（`.cursor/plans/dispatch_loss_trial_f7f448b5.plan.md`）的逐条对照

### 完全一致（沿用，不动）

| nb12 计划条目 | 本轮 |
|---|---|
| 301 开发日 = 271 训 / 最后 30 选 | 同 |
| 301 上按选中 epoch 比例重训，`scaled_count = min(80, round(best_epoch × 301/271))` | 同（复用 `run.py:88`） |
| 59 天只打一次分，不用于 early stop / 选格 / 宣布冠军 | 同 |
| `max_epochs=80`、`patience=12` | 同 |
| Oracle 窗只来自当天**训练集**真 A | 同 |
| 选模只看钱，不看 RMSE | 同 |
| 三个 compact 头（Transformer / LSTM / DLinear），contextual 特征，结构不改，只换损失项 | 同 |
| 块和口径：日中心化 → 8 格 `conv1d` → `s = S[td] − S[tc]`，**不乘 1000** | 同 |
| 损失里不处理 idle；推理时 `optimize_day(allow_idle=True)` 照常 | 同 |
| 不改 `optimize_day`、`final_metrics.json`、v1/v2 的 RMSE/MAE | 同 |
| `holdout_not_used_for_selection: true` 写进 manifest | 同 |
| 树 / Ridge 不接可导 CE | 同（见下方"口径扩张"说明） |

### 四处有意偏离（每条都有理由，也都可以关掉）

| # | nb12 | 本轮 | 为什么 | 关掉的代价 |
|---|---|---|---|---|
| **P1** | Seed 42 单次 | 网格 2 seed，最终配置 **5 seed（42–46）取均值** | 59 日均值 SE ≈ 549。nb12 的 transformer 只训了 2 epoch、单 seed，那个 6214 不是方法的性质 | 回到单点估计，无法区分方法和抽样 |
| **P2** | 选格/早停 = 30 日 **realized 均值** | 30 日 **日均 capture**（每天 realized/oracle 再等权平均） | 仍然是"用钱选"。realized 均值被尖峰日主导（val30 前 3 天占该窗 oracle 的 17.9%）；日均 capture 的信噪比 0.203 → 0.260 | 回到被 1–2 天绑架的尺子 |
| **P3** | 格子 3 点 `{(0.5,0.1),(1.0,0.1),(0.5,0.2)}` | R1 24 点、R2 9 点、R3 6 点、R6 3 点 | nb12 自己列为已知弱点（"选格极小"）；τ_tgt 是本轮的核心超参，3 点搜不到 | 搜不到 τ_tgt 的合适档位 |
| **P4** | 通过线单条 `> 6214` | 主判定 `> 6301.33`（用户指定）+ 必报按天配对 CI、无模型固定窗对照、多重比较 | 主判定没变，只是附加报告 | 报告会把噪声写成结论 |

### 口径扩张（不是偏离，是 nb12 明确留白的部分）

- **R4 / R5 不走那套可导 CE。** nb12 说"树/Ridge 没有对 3321 窗的可导 softmax，不硬塞"——这一条本轮仍然成立，我没有用 CE 训练任何树。R5 只对**冻结的**锚点曲线做凸组合；R4 的二阶段是浅层 reranker，不反传到一阶段。所以 nb12 的约束没有被违反，只是从另一扇门让 Ridge/LGB 的信息进来了。
- **R4 / R5 的协议映射**：R5 的凸权重、R4 的 `K` 与浅层模型超参，都按 (λ,τ) 同样的规则在 30 日日均 capture 上选，59 日只打一次。

### 直接可比行（P1/P2 的解法）

P1、P2 会让"本轮 vs 6301"混进"换选模尺子"的效应——这正是 nb12 自己承认过的混淆。解法是**两行都出**，成本几乎为零（网格已经跑完，只是换一个 argmax）：

- **A 行「nb12 同协议」**：seed 42 单次、按 30 日 **realized 均值** 选格 → 与 6301.33 严格可比。
- **B 行「稳健」**：5 seed 均值、按 30 日 **日均 capture** 选格 → 更可信的效应估计。

主判定看 A 行（协议一致），B 行用来判断 A 行是不是运气。两行差很大本身就是结论。

---

## 3b. 执行排期与网格（R1–R6 全跑，资源不设限）

**两段式超参搜索**（标准做法，避免用 5 seed 做网格搜索的浪费）：网格阶段每格 2 seed（42, 43）在 271 上训、在 30 日日均 capture 上比；每条路线选出的配置再在 301 上用 **5 seed（42–46）** 重训，59 日打一次分，报 5 seed 均值。

| 阶段 | 内容 | 依赖 | 序列模型训练次数 |
|---|---|---|---|
| **S0** | `diagnostics.py`：D1–D4 复算；无模型固定窗基线（全训练段 / 滚动 60 日 / weekday）；把常数窗行补进对照表 | — | **0** |
| **S1** | **R5** 决策加权凸组合：11 的钱锚（v1 Ridge / v2 TF / v2 LGB / v2 DLinear / v1 hgb_full）在 30 日日均 capture 上搜凸权重（单纯形网格 + 坐标上升） | — | **0** |
| **S2** | **R6** 廉价对照：pinball(q10/q50/q90) 与「只罚最便宜/最贵 8 格」加权 MSE。3 config × 3 family × 2 seed | — | 18 + 15 |
| **S3** | **R1** listwise：λ ∈ {0.5, 1, 2} × τ_tgt ∈ {0.05, 0.1, 0.25, 0.5} × τ_pred ∈ {0.1, 0.25} = 24 config × 3 family × 2 seed | — | 144 + 15 |
| **S4** | **R2** pairwise-diff：λ ∈ {0.5, 1, 2} × n_pairs ∈ {8, 32, 128} = 9 × 3 family × 2 seed | — | 54 + 15 |
| **S5** | **R3** SPO+：w_spo ∈ {0.25, 0.5, 1.0} × lr ∈ {1e-3, 3e-4} = 6 × 3 family × 2 seed；早停按 Sp-R-IP 用验证 regret | — | 36 + 15 |
| **S6** | **R4** reranker：一阶段来源 ∈ {S3 最佳曲线, v2 TF, R5 集成} × K ∈ {20, 50, 200} × 浅层模型 3 档 = 27 config × 5 seed。二阶段不训序列模型，成本低 | S1, S3 | 0（仅浅层） |
| **S7** | 汇总、对照表、`notebooks/notebook_12_claude.ipynb` | S0–S6 | 0 |

合计约 **252 次网格训练 + 60 次最终重训**。模型极小（`hidden=32` / `d_model=32`，`sequence_models.py`），单次 ≤80 epoch × 271 日，CPU 上分钟级。S2–S5 彼此无依赖，可并行。

**附加（不改主判定）：** 除了合同规定的「最后 30 日」选模，同时记录一个 **301 日内 rolling-origin CV**（5 折、每折 30 日、只在开发段内、不碰 holdout）选出的配置。主表仍按合同的 30 日规则出，CV 变体作为**另起一行的次要结果**，避免把「换选模尺子」混进「换损失」的效应里 —— 这正是 nb12 自己承认过的混淆。

---

## 4. 判定与预注册

**主判定（用户定义）：** 59 日 `realized_profit_mean` > 6301.33 → 记为「过线」。

**同时必报（不改变主判定，只是不让报告说谎）：**

| 对照 | 为什么 |
|---|---|
| vs 6301.33，按天配对 CI + P(≤0) | +87 量级的差在 n=59 下 P(≤0)=0.39 |
| vs 5946（无模型固定窗） | 全榜 38 条配方目前没有一条显著超过它 |
| vs 同架构 MSE 版（如 v2/lstm 5710） | nb12 用 v1/lstm 4554 做锚把效应放大了约 3 倍；公平锚点下是 +592（CI [+174, +1088]，P=0.002），这是目前唯一统计成立的结果 |
| 3 条以上新配方时的多重比较 | 零假设下「3 条里最好的 > 通过线」的概率是 0.645 |

**预注册的结论写法：**

- 过线且 vs 6301 的 CI 下界 > 0 → 「相对 incumbent 有效」。
- 过线但 CI 含 0 → 「过线，但与 incumbent 不可区分；相对无模型固定窗的增量为 X（CI …）」。
- 未过线 → 直接写未过线，并报相对 4554 / 5710 / 6301 / 5946 四个 delta。
- 任何情况都不刷新 35 配方总榜，不宣布新冠军，不改 Phase B 冻结数。

---

## 5. 交付物

```
experiments/holdout_dfl_ltr_v1/
  loss.py          R1 listwise / R2 pairwise-diff / R3 SPO+ / R6 pinball+峰谷加权
  rerank.py        R4 两段 top-K reranker
  ensemble.py      R5 决策加权凸组合
  diagnostics.py   D1–D4 复算 + 无模型固定窗基线
  run.py           S2–S5 的 271/30/301/59 runner（两段式网格，5 seed 重训）
  PLAN_ZH.md       本文件
  REPORT_ZH.md     结案（含未过线情形的如实写法）
tests/test_dfl_ltr.py
reports/holdout_dfl_ltr_v1/{leaderboard.csv, predictions.parquet,
                            dispatch_daily.csv, selection.json, manifest.json,
                            diagnostics.json, ensemble_weights.json}
notebooks/notebook_12_claude.ipynb   ← 最终交付：59 日对照表
                                       （pred_profit / realized / capture）+ 结案
```

`notebooks/notebook_12_claude.ipynb` 的结构：

1. 起点：11 的诊断 + 12 的 incumbent（6301 / 0.700），以及 D1–D4 四个实测量。
2. 六条路线各一小节：损失定义 → 选中的超参 → 59 日成绩。
3. Headline 对照表：pred_profit / realized / capture，按 realized 排序，通过线 6301.33 画成竖线；同表含无模型固定窗（5946）、气候平均（5644）、11 钱冠（6214）、12 incumbent（6301）。
4. 每行带按天配对 bootstrap CI（vs 6301）。
5. 结案按第 4 节的预注册写法，涨跌如实。RMSE 只进附录。

**测试（`tests/test_dfl_ltr.py`）**

1. `τ_tgt → 0` 时 R1 的目标分布退化为 nb12 的 one-hot（证明 12 是本损失的特例）。
2. `s(a·p + b) == a·s(p)`，且三个损失对 `b` 的梯度为 0（平移不变性）。
3. R3 的次梯度对合成线性实例与数值差分一致；`p̂ = p` 时 SPO+ 取到 0。
4. R4 的 top-K 候选集在 `K = 3321` 时退化为 `optimize_day` 的选择。
5. `optimize_from_block_values(forward_block_means(p))` 与 `optimize_day(p)` 给出同一 `(tc, td)`。
6. D1–D4 的诊断数字可复算（容差内）。

---

## 6. 风险与止损（先写下来）

1. **最可能的结果是过线但不显著。** 这条赛道上 38 条配方里没有一条显著超过无模型固定窗（5946）；6301 距它只有 +355（P=0.074）。本轮 R1–R6 六条一次跑完，所以**这一轮就是判决轮**：若六条的 5-seed 均值相对固定窗的 CI 全部含 0，**停止堆方法，写负结果**，不开第二轮。
   同时注意反向风险：六条路线 × 三个 family 意味着大量比较。零假设下「3 条里最好的 > 通过线」已经是 0.645；条数更多时这个概率更高。所以第 4 节的多重比较列是必报项，不是装饰。
2. **R4 是唯一指向 9001 的路线，也是最可能过拟合的。** top-K reranker 的候选集依赖一阶段曲线；K=50 时完美上界 7082，但 hit@50 只有 0.15，实际能拿到的只是其中一部分。二阶段必须在 271 上训、在 30 日日均 capture 上选，K 也算超参。
3. **R3 的失败不能再被当成「SPO+ 被证伪」**，反过来也一样：这次若成功，不能用来追认那次失败的 P01 残差校准。两者设定不同，报告分开写。
