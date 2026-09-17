# Audit trail

The report went through two independent review rounds before publication. Both
rounds re-derived the numbers from the frozen artifacts rather than accepting the
previous round's claims. This file summarizes what they found. The full records
are in [`audit_records/`](audit_records/).

*中文摘要见文末。*

---

## Round 1 — evidence and provenance review

Full record: [`audit_records/round1_claude_review_ZH.md`](audit_records/round1_claude_review_ZH.md)

Every headline number was recomputed from `reports/**` without reusing the
report's own generator. No numerical error was found. Three substantive gaps were:

**The model space in Equation (1) did not match the evidence set.** `H` was
written as three families while the R1 scatter plots draw on ten — ridge and a
seasonal-mean baseline sit at visually prominent positions. The formulation and
the evidence had to be made consistent.

**The two point-loss panels were produced by different fitting rules.** In v1,
tree candidates were fit for a fixed number of boosting rounds with no early
stopping and no recorded hyper-parameter search; in v2 they were tuned and stopped
on the 30 validation days. The report correctly refused to pool the panels but
did not say why.

**Only the mean was reported, never the downside.** The frozen protocol's worst
single day is −5922.66 against −2340.48 for the reference — a fact any reviewer
would compute in minutes, and one that belongs beside a headline mean gain. The
loss tail, the negative-day counts and the win/loss totals were added to the
paired-evidence table, to the results text, to the limitations, and to the daily
figure, which previously marked only the large positive day.

Five smaller items were fixed: a rounding artifact in the decomposition
identity, a hard-coded percentage in a figure annotation, ASCII hyphens standing
in for minus signs in generated tables, an unexplained vertical jitter in a
figure, and a near-empty orphan page in the English edition.

## Round 2 — adversarial audit

Full record: [`audit_records/round2_codex_audit_ZH.md`](audit_records/round2_codex_audit_ZH.md)

The second round was run with an explicit instruction to treat round 1's
conclusions as unverified claims. It re-solved all 62 curves and 3,658 actions
through the project solver using an independent block-sum implementation, with a
separately written bootstrap.

**It found a defect round 1 had introduced.** The downside paragraph added in
round 1 said that the winning-day total (+46151.09) and the losing-day total
(−17859.81) differ by the net gain of 28291.28. The two figures are already
signed: they **sum** to the net gain. Subtracting them gives 64010.89. The
sentence was wrong in both languages; it was corrected and the generator now
asserts the three-way identity.

Seven further corrections tightened claims rather than changing results:

- the v1/v2 fitting-rule description was over-generalized — the actual code has
  per-family exceptions that a manifest summary had hidden
- two validation-period summaries were presented as paired metrics of one locked
  model when they are per-seed bests at different epochs
- "all training and evaluation objectives are risk-neutral expected score"
  conflated the outer objective with the training surrogate, which is not
- the resampling intervals resample days only, holding the candidate set and the
  training randomness fixed; that conditioning is now stated prominently
- the window-ranking diagnostic and realized value are functions of the same two
  vectors, so part of their high correlation is algebraic — now said explicitly
- the amplitude sensitivity check had been applied to one diagnostic but not the
  other; it was extended to the actual training surrogate, where it gives
  +0.059 / −0.361
- some verification language claimed more than was reproducible

Both rounds confirmed the protected artifacts were unchanged: 2,090 file hashes
identical before and after.

## What neither round could establish

These remain open and are stated as open in the report:

1. **Whether the gain transfers.** 59 days, historically exposed, heavy-tailed,
   86% of the net gain on one date, paired interval crossing zero. Needs a new
   rolling evaluation period.
2. **Whether the surrogate selects better models.** This needs validation-period
   predictions excluded from fitting. Re-running inference from checkpoints that
   were themselves refit on the validation dates cannot supply it.
3. **Whether the loss is causally responsible.** Needs a matched family,
   initialization, training budget, selection rule and seed count. The one
   matched-initialization control available points the other way.
4. **Physical feasibility.** No SOC, efficiency, capacity, degradation, network
   or cycling-cost constraint is implemented.
5. **Interval coverage.** The day-block percentile intervals are a conditional
   sensitivity device; their coverage under 59 days and a small fixed candidate
   set was not established by simulation or theory.

## Reproducing the audit

`tools/independent_audit.py` in this repository performs the core of round 2's
recomputation: it rebuilds block sums with `np.convolve`, re-solves every saved
day through `src/phase_b/dispatch.py`, and compares against the packaged records.
It needs `reports/**`, which you regenerate per [DATA.md](DATA.md).

Run against the original artifacts it reports zero action mismatches across 62
curves × 59 days and agreement with the packaged records to 1.5 × 10⁻¹¹.

The machine-readable outputs from round 2 are in `audit_records/`:
`all_required_numeric_comparisons.csv`, `independent_summary.json`,
`solver_replay.json`, `controls_replay.json`, `independent_correlations.csv`,
`bootstrap_diagnostics.json` and others.

---

## 中文摘要

报告在发布前经过两轮独立审核，两轮都从冻结产物重新推导数字，而不是接受上一轮的结论。

**第一轮**复算了全部头条数字，未发现数值错误，但指出三处实质缺口：公式 (1) 的模型空间
只写了三族而 R1 证据集实际有十族；v1 与 v2 两个候选面板的拟合规则不同（v1 树模型无早停、
无超参搜索）而报告未说明；以及只报均值不报下行——冻结协议的最差单日 −5922.66 明显深于
参考的 −2340.48。这些连同五处较小问题（分解式舍入、图注硬编码百分比、表格负号、
图 5 抖动未说明、英文版孤立尾页）一并修复。

**第二轮**在"把第一轮结论视为未经验证的主张"的指令下进行，用独立实现重解了全部 62 条
曲线、3,658 个动作。它发现了第一轮**引入**的一处缺陷：下行段写"胜日合计 +46151.09 与
负日合计 −17859.81 之差即净增益 28291.28"，但两数已带符号，应为**相加**；相减会得到
64010.89。两版均已改正，生成器现在断言三项恒等式。另有七处收紧了表述而非改变结果，
包括 v1/v2 训练规则的过度概括、val30 两个汇总值被写得像同一模型的配对指标、把训练代理
与外层目标混同、重采样区间的条件范围不够醒目、窗口秩诊断与实现分存在代数耦合、
两类代理的幅度检查不对称，以及部分核验宣称超出可复现范围。

两轮均确认受保护产物未被修改（2,090 个文件哈希前后一致）。

**两轮都无法确立的**：增益能否迁移到新时期、代理损失能否用于选模、损失是否具有独立
因果贡献、电池物理可行性、以及区间的覆盖率性质。这些在报告中均作为开放问题明确标出。
