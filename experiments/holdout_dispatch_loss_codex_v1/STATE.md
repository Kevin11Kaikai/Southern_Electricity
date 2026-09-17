# Codex 调度损失试验：权威状态

- 阶段：`COMPLETE_WAITING_HUMAN_REVIEW`。任务本轮已完成；没有仍在运行的训练作业，不自动进入下一轮。
- 运行：`reports/holdout_dispatch_loss_codex_v1/runs/20260914T221004Z/`。
- 交付：`notebooks/12_codex_notebook.ipynb`，已执行 7 个只读代码单元格、0 错误、嵌入 5 张真实图。40 个保护文件未变；13 项损失/预算测试、6 项只读产物检查通过。
- 授权：用户要求按相同协议执行 Codex 路线；未授权新外部费用。预算是本轮开发安排：开始 2026-09-14 22:10:04 UTC，截止 2026-09-15 00:10:04 UTC，120 分钟 / 34 次训练 / 每次 360 秒 / 总重试上限一次；不自动续期。
- 收尾检查 2026-09-14 22:40:49 UTC，已用约 30.8 分钟开发墙钟；32 次训练、0 失败、0 重试、1 次相同 DLinear MSE 重训缓存命中。训练器内部墙钟累计 163.39 秒、CPU 636.41 秒，含进程启动约 256.09 秒；外部新增费用 0。未用完额度不是继续训练的理由。
- 协议：271 日训练 / 后续 30 日选参数和轮次 / 301 日冻结重训 / 59 日统一测试。候选冻结早于唯一一次测试回放；测试后没有新训练。
- 测试口径：2025-11-03—12-31 共 59 天；本轮训练参数与损失权重未在这些天拟合。日均原始分数 `sum(price*power)`，不是人民币。
- 结论：预选软遗憾 DLinear 5,135.06，落后 Cursor 原最佳 LSTM 6,301.33；软遗憾 LSTM 分支 6,485.73，有局部改善但非预选主方案。最高对照为同初始化 Cursor 损失 Transformer 6,570.71，不归功于 Codex 新损失。
- 机制证据：仅改选模标准没有明显改善；软遗憾作用依赖模型。LSTM 相对 Cursor 的优势前 30 日较大、后 29 日接近持平。DLinear 12-04 主要输在充电窗口，LSTM 11-23 主要赢在放电窗口。
- 限制：单种子；soft-regret 与 idle 是组合改动；软遗憾对 MSE 的比较包含三配置搜索差异；原 Cursor 构造前未固定种子，因而保留 matched CE。未建立 OS 级隔离，不声称正式隔离 Agent 实验。
- 原始证据：`comparison.csv`、`dispatch_daily.csv`、`predictions.parquet`、`analysis.json`、`ledger.jsonl`；版本与哈希见冻结文件。成本与验收见 `closeout_verification.json`、`notebook_execution.json`。
- 下一动作：等待用户 Human Review，先阅读 notebook 图 1、图 3、图 5。不要重新训练或再用测试分数调参。只读查看 notebook 不需要恢复训练；材料说明见同目录 `README_ZH.md`。
