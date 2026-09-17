# Codex 8+8 调度损失试验：已完成，等待 Human Review

阅读入口是 `notebooks/12_codex_notebook.ipynb`，含实际执行输出、五张真实结果图和全部二十行对比。

## 结果一句话

测试前选中的 Codex 软遗憾 DLinear 没有获胜；LSTM 分支有局部改善，而同初始化的 Cursor 损失 Transformer 对照更好。不要把对照的成绩记为新损失的功劳，也不要把测试后最好分支改称预先选定的主方案。

2025-11-03—12-31 的 59 天 holdout 日均原始调度分数：

- 匹配 LightGBM baseline：5,786.10。
- Cursor 原最佳 LSTM：6,301.33。
- Codex 软遗憾 LSTM：6,485.73，是声明的分支之一，不是预选主方案。
- Codex 软遗憾 DLinear：5,135.06，是按开发验证分数预选的主方案。
- 同初始化 Cursor 损失 Transformer 对照：6,570.71，不是 Codex 新损失。

本轮参数未在这 59 天拟合；损失权重和轮数只在 2025-10-04—11-02 的 30 个开发验证日选择。原始分数是 `sum(price * power)`，不是人民币。单种子结果，不声称稳定优势。

## 实际执行的协议与边界

271 个完整开发日训练，随后 30 天选择，再以冻结轮数重训全部 301 日，最后统一回放 59 天。三个模型及 67 个 contextual 特征与 Cursor 相同。LightGBM baseline 仍为原有 11 特征，与它相比的增益不单独归因于新损失。预处理只在相应训练日拟合。保持原优化器及 3,321 个合法充放组合、不交易规则。

新目标为日中心化块和 MSE 加软预期遗憾，含不交易项。对照包括同初始化 Cursor 损失、点 MSE 按 RMSE 选模、同一条 MSE 训练轨迹按调度分数选模。每种选模规则独立停止，不使用其停止后的训练轮次。三种损失参数配置在 `protocol.json` 中冻结。

原 Cursor 在模型构造后才设置种子，CPU 环境也没有完整对应记录；新运行在构造前固定 seed=42。同初始化 Cursor 对照用来限制因果解释。新损失与不交易项未分开消融；MSE 没有三配置损失搜索。模型重载一致不等于训练稳定。没有启动真实研究 Agent 会话或强化学习。

## 本轮实际命令

项目根目录，使用已有 `south_grid` Python；以下命令已实际执行。训练已经结束，不要为了看图重新训练。

```powershell
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m unittest discover -s tests -p test_codex_dispatch_loss.py -v
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m experiments.holdout_dispatch_loss_codex_v1.run prepare
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m experiments.holdout_dispatch_loss_codex_v1.run train
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m experiments.holdout_dispatch_loss_codex_v1.evaluate
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m experiments.holdout_dispatch_loss_codex_v1.report
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m unittest discover -s tests -p test_codex_dispatch_artifacts.py -v
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m experiments.holdout_dispatch_loss_codex_v1.execute_notebook
& '<python-env>/envs/south_grid/python.exe' -X utf8 -B -m experiments.holdout_dispatch_loss_codex_v1.run check-protected
```

`train` 检测到已冻结候选后直接退出；`evaluate` 检测到完整结果后直接复用。缺失文件时 notebook 不调用以上训练入口。

13 项损失/预算单元测试与 6 项只读产物检查通过；notebook 的 7 个只读代码单元格实际执行、0 错误，嵌入 5 张图片。`execute_notebook` 使用运行目录里的局部 kernel 配置，不安装全局 kernel；关闭本轮开发预算后，仍可在现有 south_grid 内核中直接打开并运行 notebook 的只读单元格。

## 证据索引

运行根目录：`reports/holdout_dispatch_loss_codex_v1/runs/20260914T221004Z/`。

- `protocol_frozen.json`：训练前方法、预算、数据和源码哈希。
- `selection_frozen.json`：只用开发验证期选择的参数、轮次。
- `candidate_freeze.json`：测试前模型哈希与主方案。
- `jobs/*/job.json`、`result.json`、`stdout.log`、`*.pt`：本任务生成的每次配置、训练、模型与失败记录。只限这个运行目录。
- `ledger.jsonl`：32 次训练启动、0 失败、0 重试、1 次完全相同重训配置缓存命中，以及评价和绘图成本。
- `comparison.csv`、`dispatch_daily.csv`、`predictions.parquet`：全部成绩、逐日决策与点预测。
- `analysis.json`：预先主方案、各分支、对照、时间分段和负结果。
- `evaluation_checks.json`：1,180 个候选日的独立计分与动作检查、重载、日期一致性。
- `protected_hashes_before.json` / `protected_hashes_after.json`：显式准入保护清单，不扫描封存区。
- `figures/`、`figure_manifest.json`、`case_details.csv`：真实结果图、来源和案例数字。

工作代码累计训练墙钟 163.39 秒、CPU 636.41 秒；含进程启动开销约 256.09 秒，不能冒称为整个开发耗时。预算起点 2026-09-14 22:10:04 UTC，120 分钟、34 次训练上限、360 秒单次超时、总重试至多一次。未新增计费 API 或云算力，未用完预算也不继续调参。

环境复用：Python 3.11.15、PyTorch 2.14.0+cpu（项目 `.torch_runtime`）、NumPy 2.4.6、pandas 3.0.5、Matplotlib 3.11.1、PyArrow 25.0.1、nbformat 5.10.4、nbclient 0.11.0、ipykernel 7.3.0。未安装新依赖、未改全局环境。运行账户不具有操作系统级数据隔离；这里只核验代码路径与实际输入，没有声称正式隔离实验。

## 支持与不支持

支持：软遗憾 LSTM 在这份测试上高于匹配 MSE 与 Cursor 损失；开发排名未能保持；12 月 4 日主方案主要输在充电窗口，11 月 23 日 LSTM 主要赢在放电窗口。

未支持：仅改选模就明显改善；新损失跨模型普遍更好；预选主流程击败 Cursor；单种子结果代表可靠性。季节变化、样本量、软硬目标失配和模型能力分别有多大责任，尚不能由本轮唯一确定。

教学材料列出 Kaggle 原作者方案及 ICML 2022 决策排序论文。它们是设计参考，不是同赛题 SOTA 排名证据。

本轮停止新训练；等用户 Human Review。
