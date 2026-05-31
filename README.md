# Southern_Electricity

这是一个用于电力/时序预测的仓库，包含 LightGBM 基线训练脚本、分析脚本、以及原始 NetCDF 格式的数据集（未包含在仓库内）。

## 目录结构（简要）
- `analyze_baseline.py`：基线分析/评估脚本
- `lgb_baseline.py`：LightGBM 训练与预测示例
- `to_sais_new/all_nc/`：按日分割的 NetCDF 数据（数据量大，默认不提交）
- `outputs/`：模型输出、图像和提交文件
- `docs/`：项目文档、研究计划与进度仪表

## 快速开始
1. 创建并激活 Python 虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Unix/macOS
.venv\Scripts\activate     # Windows PowerShell
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备数据：
- 本仓库未直接包含大体积的 NetCDF 数据（`to_sais_new/all_nc/`），请在 README 或外部存储中提供下载链接或脚本，将数据放在该目录下。

4. 运行示例：

```bash
python lgb_baseline.py    # 训练/预测（查看脚本内部参数）
python analyze_baseline.py # 生成评估报告/图表
```

## 输出
- 结果与提交文件位于 `outputs/`，示例：`outputs/output_price/lgb_baseline_output.csv`，`outputs/output_price/lgb_baseline_submit.csv`。

## 注意与建议
- 不要将 `to_sais_new/all_nc/` 的原始数据直接提交到 Git 仓库；如果需要共享，使用外部数据存储或启用 Git LFS。
- 在发布前检查 `.env` 或脚本中是否包含凭据/密钥。

## 许可证
项目以 MIT 许可证发布（见 `LICENSE`）。

## 贡献
欢迎提 issue 或 PR，若需复现实验请在 issue 中说明所使用的数据与运行命令。
