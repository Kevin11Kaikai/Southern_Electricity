# 从点预测误差到受约束调度价值

一份双语技术报告的复现包。核心问题：**当电价预测的好坏由它产生的调度决策来衡量、而不是由它自己的误差来衡量时，会发生什么。**

*English version: [README.md](README.md)*

📄 **直接阅读：** [`report/pdf/dispatch_value_report_zh.pdf`](report/pdf/dispatch_value_report_zh.pdf)（16 页）·
[`report/pdf/dispatch_value_report_en.pdf`](report/pdf/dispatch_value_report_en.pdf)（18 页）

---

## 问题

一天 96 个 15 分钟时段。电池可以按固定功率连续充电八格，随后连续放电八格，或者全天空闲——共 3321 个合法主动作加一个空闲动作。预测器给出 96 个价格；求解器据此选一个动作；该动作再用实际发生的价格结算。

项目最初按 RMSE 给预测器排名。把保存的预测曲线接入同一个调度求解器后发现：**更低的逐点误差并不能可靠地识别出更高的实现调度价值。** 报告从可行动作集合的几何结构解释这一现象，围绕决策重构学习目标，实现可训练的代理损失，并在评分前冻结的协议下评价结果。

## 证据支持什么，不支持什么

| 主张 | 状态 |
|---|---|
| 在指定的 59 日窗口内，冻结协议实现 **6693.88**，MSE 训练的参考为 **6214.36**，观测增益 **479.51（7.72%）** | 结算事实，已独立复算 |
| 点误差不能可靠地对保存候选的调度价值排序 | 成立（v1 *r* = +0.351，v2 *r* = +0.005；日中心变体同样） |
| 尺度不变的窗口排序一致性比点误差更能跟踪实现价值 | 作为**结构诊断**成立（*r* = +0.849 / +0.777），不等于已验证的选模能力 |
| 实际训练代理损失是更好的跨模型选模指标 | **不成立**——其关联不稳定（*r* = +0.320 / −0.165），报告如实报告了这一负面结果 |
| 该增益可迁移到新时期 | **未确立。** 净增益的 86.0% 来自一天；配对 7 日块区间为 [−227.87, 1626.28]；主方案最差单日（−5922.66）深于参考（−2340.48） |
| 增益由损失函数单独造成 | **未确立。** +160.99 / +318.52 是算术分解而非因果分解；同初始化条件下，**旧**损失的对照分数高于新损失的单种子均值 |
| 已验证完整电池模型 | **没有。** 未实施 SOC、效率、衰减或网络约束；结果仅在 8+8 动作合同下成立 |

这 59 日在项目早期已被查看过。全文一律称其为**历史上已暴露的评价期**，不表述为前瞻性确认集。

## 如何复现

三条路径，由浅入深。

### 一、只读

打开 `report/pdf/` 下的 PDF，不需要任何工具链。

### 二、重建全部图、表与 PDF（约 2 分钟）

```bash
pip install -r requirements.txt -r requirements-audit.txt
python tools/build_report.py          # 6 图、7 表、两版 PDF -> build/
python tools/verify_reproduction.py   # 与仓库内已存产物逐字节比对
python tools/check_numbers.py         # 正文每个数字都能追溯到 report/shared/data
```

`tools/build_report.py` 只读取 `report/shared/data/` 下的聚合证据记录，不训练模型、不读取原始数据。需要一个 LaTeX 引擎——优先检测 [Tectonic](https://tectonic-typesetting.github.io)，否则使用 `latexmk` + `xelatex`。

若需要图形输出逐字节一致，请改装 `requirements.lock.txt`；其他 matplotlib 版本能复现相同数字，但可能产生外观相同、字节不同的 PDF。

`tools/make_overleaf_zip.py` 可把两版打包成可直接上传 Overleaf 的项目。

### 三、从原始数据重新推导全部结果

**数据不随本仓库发布**——获取地址、放置路径以及八个实验目录的确切执行顺序见 [DATA.md](DATA.md)。生成 `reports/` 之后：

```bash
python report/shared/derive_records.py --repo-root . --out report/shared/data
python tools/independent_audit.py     # 不使用报告代码，独立重解 62 条曲线
```

`tools/independent_audit.py` 用 `np.convolve` 重建块和，把每一天重新送进 `src/phase_b/dispatch.py` 求解并结算，再与打包记录比对。对原始产物运行时，62 条曲线 × 59 日的动作零不一致，数值一致到 1.5 × 10⁻¹¹。

## 目录

```
report/shared/          唯一真相来源
  data/                 16 个聚合证据记录（不含价格曲线）
  figures/              6 张矢量图 + PNG 预览
  tables/               7 张自动生成的 LaTeX 表
  equations.tex         全部 19 个编号公式，两版共用
  derive_records.py     从 reports/ 重建 data/（需自备数据）
  build_figures.py      仅从 data/ 绘图
  build_tables.py       仅从 data/ 生成表格
report/{zh,en}/         各版的 main.tex 与 author.tex
report/pdf/             两份已编译 PDF
src/phase_b/            调度求解器、动作合同与特征管线
experiments/            八个实验目录（仅代码）
notebooks/              notebook 11–14，含已存输出的只读研究记录
tools/                  构建、验证与审计入口
audit_records/          两轮独立审核及其机器可读输出
configs/phase_b.toml    路径与切分定义
```

`src/phase_b/dispatch.py` 是任务的权威定义：`optimize_day` 枚举 3321 个合法窗口，按确定性破同分规则取 argmax；`contracts.py` 校验一天的功率向量是否合法。

## 验收状态

以下均在本地、使用锁定版本与 LaTeX 引擎核对通过：

- 两版均可从干净检出重新编译；页数（16 / 18）与逐页提取文本和交付 PDF 一致
- 六张图 PDF 重画后 SHA-256 完全相同
- 十四个表格文件重新生成后完全相同
- 中英文两版含有相同的 96 个小数值，不存在某个主张只在一种语言里被限定的情况
- 正文中每个小数都能在 `report/shared/data/` 中找到来源

**未在 Overleaf 线上测试。** 构建元数据中 `overleaf_online_tested` 为 `false`；ZIP 面向普通 TeX Live 上的 XeLaTeX。

## 来源与边界

目标为匿名价格序列。所引文献提供方法背景，**不能**据此认定数据来自某个特定电力市场；报告不主张市场身份、权威物理单位或任何生产部署成果。分数使用原始评分单位——不是人民币，也不是扣除成本后的交易利润。

报告与审计脚本使用 AI 辅助整理项目已有实验；每一项定量主张都可追溯到 `report/shared/data/` 中的记录与 `audit_records/` 中的两轮审核。[AUDIT.md](AUDIT.md) 记录了两轮审核各发现了什么，包括第二轮抓到并修正的一处算术符号错误。

## 许可

代码与文档：MIT，见 [LICENSE](LICENSE)。
数据记录与数据集归属：见 [NOTICE.md](NOTICE.md)。
