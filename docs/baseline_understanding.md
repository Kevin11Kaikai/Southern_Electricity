# Southern Electricity (南方电网) 基线深度解析技术说明书

本技术说明书旨在对当前项目已有的 **基线系统 (Baseline)** 进行深度解构。内容涵盖系统的工作原理、数据输入输出、优化目标、物理约束、潜在的失效模式及其检测手段，并划定智能体可修改与禁止篡改的系统边界。最后，给出第一阶段（LP vs MILP 负电价套利）的极简实验方案。

---

## 一、 当前基线系统的工作原理

目前工作区包含的基线系统主要由两个 Python 脚本构成：
1.  **[`lgb_baseline.py`](file:///d:/others/Southern_Electricity/lgb_baseline.py) (预测核心)**：
    *   **电价预测**：使用 LightGBM 回归模型（基于 RMSE 损失函数）预测 15分钟 粒度的节点电价 $A$。
    *   **特征工程**：将日前电网边界条件特征（系统负荷、风/光/水电预测、非市场化出力、联络线预测值等）与简单的时间轴特征（`hour`, `minute`, `dayofweek`, `month`）进行拼接对齐。
    *   **储能决策**：此脚本中的 `generate_strategy` 为**空占位符 (`pass`)**，不具备真实的电池充放电决策逻辑。
2.  **[`analyze_baseline.py`](file:///d:/others/Southern_Electricity/analyze_baseline.py) (校验与格式化输出)**：
    *   **格式对齐与导出**：检查预测格式是否与 [`output_demo.csv`](file:///d:/others/Southern_Electricity/output_demo.csv) 保持行数与时间戳完全对齐。
    *   **储能决策填充**：因为日前决策模块尚未开发，该脚本自动将提交流水线中的充放电有功功率 `power` **全量硬编码填充为 `0.0`**（即电池闲置、零充放电行为）。

---

## 二、 系统输入与输出

### 1. 输入数据源
*   **训练特征集**：[`to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv`](file:///d:/others/Southern_Electricity/to_sais_new/to_sais_new/train/mengxi_boundary_anon_filtered.csv)  
    包含历史时间戳及电网边界的 7 维日前状态预测值。
*   **训练电价标签**：[`to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv`](file:///d:/others/Southern_Electricity/to_sais_new/to_sais_new/train/mengxi_node_price_selected.csv)  
    包含历史时间戳及对应的真实出清实时节点电价 $A$。
*   **测试特征集**：[`to_sais_new/to_sais_new/test/test_in_feature_ori.csv`](file:///d:/others/Southern_Electricity/to_sais_new/to_sais_new/test/test_in_feature_ori.csv)  
    用于测试推理阶段的日前边界条件预测输入。
*   **模板样式文件**：[`output_demo.csv`](file:///d:/others/Southern_Electricity/output_demo.csv)  
    规定提交流水线的行数、时间序列及列字段（`times`, `实时价格`, `power`）。

### 2. 输出产物
*   **初步预测结果**：`outputs/output_price/lgb_baseline_output.csv`  
    由 `lgb_baseline.py` 直接预测出的测试集电价 $A$。
*   **标准提交流水线文件**：`outputs/output_price/lgb_baseline_submit.csv`  
    经过格式校验、列名对齐，并填充 `power` = `0.0` 的最终提交版本。
*   **4大性能评估诊断图**：`outputs/figures/` 目录下的 PNG 图表。

---

## 三、 基线系统优化的目标函数

### 1. 电价预测阶段
*   **目标函数**：最小化验证集上的均方根误差 (RMSE)。
    $$\min \text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^{N} (y_t - \hat{y}_t)^2}$$
    通过控制树分裂节点数量（`num_leaves=63`）与学习率（`learning_rate=0.05`），配合 `early_stopping` 进行泛化保护。

### 2. 储能优化阶段
*   **当前状态**：**无优化目标**。基线处于纯闲置状态：
    $$Power_t = 0.0, \quad \forall t \in \mathcal{T}$$
    储能期望总收益为 $0$。

---

## 四、 物理约束条件

对于任何实盘上线的储能决策系统，必须严格遵守以下四项**硬性物理约束**（目前基线因全填 `0.0` 处于平凡满足状态）：

1.  **物性能量守恒约束 (SOC 递推)**：
    电池当前时刻的电量状态是由上一时刻电量加上充电输入（扣除效率损耗）并减去放电输出（增加效率折算）构成的：
    $$E_t = E_{t-1} + P_{c, t} \cdot \eta \cdot \Delta t - \frac{P_{d, t} \cdot \Delta t}{\eta}$$
    *(其中 $\Delta t = 0.25 \text{ 小时}$, 单向效率 $\eta = 0.95$)*
2.  **电池容量上下限安全约束**：
    电池荷电电量 $E_t$ 不能越界，必须保持在安全 SOC 区间（$10\% \sim 90\%$）：
    $$E_{cap} \times SOC_{min} \le E_t \le E_{cap} \times SOC_{max}$$
    $$200 \text{ kWh} \le E_t \le 1800 \text{ kWh}$$
3.  **充放电额定有功功率约束**：
    充放电功率不得超过电池逆变器额定上限：
    $$0 \le P_{c, t} \le P_{max}^{chg} \quad (1000 \text{ kW})$$
    $$0 \le P_{d, t} \le P_{max}^{dis} \quad (1000 \text{ kW})$$
4.  **充放电二元互斥约束**：
    电池在物理上不能在同一时刻（15分钟时段内）既执行充电又执行放电：
    $$P_{c, t} \times P_{d, t} = 0$$

---

## 五、 系统失效模式 (Failure Modes) 与检测机制

当后续的 AutoResearch 智能体试图重构该基线时，系统可能会引入以下**失效模式**。我们必须建立严格的自动化检测机制：

| 失效模式 (Failure Mode) | 发生机制说明 | 本地检测脚本的判定逻辑 |
| :--- | :--- | :--- |
| **1. 物理空转骗补** <br>(Simultaneous Chg/Dischg) | 智能体在面对负电价时，采用连续线性松弛（LP Relaxation）模型，导致在同一时段最大化充电和放电，骗取负电价补贴。 | $\sum_{t=1}^{T} \min(P_{c,t}, P_{d,t}) > \epsilon$ <br>若在任一时段内充放电有功乘积非零，则触发物理违规报警。 |
| **2. 荷电状态越界** <br>(SOC Boundaries Violation) | 充放电累计电能计算发生浮点数截断误差或约束漏配，导致电池发生过充或过放。 | $E_t < 200 - \epsilon$ 或 $E_t > 1800 + \epsilon$ <br>重构 $E_t$ 轨迹，检测是否存在任意一步越出 $[200, 1800] \text{ kWh}$。 |
| **3. 未来信息泄露** <br>(Future Information Leakage) | 智能体在特征工程中使用了 `shift(-k)` 等前瞻未来函数，将明天或几个小时后的真实价格作为特征输入，导致预测指标虚高。 | 对特征工程代码进行 AST 静态语法审查；或者建立严格的“时序向前滚动验证（Rolling OOS）”沙箱进行交叉自检。 |
| **4. 数据制造/欺诈** <br>(Data Fabrication / Hallucination) | 智能体未真正执行物理优化，而是直接在文本报告或临时输出中编造虚假的利润数额。 | **数据源隔离审查**：不允许读取智能体自我报告的数字，利润必须由评估器基于最终导出的 `submit.csv` 进行重新独立仿真计算。 |

---

## 六、 研发边界：只读基准代码 vs 智能体可修改代码

为了实现防篡改、防数据欺诈的高置信度 AutoResearch 评测环境，项目的代码文件必须严格划分为以下两类边界：

```
Southern_Electricity/
├── [🔒 只读评测边界 - 禁止修改 (Read-Only)]
│   ├── output_demo.csv                 # 官方标准提交模板文件
│   ├── evaluate_strategy.py            # [待建立] 核心物理校验与利润审计器 (Validator)
│   ├── to_sais_new/                    # 原始数据集 (Train / Test / NC 气象数据)
│   └── docs/                           # 评测基准决策、说明与大屏文件
│
└── [🛠️ 智能体开放边界 - 允许修改 (Modify-Allowed)]
    ├── lgb_baseline.py                 # 特征工程、模型构建与策略生成的逻辑
    └── analyze_baseline.py             # 预测格式校验、可视化与终版 Submit 生成逻辑
```

*   **只读防线 (Security Boundary)**：`evaluate_strategy.py` 将作为评测系统沙箱的核心审计器。当智能体完成 `lgb_baseline.py` 的迭代后，评测框架会调用只读的 `evaluate_strategy.py` 直接提取智能体导出的 `submit.csv`，计算出最终得分。**智能体无权访问和修改校验器的内部规则。**

---

## 七、 第一阶段极简实验方案：负电价下的 LP 松弛 vs MILP

为了验证日前出清价格包含“负电价”时线性规划松弛的物理失效问题，我们需要设计如下的**第一阶段基准评测实验**：

### 1. 实验控制变量
*   **实验组 A (LP 松弛)**：
    不包含二元互斥变量 $u_t, v_t$，允许充放电功率 $P_{c,t}, P_{d,t}$ 作为独立的非负连续实数。
*   **实验组 B (MILP 约束)**：
    引入二元决策变量 $u_t, v_t \in \{0,1\}$，施加严格互斥约束 $u_t + v_t \le 1$。
*   **电价环境**：
    从历史数据中筛选出包含深度负电价的典型日（如风光大发、负荷处于低谷的午间时段）。

### 2. 实验执行步骤
1.  **数据提取**：截取包含负电价的 96 个时段（一天）的价格序列。
2.  **模型求解**：在本地调用 Pyomo 编写两个模型，分别使用 `HiGHS` 或 `CBC` 对实验组 A 和 B 进行求解。
3.  **结果导出**：记录各自产生的充电功率、放电功率、SOC 荷电曲线以及期望收益值。
4.  **物理校验检测**：
    *   统计实验组 A 的 $\sum \min(P_{c,t}, P_{d,t})$，若数值大于 0.1，证明 **LP 松弛在负电价下物理失效，产生了电池自我空转骗取补贴的物理谬误**。
    *   统计实验组 B 的结果，证明其虽然求解计算时间略有增加，但在物理上是严格互斥可执行的，收益也是合规的。
