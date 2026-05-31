# Southern Electricity (南方电网/电力市场) 提分指南：如何大幅超越基线 (Baseline)

本指南针对**电价预测 (预测算法)** 与 **储能充放电决策 (策略控制)** 两大核心任务，提供了工业界通用的高阶调优思路与具体落地计划。

---

## 🚀 核心提分策略总览

要在本项目中跑出远超 Baseline 的成绩，优化工作必须双管齐下：

```mermaid
graph TD
    A[提分双支柱] --> B[电价预测端 (降低预测误差)]
    A --> C[储能控制端 (实现从 0 到 100 万的收益突破)]
    
    B --> B1[核心时序特征工程 Lag / Rolling]
    B --> B2[基于 NC 天气预报格点的新能源出力特征]
    B --> B3[非对称损失函数与分位数回归]
    
    C --> C1[基于线性规划 (Linear Programming) 的最优套利]
    C --> C2[基于滚动时域控制 (MPC) 的抗预测误差算法]
    C --> C3[多场景随机规划 (Stochastic LP)]
```

---

## 一、 储能决策端优化：从 0 收益到数学最优解 (第一优先级)

目前 Baseline 默认的 `power` 输出全部填充为 `0.0`，储能完全闲置，**项目收益为零**。因此，实现一个符合物理约束的储能控制策略是**最立竿见影的提分手段**。

### 1. 黄金标准：基于线性规划 (Linear Programming, LP) 的数学最优套利

由于电价预测值 $Price_t$ 是已知的，在已知序列下，储能电池的充放电行为可以完美地建模为一个**单目标线性规划**问题。这比任何人工设计的启发式规则（如高抛低吸阈值）都要优越，且求解速度极快（几千个决策变量在 1 秒内即可求出全局数学最优解）。

#### 数学模型构建：

*   **决策变量**：
    *   每个时段 $t$ 的放电功率 $P_t^{dis} \ge 0$
    *   每个时段 $t$ 的充电功率 $P_t^{chg} \ge 0$
    *   每个时段结束时的电池电量状态 $SOC_t$（State of Charge，百分比）

*   **目标函数（收益最大化）**：
    $$\max \sum_{t=1}^{T} \left( Price_t \times (P_t^{dis} - P_t^{chg}) \times \Delta t \right)$$
    *(其中 $\Delta t = 0.25$ 小时)*

*   **物理约束条件**：
    1.  **电量递推约束**：
        $$SOC_t = SOC_{t-1} + \left( P_t^{chg} \times \eta_{chg} - \frac{P_t^{dis}}{\eta_{dis}} \right) \times \frac{\Delta t}{E_{cap}}$$
        *(其中 $E_{cap}$ 为电池总容量，$\eta_{chg}, \eta_{dis}$ 分别为充、放电效率，一般在 $0.9 \sim 0.95$ 之间)*
    2.  **容量上下限约束**：
        $$SOC_{min} \le SOC_t \le SOC_{max}$$
        *(一般限制在 $10\% \le SOC_t \le 90\%$ 以保护电池寿命)*
    3.  **功率上限约束**：
        $$0 \le P_t^{chg} \le P_{max}^{chg}$$
        $$0 \le P_t^{dis} \le P_{max}^{dis}$$
    4.  **互斥约束（可选，通常松弛）**：
        电池在同一时刻不能既充电又放电。

#### Python 落地代码推荐：
使用内置的 `scipy.optimize.linprog` 或三方库 `PuLP`。以下是使用 `PuLP` 的伪代码框架：
```python
import pulp

def optimize_storage(prices, E_cap=2000, P_max=1000, eta=0.95, soc_min=0.1, soc_max=0.9):
    T = len(prices)
    prob = pulp.LpProblem("Battery_Arbitrage", pulp.LpMaximize)
    
    # 定义变量
    p_chg = [pulp.LpVariable(f"p_chg_{t}", lowBound=0, upBound=P_max) for t in range(T)]
    p_dis = [pulp.LpVariable(f"p_dis_{t}", lowBound=0, upBound=P_max) for t in range(T)]
    soc = [pulp.LpVariable(f"soc_{t}", lowBound=soc_min, upBound=soc_max) for t in range(T)]
    
    # 目标函数
    prob += pulp.lpSum([prices[t] * (p_dis[t] - p_chg[t]) * 0.25 for t in range(T)])
    
    # 约束条件
    for t in range(T):
        if t == 0:
            # 假设初始电量为 50%
            prob += soc[t] == 0.5 + (p_chg[t] * eta - p_dis[t] / eta) * 0.25 / E_cap
        else:
            prob += soc[t] == soc[t-1] + (p_chg[t] * eta - p_dis[t] / eta) * 0.25 / E_cap
            
    # 求解
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 提取决策
    power_strategy = [pulp.value(p_dis[t]) - pulp.value(p_chg[t]) for t in range(T)]
    return power_strategy
```

---

## 二、 预测算法端优化：精准捕捉“电价尖峰”与“低谷”

电价预测的核心难点在于**捕捉极端的峰值（用户用电尖峰）和极低的谷值（午间光伏大发）**。普通的回归模型会因为“均值回归”倾向，将这些尖峰抹平。

### 1. 核心时序特征工程 (Lag & Rolling)
电网价格具有极强的周期惯性，昨天的电价、上周同期的电价是极强的锚定值。
*   **历史滞后特征 (Lag)**：
    *   `price_lag_96`：前 1 天同一时刻的电价。
    *   `price_lag_192`：前 2 天同一时刻的电价。
    *   `price_lag_672`：前 7 天同一时刻的电价。
*   **边界条件滚动特征 (Rolling)**：
    *   计算未来 2 小时、4 小时的滚动负荷均值和最大值。
    *   预测时刻的“净负荷预测”（`系统负荷预测值` - `风电预测值` - `光伏预测值`）。净负荷才是真正需要火电调节的部分，**净负荷越高，电价尖峰概率越大；净负荷为负，电价大概率为 0 甚至负值**。

### 2. 数值天气预报网格数据 (`.nc` 目录) 的高阶挖矿
`all_nc` 目录下的一年期数值天气预报包含三维时空气象网格。
*   **局限性分析**：训练集给出的 `风电预测值` 和 `光伏预测值` 往往是区域整体的粗糙估算，存在严重时滞和系统性误差。
*   **降维挖矿方案**：
    *   使用 `xarray` 和 `netCDF4` 库读取 `.nc` 文件。
    *   根据该区域风电场/光伏站的经纬度范围，提取关键的**辐射强度 (Radiation)**、** hub-height风速 (Wind Speed)**、**温度 (Temperature)**。
    *   利用这些气象特征直接建立辅助的“新能源实际出力预测模型”，或者将其作为强特征直接输入 LightGBM，能让模型提前感知“强冷空气导致的强风发电”或“大雾阴天导致的阳光骤降”，大幅提升电价波动预测的精度。

### 3. 非对称损失函数与分位数回归 (Quantile Regression)
*   **背景**：常规均方误差 (MSE) 对高估和低估的惩罚是对称的。但在储能决策中，如果模型**低估了电价波峰**，储能系统会认为价格不高而**放弃放电**，错失巨大的套利机会。
*   **提分方案**：
    *   引入**分位数回归 (Quantile Loss)**：让模型预测电价的 $90\%$ 分位数和 $10\%$ 分位数。
    *   在储能优化时，利用 $90\%$ 分位数感知电价上涨上限，提前做好放电准备；利用 $10\%$ 分位数感知电价底线，精准抄底充电。
    *   可以使用 LightGBM 的 `objective='quantile'`, `alpha=0.9` 直接训练。

### 4. 引入更先进的模型与融合 (Ensemble)
*   **CatBoost**：对于时间特征（小时、星期几等类别特征）拥有极强的处理能力，且抗过拟合能力优秀，极其适合本数据集。
*   **XGBoost**：对于局部的尖峰信号有较强的分裂敏感性。
*   **模型融合**：最终的预测电价输出可以采用 $0.4 \times \text{LightGBM} + 0.4 \times \text{CatBoost} + 0.2 \times \text{XGBoost}$ 的加权融合方式，以提供最稳定的泛化表现。

---

## 三、 提分工作落地计划 (时间线)

### 📅 第一周：攻克“储能套利策略”，实现从 0 到 1 的突破 (收益激增)
*   [ ] **建立评估器**：根据历史真实电价，写一个“历史套利计算器”脚本。输入任何一个 `power` 控制策略，能自动计算出在该电价下的理论净利润。
*   [ ] **开发线性规划模型**：使用 `PuLP` 实现上述的 LP 储能套利算法，在测试集上用当前的 LightGBM 预测价格跑出第一版有实际充放电操作的 `submit.csv`。
*   [ ] **结果校验**：检查生成的 `power` 充放电时机是否完全符合“电价低时充电（负功率）、电价高时放电（正功率）”的逻辑，确保物理约束（SOC 不超限）无误。

### 📅 第二周：时序特征工程与高阶特征注入 (预测精度提分)
*   [ ] **添加 Lag 特征**：在 `lgb_baseline.py` 中引入前 1 天、前 7 天的历史电价滞后列。
*   [ ] **构造“净负荷”特征**：计算 `系统负荷 - 风光总加` 的差值与比例，作为核心分裂特征。
*   [ ] **评估效果**：运行 `analyze_baseline.py`，对比新的 RMSE/MAE。确保验证集 RMSE 降至 `0.5` 以下。

### 📅 第三周：天气网格数据 (`.nc`) 挖矿与多模型融合 (冲刺高分)
*   [ ] **NC 特征工程**：编写脚本，使用 `xarray` 批量抽取 `.nc` 文件中关键经纬度的辐照度与风速，融合入训练集。
*   [ ] **模型融合**：加入 CatBoost 与 XGBoost 训练，使用简单平均或加权平均的形式生成最终电价。
*   [ ] **联合优化**：利用分位数预测或带置信度的随机规划，降低因预测电价偏差导致的储能错误充放电风险，锁定最终高分。
