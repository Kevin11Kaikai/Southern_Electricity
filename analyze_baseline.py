"""
Baseline Analysis: 格式检查 + 可视化
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互模式，避免弹窗卡住
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── 路径 ──────────────────────────────────────────
DATA_ROOT   = r'd:\others\Southern_Electricity\to_sais_new\to_sais_new'
OUTPUT_ROOT = r'd:\others\Southern_Electricity\outputs'
FIG_DIR     = os.path.join(OUTPUT_ROOT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

pred_path  = os.path.join(OUTPUT_ROOT, 'output_price', 'lgb_baseline_output.csv')
demo_path  = r'd:\others\Southern_Electricity\output_demo.csv'
train_feat = os.path.join(DATA_ROOT, 'train', 'mengxi_boundary_anon_filtered.csv')
train_lbl  = os.path.join(DATA_ROOT, 'train', 'mengxi_node_price_selected.csv')

# 图片样式
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'DejaVu Sans',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.facecolor': '#1e1e2e',
    'figure.facecolor': '#13131f',
    'text.color': 'white',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'axes.edgecolor': '#444',
    'grid.color': '#333',
})

# ══════════════════════════════════════════════════════
# 1. 格式检查
# ══════════════════════════════════════════════════════
print("=" * 60)
print("【1】格式一致性检查")
print("=" * 60)

df_pred = pd.read_csv(pred_path)
df_demo = pd.read_csv(demo_path)

print(f"lgb_baseline_output.csv  → shape={df_pred.shape}, cols={df_pred.columns.tolist()}")
print(f"output_demo.csv          → shape={df_demo.shape}, cols={df_demo.columns.tolist()}")

# 行数一致?
row_match = df_pred.shape[0] == df_demo.shape[0]
print(f"\n行数一致: {row_match}  ({df_pred.shape[0]} vs {df_demo.shape[0]})")

# 时间列对齐?
df_pred['times'] = pd.to_datetime(df_pred['times'])
df_demo['times'] = pd.to_datetime(df_demo['times'])
time_match = (df_pred['times'].values == df_demo['times'].values).all()
print(f"时间列完全对齐: {time_match}")

# 列名差异
demo_price_col = [c for c in df_demo.columns if c != 'times'][0]
print(f"\n输出列名对比:")
print(f"  lgb_baseline_output → '{df_pred.columns[1]}'")
print(f"  output_demo         → '{demo_price_col}' + 'power'")
print(f"\n⚠️  格式差异: output_demo 有 'power' 列（充放电功率），基线输出中没有。")
print(f"    另外列名不同: 基线输出用 'A'，demo 用 '{demo_price_col}'")
print(f"    建议提交时将列名改为 '{demo_price_col}' 并补充 power 列。")

# ══════════════════════════════════════════════════════
# 2. 重建验证集 (最后20%训练数据)
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【2】重建验证集用于评估可视化")
print("=" * 60)

df_feat  = pd.read_csv(train_feat, encoding='utf-8')
df_label = pd.read_csv(train_lbl,  encoding='utf-8')
df_train = pd.merge(df_feat, df_label, on='times', how='inner')
df_train['times'] = pd.to_datetime(df_train['times'])
df_train = df_train.sort_values('times').reset_index(drop=True)

feature_cols = ['系统负荷预测值', '风光总加预测值', '联络线预测值',
                '风电预测值', '光伏预测值', '水电预测值', '非市场化机组预测值']

def add_time_features(df):
    df = df.copy()
    df['hour']      = df['times'].dt.hour
    df['minute']    = df['times'].dt.minute
    df['dayofweek'] = df['times'].dt.dayofweek
    df['month']     = df['times'].dt.month
    return df

df_train = add_time_features(df_train)
split_idx = int(len(df_train) * 0.8)
df_val = df_train.iloc[split_idx:].copy()

# 用 lightgbm 重新训练预测验证集（快速版本）
import lightgbm as lgb

all_features = feature_cols + ['hour', 'minute', 'dayofweek', 'month']
X = df_train[all_features].values
y = df_train['A'].values
X_tr, X_val_arr = X[:split_idx], X[split_idx:]
y_tr, y_val_arr = y[:split_idx], y[split_idx:]

tr_set  = lgb.Dataset(X_tr,      label=y_tr,      feature_name=all_features)
val_set = lgb.Dataset(X_val_arr, label=y_val_arr, feature_name=all_features, reference=tr_set)

params = {'objective': 'regression', 'metric': 'rmse',
          'learning_rate': 0.05, 'num_leaves': 63, 'verbose': -1}

model = lgb.train(params, tr_set, num_boost_round=1000,
                  valid_sets=[val_set],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

y_val_pred = model.predict(X_val_arr)
y_true     = y_val_arr

rmse_total = np.sqrt(mean_squared_error(y_true, y_val_pred))
mae_total  = mean_absolute_error(y_true, y_val_pred)
print(f"\n验证集 RMSE: {rmse_total:.6f}  MAE: {mae_total:.6f}")

df_val = df_val.copy()
df_val['y_pred'] = y_val_pred
df_val['error']  = y_val_pred - df_val['A'].values

# ══════════════════════════════════════════════════════
# 3. 图1 — 预测值分布图（测试集预测 vs 验证集真实值）
# ══════════════════════════════════════════════════════
print("\n绘制图1: 预测值分布图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('LGB Baseline — Prediction Distribution', color='white', fontsize=14, fontweight='bold')

# 验证集: 真实 vs 预测
ax = axes[0]
ax.hist(y_true,     bins=80, alpha=0.7, color='#7eb8f7', label='Ground Truth (Val)')
ax.hist(y_val_pred, bins=80, alpha=0.7, color='#f97316', label='Predicted (Val)')
ax.set_title('Validation Set Distribution', color='white')
ax.set_xlabel('Node Price A')
ax.set_ylabel('Count')
ax.legend(facecolor='#2a2a3e', labelcolor='white')

# 测试集预测分布
ax2 = axes[1]
ax2.hist(df_pred['A'].values, bins=80, color='#a78bfa', alpha=0.85, label='Test Predictions')
ax2.set_title('Test Set Prediction Distribution', color='white')
ax2.set_xlabel('Node Price A')
ax2.set_ylabel('Count')
ax2.legend(facecolor='#2a2a3e', labelcolor='white')

plt.tight_layout()
p1 = os.path.join(FIG_DIR, 'fig1_prediction_distribution.png')
fig.savefig(p1, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  保存: {p1}")

# ══════════════════════════════════════════════════════
# 4. 图2 — 验证集真实值 vs 预测值时间曲线（取前2周）
# ══════════════════════════════════════════════════════
print("绘制图2: 真实值 vs 预测值曲线...")

# 取前 2016 个点 (15min * 2016 = 3周)
N = min(2016, len(df_val))
df_plot = df_val.iloc[:N].copy()

fig, ax = plt.subplots(figsize=(16, 5))
fig.suptitle('LGB Baseline — Validation Set: True vs Predicted (First 3 Weeks)',
             color='white', fontsize=13, fontweight='bold')

ax.plot(df_plot['times'], df_plot['A'],      color='#7eb8f7', lw=1.0, alpha=0.9, label='Ground Truth')
ax.plot(df_plot['times'], df_plot['y_pred'], color='#f97316', lw=1.0, alpha=0.85, label='Predicted', linestyle='--')
ax.set_xlabel('Time')
ax.set_ylabel('Node Price A')
ax.legend(facecolor='#2a2a3e', labelcolor='white')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30)

plt.tight_layout()
p2 = os.path.join(FIG_DIR, 'fig2_true_vs_predicted.png')
fig.savefig(p2, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  保存: {p2}")

# ══════════════════════════════════════════════════════
# 5. 图3 — 误差随时间变化（滚动均值）
# ══════════════════════════════════════════════════════
print("绘制图3: 误差随时间变化图...")

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
fig.suptitle('LGB Baseline — Error Analysis over Time', color='white', fontsize=13, fontweight='bold')

# 上图: 原始误差
ax1 = axes[0]
ax1.plot(df_val['times'], df_val['error'], color='#f472b6', lw=0.6, alpha=0.7, label='Error (pred - true)')
ax1.axhline(0, color='#fff', lw=0.8, linestyle='--', alpha=0.5)
ax1.fill_between(df_val['times'], df_val['error'], 0,
                 where=(df_val['error'] > 0), alpha=0.25, color='#f97316', label='Over-predict')
ax1.fill_between(df_val['times'], df_val['error'], 0,
                 where=(df_val['error'] < 0), alpha=0.25, color='#7eb8f7', label='Under-predict')
ax1.set_ylabel('Error')
ax1.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=8)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax1.xaxis.set_major_locator(mdates.MonthLocator())

# 下图: 滚动 RMSE (窗口96 = 1天)
window = 96
df_val['abs_err'] = df_val['error'].abs()
df_val['rolling_mae']  = df_val['abs_err'].rolling(window, min_periods=1).mean()
df_val['rolling_rmse'] = (df_val['error'] ** 2).rolling(window, min_periods=1).mean().apply(np.sqrt)

ax2 = axes[1]
ax2.plot(df_val['times'], df_val['rolling_mae'],  color='#34d399', lw=1.2, label='Rolling MAE (1-day)')
ax2.plot(df_val['times'], df_val['rolling_rmse'], color='#f97316', lw=1.2, label='Rolling RMSE (1-day)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Error Metric')
ax2.legend(facecolor='#2a2a3e', labelcolor='white')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_major_locator(mdates.MonthLocator())

plt.tight_layout()
p3 = os.path.join(FIG_DIR, 'fig3_error_over_time.png')
fig.savefig(p3, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  保存: {p3}")

# ══════════════════════════════════════════════════════
# 6. 图4 — 按小时统计 MAE / RMSE
# ══════════════════════════════════════════════════════
print("绘制图4: 按小时统计 MAE/RMSE...")

hourly = df_val.groupby('hour').apply(
    lambda g: pd.Series({
        'MAE':  mean_absolute_error(g['A'], g['y_pred']),
        'RMSE': np.sqrt(mean_squared_error(g['A'], g['y_pred'])),
        'count': len(g)
    })
).reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle('LGB Baseline — MAE & RMSE by Hour of Day', color='white', fontsize=13, fontweight='bold')

x = np.arange(24)
w = 0.38
bars1 = ax.bar(x - w/2, hourly['MAE'],  width=w, color='#7eb8f7', alpha=0.85, label='MAE')
bars2 = ax.bar(x + w/2, hourly['RMSE'], width=w, color='#f97316', alpha=0.85, label='RMSE')

ax.set_xticks(x)
ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, fontsize=8)
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Error Metric')
ax.legend(facecolor='#2a2a3e', labelcolor='white')

# 标注全局均值线
ax.axhline(mae_total,  color='#7eb8f7', lw=1.2, linestyle='--', alpha=0.6, label=f'Avg MAE={mae_total:.3f}')
ax.axhline(rmse_total, color='#f97316', lw=1.2, linestyle='--', alpha=0.6, label=f'Avg RMSE={rmse_total:.3f}')
ax.legend(facecolor='#2a2a3e', labelcolor='white', fontsize=9)

plt.tight_layout()
p4 = os.path.join(FIG_DIR, 'fig4_hourly_mae_rmse.png')
fig.savefig(p4, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"  保存: {p4}")

# ══════════════════════════════════════════════════════
# 7. 格式对齐后的输出
# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【3】生成格式对齐版本 (对齐 output_demo 格式)")
print("=" * 60)

df_submit = pd.DataFrame({
    'times':  df_pred['times'],
    '实时价格': df_pred['A'],
    'power':  0.0,
})
submit_path = os.path.join(OUTPUT_ROOT, 'output_price', 'lgb_baseline_submit.csv')
df_submit.to_csv(submit_path, index=False, encoding='utf-8-sig')
print(f"已保存格式对齐文件: {submit_path}")
print(f"列名: {df_submit.columns.tolist()}, shape={df_submit.shape}")

# ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("【全部完成】figures 目录内容:")
for f in sorted(os.listdir(FIG_DIR)):
    fpath = os.path.join(FIG_DIR, f)
    print(f"  {f}  ({os.path.getsize(fpath)/1024:.0f} KB)")
print("=" * 60)
