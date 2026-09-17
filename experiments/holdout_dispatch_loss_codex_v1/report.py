"""Read saved evidence and draw the notebook figures. Never train or retune."""

from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .common import PROTOCOL_PATH, Ledger, read_json, write_json, sha256
from .run import run_directory


LABELS = {
    "lgb_baseline": "匹配 LightGBM baseline",
    "nb11_v2_transformer": "Notebook 11 / v2 Transformer",
    "nb11_v2_ridge": "Notebook 11 / v2 Ridge",
    "nb11_v1_transformer": "Notebook 11 / v1 Transformer（历史参考）",
    "nb11_v1_lstm": "Notebook 11 / v1 LSTM（历史参考）",
    "cursor_original_transformer": "Cursor 原结果 · Transformer",
    "cursor_original_lstm": "Cursor 原结果 · LSTM",
    "cursor_original_dlinear": "Cursor 原结果 · DLinear",
}
for _family in ["transformer", "lstm", "dlinear"]:
    _name = {"transformer": "Transformer", "lstm": "LSTM", "dlinear": "DLinear"}[_family]
    LABELS[f"codex_soft_regret_{_family}_realized"] = f"Codex 软遗憾 · {_name}"
    LABELS[f"codex_cursor_ce_{_family}_realized"] = f"同初始化 Cursor 损失 · {_name}"
    LABELS[f"codex_mse_{_family}_realized"] = f"MSE / 按调度分数选 · {_name}"
    LABELS[f"codex_mse_{_family}_rmse"] = f"MSE / 按 RMSE 选 · {_name}"

TEST_CAPTION = "59 天 holdout：2025-11-03—12-31；本轮参数未在这 59 天拟合；原始分数不是人民币。"
BLUE, ORANGE, RED, GRAY, GREEN = "#256b9b", "#d08324", "#b34849", "#818b96", "#387f6a"


def load_evidence():
    protocol = read_json(PROTOCOL_PATH)
    run_dir = run_directory(protocol)
    marker = read_json(run_dir / "evaluation_complete.json")
    for name in ["analysis", "comparison"]:
        suffix = ".json" if name == "analysis" else ".csv"
        if sha256(run_dir / (name + suffix)) != marker[name + "_sha256"]:
            raise RuntimeError(f"evaluated {name} changed; do not render silently")
    comparison = pd.read_csv(run_dir / "comparison.csv")
    daily = pd.read_csv(run_dir / "dispatch_daily.csv", parse_dates=["date"])
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    predictions["times"] = pd.to_datetime(predictions.times)
    return run_dir, protocol, read_json(run_dir / "analysis.json"), comparison, daily, predictions


def public_table(comparison):
    table = comparison.copy()
    table["方案"] = table.candidate.map(LABELS)
    table["角色"] = table.group.map({
        "baseline": "匹配基线", "nb11_v2": "Notebook 11 参考", "nb11_v1_context_only": "非匹配历史参考",
        "cursor_original": "Cursor 原结果", "codex_main": "Codex 预先声明的分支",
        "matched_cursor_control": "Cursor 损失对照（不是 Codex 新损失）",
        "mse_realized_control": "只改选模对照", "mse_rmse_control": "MSE 参照",
    })
    table.loc[table.preselected_primary.fillna(False).astype(bool), "角色"] = "测试前选定的 Codex 主方案"
    cols = ["方案", "角色", "realized_profit_mean", "difference_vs_cursor_best", "negative_days", "idle_days", "RMSE"]
    return table[cols].rename(columns={"realized_profit_mean": "59日日均原始调度分数", "difference_vs_cursor_best": "比Cursor原最佳多/少", "negative_days": "负分日", "idle_days": "不交易日", "RMSE": "点预测RMSE（辅助）"})


def control_table(analysis):
    rows = []
    for control in analysis["controls"]:
        rows.append({
            "模型": control["family"],
            "只改选模：按分数减按RMSE": control["selection_only"]["mean_difference"],
            "软遗憾减MSE（都按分数选）": control["soft_vs_mse_money"]["mean_difference"],
            "软遗憾减同初始化Cursor损失": control["soft_vs_matched_ce"]["mean_difference"],
        })
    return pd.DataFrame(rows)


def cost_table(run_dir):
    rows = [read_json_line(line) for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if line]
    complete = pd.DataFrame([r for r in rows if r["event"] == "training_end" and r.get("status") == "ok"])
    table = complete.groupby("method", sort=False).agg(
        launches=("job_id", "size"), worker_seconds=("seconds", "sum"), cpu_seconds=("cpu_seconds", "sum"),
        process_wall_seconds=("subprocess_wall_seconds", "sum"), optimizer_steps=("optimizer_steps", "sum"),
    ).reset_index()
    return table.rename(columns={"method": "方法", "launches": "实际训练启动", "worker_seconds": "工作代码墙钟秒", "cpu_seconds": "CPU秒", "process_wall_seconds": "含启动开销墙钟秒", "optimizer_steps": "优化器步数"})


def read_json_line(line):
    import json
    return json.loads(line)


def style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 11, "axes.titlesize": 13,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def save(fig, figures, name):
    path = figures / (name + ".png")
    fig.savefig(path, dpi=155, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_overview(figures, analysis, comparison):
    ids = ["lgb_baseline", "nb11_v2_transformer", "cursor_original_transformer", "cursor_original_lstm", "cursor_original_dlinear", "codex_soft_regret_transformer_realized", "codex_soft_regret_lstm_realized", "codex_soft_regret_dlinear_realized", "codex_cursor_ce_transformer_realized"]
    data = comparison.set_index("candidate").loc[ids]
    labels = [LABELS[name] for name in ids]
    labels[7] += "  ← 测试前主方案"
    labels[8] += "  ← 对照，不是新方法"
    colors = [GRAY, GRAY, ORANGE, ORANGE, ORANGE, BLUE, BLUE, RED, GREEN]
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.barh(np.arange(len(ids)), data.realized_profit_mean, color=colors, height=.65)
    ax.set_yticks(np.arange(len(ids)), labels)
    ax.invert_yaxis()
    for i, value in enumerate(data.realized_profit_mean):
        ax.text(value + 55, i, f"{value:,.2f}", va="center", fontsize=11)
    ax.axvline(analysis["cursor_original_best_score"], color=ORANGE, ls="--", lw=1, label="Cursor 原最佳")
    ax.set_xlim(0, float(data.realized_profit_mean.max()) * 1.18)
    ax.set_xlabel("日均原始调度分数：sum(真实价格 × 锁定动作)，越高越好")
    ax.set_ylabel("冻结方案")
    ax.set_title("没有全面赢：LSTM 分支改善，但测试前选出的 DLinear 主方案失败", loc="left", pad=18)
    ax.grid(axis="x", alpha=.16)
    ax.set_axisbelow(True)
    fig.text(.02, -.015, TEST_CAPTION + " 来源：comparison.csv；各分支全部保留。", fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "01_holdout_scores")


def figure_controls(figures, comparison):
    families = ["transformer", "lstm", "dlinear"]
    methods = [("mse", "rmse", "点 MSE，按 RMSE 选", GRAY), ("mse", "realized", "点 MSE，按调度分数选", BLUE), ("cursor_ce", "realized", "Cursor 硬窗分类，同初始化", ORANGE), ("soft_regret", "realized", "Codex 软遗憾＋不交易项", GREEN)]
    indexed = comparison.set_index("candidate")
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5), sharey=True)
    for ax, family in zip(axes, families):
        for j, (method, policy, label, color) in enumerate(methods):
            value = indexed.loc[f"codex_{method}_{family}_{policy}", "realized_profit_mean"]
            ax.bar(j, value, color=color, width=.66, label=label)
            ax.text(j, value + 65, f"{value:.1f}", ha="center", fontsize=10)
        ax.set_xticks(range(4), ["MSE\n误差选", "MSE\n分数选", "硬窗\n分类", "软\n遗憾"])
        ax.set_title(family.upper() if family == "lstm" else family.capitalize())
        ax.set_xlabel("训练目标 / 选模方式")
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("59 日日均原始调度分数，越高越好")
    axes[0].set_ylim(0, 7500)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.07), frameon=False)
    fig.text(.02, -.02, TEST_CAPTION + " 同一模型内初始化相同；MSE 无三点损失参数搜索。", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94))
    return save(fig, figures, "02_selection_and_loss_controls")


def figure_selection(figures, comparison):
    main = comparison.loc[comparison.group == "codex_main"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6))
    for row in main.itertuples():
        color = RED if row.family == "dlinear" else (BLUE if row.family == "lstm" else GRAY)
        for ax, field in zip(axes, ["validation_realized", "realized_profit_mean"]):
            value = getattr(row, field)
            ax.scatter(row.family, value, color=color, s=80, zorder=3)
            ax.annotate(f"{value:,.2f}", (row.family, value), xytext=(0, 12), textcoords="offset points", ha="center")
    axes[0].set_title("先选：30 天开发验证（10-04—11-02）")
    axes[1].set_title("再考：59 天 holdout（11-03—12-31）")
    for ax in axes:
        ax.set_ylabel("该时期日均原始调度分数")
        ax.set_xlabel("Codex 预先声明的三个模型分支")
        ax.set_ylim(4600, 9000)
        ax.margins(x=.2)
        ax.grid(axis="y", alpha=.18)
    fig.suptitle("开发期第一名，不等于测试期第一名：主方案没有事后换成 LSTM", fontsize=13, y=1.05)
    fig.text(.02, -.04, "两段日期难度不同，不能把两列的绝对下降全归为过拟合；能确定的是模型排序反转。测试期不选参数。", fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "03_validation_selection_reversal")


def figure_daily(figures, analysis, daily):
    wide = daily.pivot(index="date", columns="candidate", values="realized_profit").sort_index()
    ref = wide[analysis["cursor_original_best_candidate"]]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    names = [(analysis["primary_candidate"], "预选主方案 DLinear"), (analysis["observed_best_codex_candidate"], "LSTM 分支（并非预选主方案）")]
    for row_i, (candidate, title) in enumerate(names):
        delta = wide[candidate] - ref
        axes[row_i, 0].bar(delta.index, delta, color=np.where(delta >= 0, BLUE, RED), width=.8)
        axes[row_i, 1].plot(delta.index, delta.cumsum(), color=BLUE, lw=1.7)
        axes[row_i, 0].set_title(f"{title}：每天比 Cursor LSTM 多 / 少")
        axes[row_i, 1].set_title(f"累计差：终点 {delta.sum():+,.0f}")
        axes[row_i, 0].set_ylabel("单日原始分数差")
        axes[row_i, 1].set_ylabel("累计原始分数差")
        for ax in axes[row_i]:
            ax.axhline(0, color=GRAY, lw=1)
            ax.axvline(pd.Timestamp("2025-12-03"), color=GRAY, ls=":", lw=1)
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            ax.set_xlabel("日期（虚线分开前 30 天 / 后 29 天）")
            ax.grid(alpha=.12)
    fig.text(.015, -.02, TEST_CAPTION + " 正负结果全部展示；来源：dispatch_daily.csv。", fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "04_daily_differences")


def case_records(analysis, daily, predictions):
    cases = [
        (analysis["primary_vs_cursor_daily"]["worst_day"], analysis["primary_candidate"], "主方案相对 Cursor LSTM 损失最大的一天"),
        (next(c for c in analysis["controls"] if c["family"] == "lstm")["soft_vs_cursor_original"]["best_day"], "codex_soft_regret_lstm_realized", "LSTM 分支相对 Cursor LSTM 改善最大的一天"),
    ]
    rows = []
    for date, candidate, reason in cases:
        point = predictions.loc[predictions.times.dt.normalize() == pd.Timestamp(date)]
        for name in [candidate, "cursor_original_lstm"]:
            row = daily.loc[(daily.date == pd.Timestamp(date)) & (daily.candidate == name)].iloc[0]
            c, d = int(row.tc), int(row.td)
            actual = point.A.to_numpy()
            rows.append({"date": date, "candidate": name, "selection_reason": reason,
                         "charge_start": f"{c // 4:02d}:{(c % 4) * 15:02d}", "discharge_start": f"{d // 4:02d}:{(d % 4) * 15:02d}",
                         "actual_charge_mean": float(actual[c:c+8].mean()), "actual_discharge_mean": float(actual[d:d+8].mean()),
                         "realized_score": float(row.realized_profit), "predicted_score": float(row.predicted_profit), "oracle_score": float(row.oracle_profit)})
    return rows


def figure_cases(figures, analysis, daily, predictions, cases):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.6), sharex=True)
    for i in range(2):
        pair = cases[2*i:2*i+2]
        point = predictions.loc[predictions.times.dt.normalize() == pd.Timestamp(pair[0]["date"])]
        t = np.arange(96) / 4
        all_values = [point.A.to_numpy()] + [point[r["candidate"]].to_numpy() for r in pair]
        lo, hi = min(v.min() for v in all_values), max(v.max() for v in all_values)
        span = max(hi-lo, .1)
        for j, record in enumerate(pair):
            ax = axes[i, j]
            row = daily.loc[(daily.date == pd.Timestamp(record["date"])) & (daily.candidate == record["candidate"])].iloc[0]
            c, d = int(row.tc), int(row.td)
            ax.plot(t, point.A, color="#252a30", lw=1.4, label="真实价格（只用于评分）")
            ax.plot(t, point[record["candidate"]], color=BLUE, lw=1.2, ls="--", label="冻结模型的预测")
            ax.axvspan(c/4, (c+8)/4, color=BLUE, alpha=.16, label="按预测选的充电窗口")
            ax.axvspan(d/4, (d+8)/4, color=ORANGE, alpha=.2, label="按预测选的放电窗口")
            ax.set_title(f"{record['date']} · {LABELS[record['candidate']]}\n实际分数 {record['realized_score']:+,.0f}；预测自评分 {record['predicted_score']:+,.0f}", fontsize=11)
            ax.set_ylim(lo-.1*span, hi+.1*span)
            ax.set_ylabel("价格 A（数据原尺度）")
            ax.set_xlabel("北京时间（小时）")
            ax.set_xticks(np.arange(0, 25, 4))
            ax.grid(alpha=.13)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.075), frameon=False)
    fig.text(.015, -.02, "按事后差值最大 / 最小选案例，解释机制而非代表平均效果；未用真实价格平移预测曲线。" + TEST_CAPTION, fontsize=9)
    fig.tight_layout()
    return save(fig, figures, "05_actual_decision_examples")


def build():
    started = time.perf_counter()
    run_dir, protocol, analysis, comparison, daily, predictions = load_evidence()
    style()
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    cases = case_records(analysis, daily, predictions)
    outputs = [figure_overview(figures, analysis, comparison), figure_controls(figures, comparison),
               figure_selection(figures, comparison), figure_daily(figures, analysis, daily),
               figure_cases(figures, analysis, daily, predictions, cases)]
    public_table(comparison).to_csv(run_dir / "comparison_zh.csv", index=False, encoding="utf-8-sig")
    control_table(analysis).to_csv(run_dir / "controls_zh.csv", index=False, encoding="utf-8-sig")
    cost_table(run_dir).to_csv(run_dir / "costs_zh.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cases).to_csv(run_dir / "case_details.csv", index=False)
    write_json(run_dir / "figure_manifest.json", {
        "sources": ["comparison.csv", "dispatch_daily.csv", "predictions.parquet", "analysis.json", "ledger.jsonl"],
        "figures": [{"path": str(p.relative_to(run_dir)), "sha256": sha256(p)} for p in outputs],
        "generator": "experiments/holdout_dispatch_loss_codex_v1/report.py", "training_performed": False,
        "case_selection": "Primary worst paired day and LSTM branch best paired day versus Cursor original LSTM, selected after evaluation for illustration only.",
    })
    Ledger(run_dir, protocol).add("report_figures_built", seconds=time.perf_counter()-started, count=len(outputs), training=False)
    print("\n".join(str(p) for p in outputs))
    print(pd.DataFrame(cases).to_string(index=False))


if __name__ == "__main__":
    build()
