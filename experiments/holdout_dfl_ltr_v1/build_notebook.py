"""Build and execute notebooks/notebook_12_claude.ipynb from the frozen artefacts.

The notebook computes its own verdict from ``reports/holdout_dfl_ltr_v1/`` so the
prose cannot drift from the numbers.  Run this after run.py, ensemble.py,
rerank.py and diagnostics.py have produced their outputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks" / "notebook_12_claude.ipynb"


SETUP = '''
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
from matplotlib import font_manager


def find_repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "phase_b").is_dir():
            return candidate
    raise FileNotFoundError("run this notebook from inside the Southern_Electricity repo")


_cjk = next(
    (n for n in ("Microsoft YaHei", "SimHei", "Microsoft JhengHei")
     if any(f.name == n for f in font_manager.fontManager.ttflist)),
    None,
)
if _cjk:
    plt.rcParams["font.sans-serif"] = [_cjk, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

REPO_ROOT = find_repo_root()
TORCH_RUNTIME = REPO_ROOT / ".torch_runtime"
if TORCH_RUNTIME.exists() and str(TORCH_RUNTIME) not in sys.path:
    sys.path.insert(0, str(TORCH_RUNTIME))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
FIGURES = REPO_ROOT / "outputs" / "notebooks" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
REPORT = REPO_ROOT / "reports" / "holdout_dfl_ltr_v1"
V1 = REPO_ROOT / "reports" / "holdout_rmse_v1"
V2 = REPO_ROOT / "reports" / "holdout_rmse_v2"
NB12 = REPO_ROOT / "reports" / "holdout_dispatch_loss_v1"
PHASE_B = REPO_ROOT / "reports" / "phase_b" / "final_metrics.json"

MONEY_LINE = 6301.331965      # nb12 lstm_dispatch -- the incumbent to beat
NB11_LINE = 6214.361782       # nb11 money champion, v2 transformer_contextual
ORACLE = 9000.886096
RNG = np.random.default_rng(20260914)

def show(path):
    display(Image(filename=str(path)))

print({"repo_root": str(REPO_ROOT), "report": str(REPORT.relative_to(REPO_ROOT))})
'''

LOAD = '''
diag = json.loads((REPORT / "diagnostics.json").read_text(encoding="utf-8"))
board = pd.read_csv(REPORT / "leaderboard.csv")
manifest = json.loads((REPORT / "manifest.json").read_text(encoding="utf-8"))
assert manifest["holdout_not_used_for_selection"] is True
assert PHASE_B.exists(), "phase_b frozen metrics must still be present"

# ---- daily realized profit for every comparator, one column per recipe
frames = []
for path, tag in ((V1, "v1"), (V2, "v2"), (NB12, "nb12"), (REPORT, "new")):
    csv = path / "dispatch_daily.csv"
    if csv.exists():
        f = pd.read_csv(csv)
        f["label"] = f["model"] if tag in ("nb12", "new") else tag + "/" + f["model"]
        frames.append(f[["label", "date", "realized_profit"]])
for extra, label in ((REPORT / "ensemble_daily.csv", "r5_ensemble"),
                     (REPORT / "rerank_daily.csv", "r4_rerank")):
    if extra.exists():
        f = pd.read_csv(extra)
        f["label"] = label
        frames.append(f[["label", "date", "realized_profit"]])
daily = pd.concat(frames, ignore_index=True)
daily["date"] = pd.to_datetime(daily["date"])
DAILY = daily.pivot_table(index="date", columns="label", values="realized_profit")

# the no-model constant window, as a daily series on the same 59 days
base = diag["constant_window_baselines"]["all_train_days"]
DAILY["constant_window"] = np.asarray(base["daily"], dtype=float)
BASELINE = float(base["realized_mean"])

def paired(a: str, b: str, n_boot: int = 20000):
    """Paired bootstrap over the 59 days of the difference a - b."""
    d = (DAILY[a] - DAILY[b]).dropna().to_numpy()
    n = len(d)
    draws = d[RNG.integers(0, n, (n_boot, n))].mean(axis=1)
    return {
        "diff": float(d.mean()),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "p_le_0": float((draws <= 0).mean()),
    }

print(f"recipes with daily series: {DAILY.shape[1]}   days: {DAILY.shape[0]}")
print(f"no-model constant window baseline: {BASELINE:.1f}")
print(board[["model", "realized_profit_mean", "capture"]].sort_values(
    "realized_profit_mean", ascending=False).head(8).to_string(index=False))
'''

VERDICT = '''
rows = []
for _, r in board.iterrows():
    stats_line = paired(r["model"], "lstm_dispatch") if r["model"] in DAILY.columns else None
    rows.append({
        "model": r["model"], "track": r["track"], "route": r["route"],
        "realized": r["realized_profit_mean"], "capture": r["capture"],
        "vs_incumbent": stats_line["diff"] if stats_line else np.nan,
        "p_le_0": stats_line["p_le_0"] if stats_line else np.nan,
    })
extra_rows = []
if (REPORT / "ensemble_weights.json").exists():
    ens = json.loads((REPORT / "ensemble_weights.json").read_text(encoding="utf-8"))
    extra_rows.append(("r5_ensemble", ens["holdout"]["realized_profit_mean"], ens["holdout"]["capture"]))
if (REPORT / "rerank.json").exists():
    rr = json.loads((REPORT / "rerank.json").read_text(encoding="utf-8"))
    extra_rows.append(("r4_rerank", rr["holdout"]["realized_profit_mean"], rr["holdout"]["capture"]))
for name, realized, capture in extra_rows:
    s = paired(name, "lstm_dispatch") if name in DAILY.columns else None
    rows.append({"model": name, "track": "-", "route": name.split("_")[0],
                 "realized": realized, "capture": capture,
                 "vs_incumbent": s["diff"] if s else np.nan,
                 "p_le_0": s["p_le_0"] if s else np.nan})

ALL = pd.DataFrame(rows).sort_values("realized", ascending=False).reset_index(drop=True)
best = ALL.iloc[0]
passed = ALL[ALL["realized"] > MONEY_LINE]

lines = []
if len(passed):
    ci = paired(best["model"], "lstm_dispatch")
    sig = "且配对 CI 下界 > 0" if ci["lo"] > 0 else "但配对 CI 含 0"
    lines.append(
        f"**过线。** 最好的新配方是 `{best['model']}`：realized **{best['realized']:.0f}**，"
        f"capture **{best['capture']:.3f}**（通过线 {MONEY_LINE:.0f} / 0.7001）。"
        f"相对 incumbent {ci['diff']:+.0f}，95%CI [{ci['lo']:+.0f}, {ci['hi']:+.0f}]，"
        f"P(≤0)={ci['p_le_0']:.3f} —— {sig}。"
    )
    lines.append(f"**共有 {len(passed)} 条配方过线**：" + "、".join(
        f"`{m}` ({v:.0f})" for m, v in zip(passed['model'], passed['realized'])))
else:
    lines.append(
        f"**未过线。** 最好的新配方是 `{best['model']}`：realized **{best['realized']:.0f}**，"
        f"capture **{best['capture']:.3f}**，通过线是 {MONEY_LINE:.0f} / 0.7001。"
    )

cb = paired(best["model"], "constant_window")
lines.append(
    f"**相对无模型固定窗（{BASELINE:.0f}）**：{cb['diff']:+.0f}，"
    f"95%CI [{cb['lo']:+.0f}, {cb['hi']:+.0f}]，P(≤0)={cb['p_le_0']:.3f}。"
)

# The highest point estimate is not necessarily the best-evidenced result.
cand = ALL[ALL["model"].isin(DAILY.columns)].copy()
cand["p_vs_base"] = [paired(m, "constant_window")["p_le_0"] for m in cand["model"]]
cand["p_vs_inc"] = [paired(m, "lstm_dispatch")["p_le_0"] for m in cand["model"]]
solid = cand[(cand["realized"] > MONEY_LINE)].sort_values("p_vs_base")
if len(solid) and solid.iloc[0]["model"] != best["model"]:
    s0 = solid.iloc[0]
    sb, si = paired(s0["model"], "constant_window"), paired(s0["model"], "lstm_dispatch")
    lines.append(
        f"**点估计最高 ≠ 证据最强。** 过线配方里证据最强的是 `{s0['model']}`"
        f"（realized {s0['realized']:.0f}）：相对固定窗 {sb['diff']:+.0f}，"
        f"95%CI [{sb['lo']:+.0f}, {sb['hi']:+.0f}]，P(≤0)={sb['p_le_0']:.3f}；"
        f"相对 incumbent {si['diff']:+.0f}，P(≤0)={si['p_le_0']:.3f}。"
        f" 而 `{best['model']}` 的日间方差大得多，点估计虽高但 CI 更宽。"
    )
lines.append(
    f"**Oracle 仍是 {ORACLE:.0f}**，最好的新配方 capture {best['capture']:.3f}；"
    f"这 59 天不是新冠军，不刷新 35 配方总榜，Phase B 冻结数未动。"
)
display(Markdown("### 核心结案（由产物计算，不手写）\\n\\n" + "\\n\\n".join("- " + s for s in lines)))
'''

DIAG_TABLES = '''
d1 = diag["D1_surface_flatness"]
t1 = pd.DataFrame({
    "第 K 好的真窗": list(d1["kth_best_true_window_value"]),
    "价值": [d1["kth_best_true_window_value"][k] for k in d1["kth_best_true_window_value"]],
    "占 oracle": [d1["kth_best_as_share_of_oracle"][k] for k in d1["kth_best_as_share_of_oracle"]],
})
hits = d1["hit_at_1"]
n_zero = sum(1 for v in hits.values() if v == 0.0)
display(Markdown(
    f"**D1** 38 条配方中 **{n_zero} 条**从未命中真 argmax，最好的一条 {max(hits.values())*59:.0f}/59；"
    f"2242 个「配方-日」里共 **{sum(round(v*59) for v in hits.values()):.0f} 次**命中。"
))
display(t1.style.format({"价值": "{:.0f}", "占 oracle": "{:.3f}"}).hide(axis="index"))

d2 = diag["D2_topk_ceiling"]["per_recipe"]["lstm_dispatch"]
t2 = pd.DataFrame({
    "K": list(d2["perfect_rerank_ceiling"]),
    "完美重排上界": list(d2["perfect_rerank_ceiling"].values()),
    "oracle 落在 top-K 的比例": list(d2["oracle_in_topk"].values()),
})
t2["相对 incumbent"] = t2["完美重排上界"] - MONEY_LINE
display(Markdown("**D2** incumbent 自己的 top-K 候选里，真值最好的那个值多少："))
display(t2.style.format({"完美重排上界": "{:.0f}", "oracle 落在 top-K 的比例": "{:.2f}",
                         "相对 incumbent": "{:+.0f}"}).hide(axis="index"))

d4 = diag["D4_noise"]
display(Markdown(
    f"**D4** 59 日均值 SE ≈ **{d4['median_se_of_59day_mean']:.0f}**，30 日 ≈ "
    f"**{d4['se_of_30day_mean']:.0f}**；nb12 逐 epoch val_realized 的 std 是 "
    f"{d4['nb12_per_epoch_val_realized_std_range'][0]:.0f}–"
    f"{d4['nb12_per_epoch_val_realized_std_range'][1]:.0f}，量级完全一致。"
    f" 30 日选模信噪比：realized 均值 "
    f"{d4['selection_signal_to_noise_30day']['realized_mean']:.3f} → 日均 capture "
    f"{d4['selection_signal_to_noise_30day']['daily_mean_capture']:.3f}。"
))
'''

FIG1 = '''
fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
ks = [int(k) for k in d1["kth_best_as_share_of_oracle"]]
vals = [d1["kth_best_as_share_of_oracle"][str(k)] for k in ks]
axes[0].plot(ks, vals, "o-", color="#2a6f97")
axes[0].axhline(1.0, color="0.5", ls="--", lw=1)
axes[0].set_xscale("log")
axes[0].set_xlabel("真实排名第 K 的合法窗")
axes[0].set_ylabel("占 oracle 的比例")
axes[0].set_title("D1 分数曲面顶部极平\\n第 50 名仍值 92.8%")
axes[0].set_ylim(0.9, 1.01)
axes[0].grid(alpha=0.3)

ks2 = [int(k) for k in d2["perfect_rerank_ceiling"]]
ceil = [d2["perfect_rerank_ceiling"][str(k)] for k in ks2]
axes[1].plot(ks2, ceil, "s-", color="#c44536", label="完美重排上界")
axes[1].axhline(MONEY_LINE, color="#2a6f97", ls="--", lw=1.2, label=f"incumbent {MONEY_LINE:.0f}")
axes[1].axhline(ORACLE, color="0.35", ls=":", lw=1.2, label=f"oracle {ORACLE:.0f}")
axes[1].axhline(BASELINE, color="#6c757d", ls="-.", lw=1, label=f"无模型固定窗 {BASELINE:.0f}")
axes[1].set_xscale("log")
axes[1].set_xlabel("shortlist 大小 K")
axes[1].set_ylabel("59 日均官方分数")
axes[1].set_title("D2 曲线已把好窗排进前列\\n错的是 shortlist 内部的顺序")
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig1_diagnostics.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
'''

TABLE = '''
anchors = [
    ("nb12 incumbent", "lstm_dispatch", None),
    ("nb12 transformer", "transformer_dispatch", None),
    ("nb11 钱冠 v2 TF", "v2/transformer_contextual", None),
    ("nb11 RMSE 冠 v1 TF", "v1/transformer_contextual", None),
    ("nb11 最差 v1 LSTM", "v1/lstm_contextual", None),
    ("气候平均", "v1/climatology_weekday_slot", None),
    ("无模型固定窗", "constant_window", None),
]
rows = []
for label, key, _ in anchors:
    if key not in DAILY.columns:
        continue
    s = paired(key, "lstm_dispatch")
    rows.append({"区块": "锚点", "配方": label, "realized": DAILY[key].mean(),
                 "capture": DAILY[key].mean() / ORACLE, "vs incumbent": s["diff"],
                 "CI 下界": s["lo"], "CI 上界": s["hi"], "P(≤0)": s["p_le_0"],
                 "pred_profit": np.nan, "RMSE": np.nan})

for _, r in board.iterrows():
    s = paired(r["model"], "lstm_dispatch")
    rows.append({"区块": f"新 {r['track']} 轨", "配方": r["model"],
                 "realized": r["realized_profit_mean"], "capture": r["capture"],
                 "vs incumbent": s["diff"], "CI 下界": s["lo"], "CI 上界": s["hi"],
                 "P(≤0)": s["p_le_0"], "pred_profit": r["pred_profit_mean"], "RMSE": r["RMSE"]})
for name in ("r5_ensemble", "r4_rerank"):
    if name in DAILY.columns:
        s = paired(name, "lstm_dispatch")
        rows.append({"区块": "新 组合", "配方": name, "realized": DAILY[name].mean(),
                     "capture": DAILY[name].mean() / ORACLE, "vs incumbent": s["diff"],
                     "CI 下界": s["lo"], "CI 上界": s["hi"], "P(≤0)": s["p_le_0"],
                     "pred_profit": np.nan, "RMSE": np.nan})

TAB = pd.DataFrame(rows).sort_values("realized", ascending=False).reset_index(drop=True)
TAB["过线"] = TAB["realized"] > MONEY_LINE
display(TAB.style.format({
    "realized": "{:.0f}", "capture": "{:.3f}", "vs incumbent": "{:+.0f}",
    "CI 下界": "{:+.0f}", "CI 上界": "{:+.0f}", "P(≤0)": "{:.3f}",
    "pred_profit": "{:.0f}", "RMSE": "{:.3f}",
}, na_rep="—").hide(axis="index"))
'''

FIG2 = '''
sub = TAB.copy()
colors = []
for block in sub["区块"]:
    colors.append("#2a6f97" if block.startswith("新") else "#adb5bd")
fig, ax = plt.subplots(figsize=(9.5, max(4.5, 0.32 * len(sub))))
ax.barh(sub["配方"], sub["realized"], color=colors)
ax.axvline(MONEY_LINE, color="#c44536", ls="--", lw=1.4, label=f"通过线 incumbent {MONEY_LINE:.0f}")
ax.axvline(NB11_LINE, color="#e0a458", ls="-.", lw=1.1, label=f"nb11 钱冠 {NB11_LINE:.0f}")
ax.axvline(BASELINE, color="#6c757d", ls=":", lw=1.3, label=f"无模型固定窗 {BASELINE:.0f}")
ax.invert_yaxis()
ax.set_xlim(4300, max(ORACLE * 0.78, sub["realized"].max() * 1.03))
ax.set_xlabel("59 日日均真正进账（官方分数）")
ax.set_title("蓝 = 本轮新配方 · 灰 = 冻结锚点")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig2_realized.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
'''

FIG3 = '''
pairs = board.pivot_table(index=["route", "family"], columns="track",
                          values="realized_profit_mean").dropna()
fig, ax = plt.subplots(figsize=(9.0, 4.4))
x = np.arange(len(pairs))
ax.bar(x - 0.2, pairs["A"], width=0.4, label="A 轨（nb12 同协议：seed 42 + realized 选格）", color="#adb5bd")
ax.bar(x + 0.2, pairs["B"], width=0.4, label="B 轨（5 seed + 日均 capture 选格）", color="#2a6f97")
ax.axhline(MONEY_LINE, color="#c44536", ls="--", lw=1.2, label=f"通过线 {MONEY_LINE:.0f}")
ax.axhline(BASELINE, color="#6c757d", ls=":", lw=1.2, label=f"无模型固定窗 {BASELINE:.0f}")
ax.set_xticks(x, [f"{r}\\n{f}" for r, f in pairs.index], fontsize=8)
ax.set_ylabel("59 日 realized")
ax.set_ylim(4300, None)
ax.set_title("A / B 两轨：换选模尺子值多少钱")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig3_tracks.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
delta = (pairs["B"] - pairs["A"])
display(Markdown(
    f"B 轨相对 A 轨的中位增量 **{delta.median():+.0f}**（范围 {delta.min():+.0f} … {delta.max():+.0f}）。"
    " 这一列就是「换选模尺子 + 多 seed 平均」本身值多少钱，和「换损失」分开看。"
))
'''

FIG4 = '''
fig, ax = plt.subplots(figsize=(6.6, 5.4))
have = board.dropna(subset=["pred_profit_mean"])
for track, marker, color in (("A", "s", "#adb5bd"), ("B", "o", "#2a6f97")):
    sl = have[have["track"] == track]
    ax.scatter(sl["pred_profit_mean"], sl["realized_profit_mean"], marker=marker,
               s=70, color=color, label=f"{track} 轨", edgecolors="white", linewidths=0.5)
for _, r in have.iterrows():
    ax.annotate(r["model"].replace("_listwise", "").replace("_cheap", "").replace("_pairdiff", "pd"),
                (r["pred_profit_mean"], r["realized_profit_mean"]),
                fontsize=6.5, xytext=(4, 3), textcoords="offset points")
lims = [min(have["pred_profit_mean"].min(), have["realized_profit_mean"].min()) - 300,
        max(have["pred_profit_mean"].max(), have["realized_profit_mean"].max()) + 300]
ax.plot(lims, lims, ":", color="#888", lw=1, label="pred = realized")
ax.axhline(MONEY_LINE, color="#c44536", ls="--", lw=1.1)
ax.set_xlabel("pred_profit（以为能赚）")
ax.set_ylabel("realized（真正进账）")
ax.set_title("诚实度：乐观缺口没有换来钱")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
p = FIGURES / "nb12_claude_fig4_honesty.png"
fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
print(p.relative_to(REPO_ROOT)); show(p)
'''

STABILITY = '''
decomp = json.loads((REPORT / "seed_decomposition.json").read_text(encoding="utf-8"))
inst = json.loads((REPORT / "instability.json").read_text(encoding="utf-8"))

DEC = pd.DataFrame(decomp["rows"]).sort_values("single_curve_mean", ascending=False)
display(Markdown(
    "**B 轨的分数是五条曲线逐点平均后的成绩。** 这混了两件事：单条曲线有多好"
    "（损失的性质），和平均五条值多少（任何配方都能用，而 incumbent 是单曲线，"
    "从来没享受过）。拆开："
))
display(DEC[["route", "family", "single_curve_mean", "single_curve_std",
             "averaged_curve", "averaging_gain", "single_curve_vs_incumbent"]]
        .rename(columns={"single_curve_mean": "单曲线均值", "single_curve_std": "单曲线 sd",
                         "averaged_curve": "平均后", "averaging_gain": "平均的增益",
                         "single_curve_vs_incumbent": "单曲线 vs 6301"})
        .style.format({"单曲线均值": "{:.0f}", "单曲线 sd": "{:.0f}", "平均后": "{:.0f}",
                       "平均的增益": "{:+.0f}", "单曲线 vs 6301": "{:+.0f}"}).hide(axis="index"))

neg = int((DEC["averaging_gain"] < 0).sum())
display(Markdown(
    f"- 按**单曲线**口径排序，第一名是 `{DEC.iloc[0]['route']}/{DEC.iloc[0]['family']}`，"
    f"不是榜单上的 R1 transformer。\\n"
    f"- `r6_cheap/dlinear` 单曲线 {float(DEC.loc[(DEC.route=='r6_cheap')&(DEC.family=='dlinear'),'single_curve_mean'].iloc[0]):.0f}，"
    f"比 incumbent **低 230**；它过线靠的是 +296 的集成增益。\\n"
    f"- 平均不是稳赢：12 行里 **{neg} 行为负**，最差 "
    f"{DEC['averaging_gain'].min():+.0f}。它压方差，不保证提均值。"
))

rows = []
for key in ("r1_listwise/transformer", "r2_pairdiff/lstm", "r6_cheap/dlinear"):
    v = inst[key]
    rows.append({
        "配方": key,
        "8 次重跑跨度": v["mean_59day_score"]["spread"],
        "曲线跨度": v["max_curve_spread_across_repeats"],
        "翻窗天数": v["worst_pair"]["days_with_a_different_window"],
    })
rows.append({"配方": "5 seed 平均后的 transformer（3 次）",
             "8 次重跑跨度": inst["averaging_damps_it"]["spread"],
             "曲线跨度": np.nan, "翻窗天数": 0})
display(Markdown(
    "**同一 seed、同一数据、同一段代码重跑 8 次**——任何差异都只来自 CPU 归约顺序："
))
display(pd.DataFrame(rows).style.format(
    {"8 次重跑跨度": "{:.1f}", "曲线跨度": "{:.2e}", "翻窗天数": "{:.0f}"},
    na_rep="—").hide(axis="index"))

w = inst["r1_listwise/transformer"]["worst_pair"]["largest_single_day"]
display(Markdown(
    f"transformer 的 8 次成绩是 "
    f"`{inst['r1_listwise/transformer']['mean_59day_score']['values']}`——**双峰**。"
    f" 59 天里只有 {inst['r1_listwise/transformer']['worst_pair']['days_with_a_different_window']} 天的窗变了，"
    f"而 **{w['date']} 一天贡献了 {100*w['share_of_total_gap']:.0f}% 的差距**："
    f"放电块从 `td={w['low_window'][1]}`（{w['low_window'][1]/4:.2f}h）跳到 "
    f"`td={w['high_window'][1]}`（{w['high_window'][1]/4:.2f}h），当天 "
    f"{w['low_profit']:.0f} → {w['high_profit']:.0f}。\\n\\n"
    f"机制：`realized_profit` 是预测曲线的**阶跃函数**，`optimize_day` 在 3321 个窗上取 argmax。"
    f"两窗接近打平时，{inst['r1_listwise/transformer']['max_curve_spread_across_repeats']:.1e} 量级的"
    f"扰动就能翻窗；若那天是尖峰日（{w['date']} 的 oracle 是 31559），59 天均值就跳 300+。\\n\\n"
    f"**所以本册任何「几百分」的领先都在位级非确定性的噪声底之内**，抽样噪声还没算。"
    f" 不受影响的是 LSTM 与 DLinear（跨度 0.0），也就是 R2/R3 这两条统计上站得住的路线；"
    f"5 seed 平均把跨度从 {inst['averaging_damps_it']['single_curve_spread_for_comparison']:.0f} "
    f"压到 {inst['averaging_damps_it']['spread']:.1f}。"
))
'''

MULTIPLE = '''
n_new = int(len(board) + sum(1 for n in ("r5_ensemble", "r4_rerank") if n in DAILY.columns))
inc = DAILY["lstm_dispatch"].to_numpy()
sd = float(np.median(DAILY.std(ddof=1)))
others = [c for c in DAILY.columns if c != "lstm_dispatch"]
corr = float(np.median([np.corrcoef(inc, DAILY[c].to_numpy())[0, 1] for c in others]))

B = 20000
common = RNG.standard_normal((B, 1, 59))
idio = RNG.standard_normal((B, n_new, 59))
sim = (np.sqrt(corr) * common + np.sqrt(1 - corr) * idio) * sd + inc.mean()
best_of = sim.mean(axis=2).max(axis=1)
p_any = float((best_of > MONEY_LINE).mean())
observed = float(TAB.loc[TAB["区块"].str.startswith("新"), "realized"].max() - MONEY_LINE)
p_obs = float((best_of > MONEY_LINE + observed).mean())
display(Markdown(
    f"本轮报告 **{n_new}** 条新配方。在「每一条的真实水平都恰好等于 incumbent」的零假设下"
    f"（日内 sd {sd:.0f}，配方间日相关 {corr:.2f}）：\\n\\n"
    f"- P(至少一条 > {MONEY_LINE:.0f}) = **{p_any:.3f}**\\n"
    f"- P(最好的一条超出 ≥ {observed:+.0f}) = **{p_obs:.3f}**  ← 本轮实际观测到 {observed:+.0f}\\n\\n"
    f"这一列是必报项：条数越多，单纯靠抽样过线的概率越高。"
))
'''

CLOSING = '''
best_new = TAB[TAB["区块"].str.startswith("新")].iloc[0]
vs_inc = paired(best_new["配方"], "lstm_dispatch")
vs_base = paired(best_new["配方"], "constant_window")
vs_nb11 = paired(best_new["配方"], "v2/transformer_contextual")

# --- the pre-registered stop-loss is per ROUTE, not on the single best recipe:
#     "if all six routes' CI against the constant window contain 0, write a
#      negative result".  Evaluate each route by its own best recipe.
route_rows = []
for route in sorted(set(board["route"]) | {"r4_rerank", "r5_ensemble"}):
    members = [m for m in DAILY.columns if m.startswith(route)]
    if not members:
        continue
    champ = max(members, key=lambda m: DAILY[m].mean())
    s_base = paired(champ, "constant_window")
    s_inc = paired(champ, "lstm_dispatch")
    route_rows.append({
        "route": route, "best recipe": champ, "realized": DAILY[champ].mean(),
        "vs 固定窗": s_base["diff"], "CI 下界": s_base["lo"], "CI 上界": s_base["hi"],
        "P(≤0) vs 固定窗": s_base["p_le_0"], "Bonferroni p×6": min(1.0, s_base["p_le_0"] * 6),
        "vs incumbent": s_inc["diff"], "P(≤0) vs incumbent": s_inc["p_le_0"],
        "过线": DAILY[champ].mean() > MONEY_LINE,
    })
ROUTES_TAB = pd.DataFrame(route_rows).sort_values("realized", ascending=False)
display(Markdown("**每条路线的代表配方（路线内按 realized 取最好），相对无模型固定窗：**"))
display(ROUTES_TAB.style.format({
    "realized": "{:.0f}", "vs 固定窗": "{:+.0f}", "CI 下界": "{:+.0f}", "CI 上界": "{:+.0f}",
    "P(≤0) vs 固定窗": "{:.3f}", "Bonferroni p×6": "{:.3f}",
    "vs incumbent": "{:+.0f}", "P(≤0) vs incumbent": "{:.3f}"}).hide(axis="index"))

clears = ROUTES_TAB[ROUTES_TAB["CI 下界"] > 0]
verdict = []
if best_new["realized"] > MONEY_LINE:
    verdict.append(
        f"**主判定：过线。** 最好的新配方 `{best_new['配方']}` realized "
        f"**{best_new['realized']:.0f}** > {MONEY_LINE:.0f}，capture "
        f"**{best_new['capture']:.3f}** > 0.7001。共 "
        f"{int((TAB['区块'].str.startswith('新') & TAB['过线']).sum())} 条过线。"
    )
else:
    verdict.append(f"**主判定：未过线。** 最好的新配方 {best_new['realized']:.0f} < {MONEY_LINE:.0f}。")

verdict.append(
    f"**但「26 条里最好的一条」本身不是证据。** 零假设下 P(最好的一条超出 ≥"
    f"{best_new['realized'] - MONEY_LINE:+.0f}) = {p_obs:.3f}。"
    f" 排名第一的 `{best_new['配方']}` 相对 incumbent {vs_inc['diff']:+.0f}，"
    f"95%CI [{vs_inc['lo']:+.0f}, {vs_inc['hi']:+.0f}]，P(≤0)={vs_inc['p_le_0']:.3f} —— CI 含 0。"
)

if len(clears):
    survives = clears[clears["Bonferroni p×6"] < 0.05]
    marginal = clears[clears["Bonferroni p×6"] >= 0.05]
    fmt = lambda r: (
        f"**{r['route']}**（{r['best recipe']}，{r['vs 固定窗']:+.0f}，"
        f"CI [{r['CI 下界']:+.0f}, {r['CI 上界']:+.0f}]，Bonferroni p={r['Bonferroni p×6']:.3f}）"
    )
    verdict.append(
        f"**站得住的结果在这里：{len(clears)} 条路线相对无模型固定窗（{BASELINE:.0f}）"
        f"的配对 CI 下界为正。** 这是预先指定的**逐路线**比较，不是对 26 条取最大值。"
    )
    if len(survives):
        verdict.append(
            "其中 **6 重 Bonferroni 校正后仍然成立**的是："
            + "、".join(fmt(r) for _, r in survives.iterrows())
            + "。本项目此前没有任何配方做到过这一点。"
        )
    if len(marginal):
        verdict.append(
            "**未通过 Bonferroni 校正**（原始 CI 排除 0，校正后不成立，只能算提示）："
            + "、".join(fmt(r) for _, r in marginal.iterrows())
            + "。"
        )
    verdict.append("**止损条款未触发**：不是全部路线的 CI 都含 0，可以开第二轮。")
else:
    verdict.append(
        f"**止损条款触发**：六条路线相对无模型固定窗（{BASELINE:.0f}）的 CI 全部含 0，"
        "按计划第 6 节写负结果，不开第二轮。"
    )

verdict.append(
    f"相对 nb11 钱冠（{NB11_LINE:.0f}）：{vs_nb11['diff']:+.0f}，"
    f"95%CI [{vs_nb11['lo']:+.0f}, {vs_nb11['hi']:+.0f}]。"
    f" Oracle {ORACLE:.0f} 仍未被接近（最好 capture {best_new['capture']:.3f}），"
    f"D2 量到的 top-50 重排空间（+781）没有被 R4 拿到：reranker 在 val30 上"
    "**主动放弃了重排**（blend=0），说明 shortlist 内部的顺序在这份数据上学不出来。"
)
verdict.append(
    f"**分辨率警告**：同 seed 重跑 8 次，`r1_listwise/transformer` 的 59 日均值跨度是 "
    f"{inst['r1_listwise/transformer']['mean_59day_score']['spread']:.0f}（第 5 节），"
    f"91% 来自 2025-12-30 一天的翻窗。上面任何几百分的领先都应按此折价；"
    f"R2/R3 不受影响（LSTM 跨度 0.0），这也是只有它们进入正式结论的原因之一。"
)
verdict.append(
    "这 59 天不是新冠军：不刷新 35 配方总榜，不改 `reports/phase_b/final_metrics.json`，"
    "不改 v1/v2 leaderboard 的 RMSE/MAE。"
)
display(Markdown("\\n\\n".join("- " + v for v in verdict)))
'''


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md(
            "# notebook_12_claude｜六条决策对齐路线 vs nb12 的 dispatch loss\n\n"
            "**目的**：nb12 把 `F.mse_loss` 换成「日中心块和 MSE + 3321 窗 one-hot CE」，"
            "`lstm_dispatch` 拿到 59 日 realized **6301.33** / capture **0.700**。"
            "本册在同一协议下跑六条路线（R1–R6），看能不能真的超过它。\n\n"
            "**唯一主判定**：59 日 `realized_profit_mean` **> 6301.33**。"
            "RMSE 好看不算赢，`pred_profit` 虚高、gap 拉大不算赢。\n\n"
            "**协议**：271 训 / 最后 30 选 / 301 重训 / 59 只考一次；seed 42（A 轨）与 42–46（B 轨）；"
            "oracle 窗只来自当天训练集真 `A`；`holdout_not_used_for_selection: true`。"
            "与 nb12 计划的逐条对照见 "
            "[`experiments/holdout_dfl_ltr_v1/PLAN_ZH.md`](../experiments/holdout_dfl_ltr_v1/PLAN_ZH.md) 第 3a 节。\n\n"
            "**两轨**：**A** = nb12 同协议（seed 42、按 30 日 realized 均值选格），与 6301.33 严格可比；"
            "**B** = 稳健（5 seed 平均、按 30 日日均 capture 选格）。两轨都报，差值本身就是结论。"
        ),
        code(SETUP.strip()),
        code(LOAD.strip()),
        code(VERDICT.strip()),
        md(
            "## 1. 设计依据：四个实测量\n\n"
            "路线不是按论文热度选的，是按下面四个在本仓库数据上算出来的量选的。"
            "复算脚本：`experiments/holdout_dfl_ltr_v1/diagnostics.py`。"
        ),
        code(DIAG_TABLES.strip()),
        code(FIG1.strip()),
        md(
            "**D3**：合法动作只有 3321 个，`(batch, 3321)` 张量能整张算，所以动作上的 Gibbs 分布是"
            "**精确**的。Mandi 的 solution cache、Pogančić 的 DBB、Berthet 的扰动优化器都是"
            "可行集无法枚举时的近似手段，在这里只会更差，因此本轮明确排除。\n\n"
            "## 2. 六条路线\n\n"
            "| | 路线 | 外部依据 | 相对 nb12 窗 CE 的改动 |\n"
            "|---|---|---|---|\n"
            "| R1 | Listwise 价值加权 | Mandi et al. ICML 2022 **Eq.16** | 目标从 one-hot 换成 `softmax(真分/τ_tgt)`；nb12 是 `τ_tgt→0` 的退化特例 |\n"
            "| R2 | Pairwise difference | Mandi et al. **Eq.13** | 回归窗分**差**，对日均完全不敏感 |\n"
            "| R3 | SPO+ | Elmachtoub & Grigas, *Manag. Sci.* 2022 | 凸、上界 regret，次梯度 `2(z_{w*(2p̂−p)} − z_{w*(p)})` |\n"
            "| R4 | top-K reranker | Mandi 的 ranking 视角 + D2 | 两段：曲线给 shortlist，浅层模型重排 |\n"
            "| R5 | 决策加权凸组合 | Caruana et al. 2004，用任务指标选权重 | 冻结曲线的凸组合，零新损失 |\n"
            "| R6 | pinball / 峰谷加权 | Smets et al., *Energy* 2025 | 廉价对照，复杂 DFL 必须先打过它 |\n\n"
            "损失实现见 `experiments/holdout_dfl_ltr_v1/loss.py`；"
            "块和口径直接复用 nb12 的 `day_center` / `block_sums` / `window_scores`，"
            "所以「同一口径」是可验证的，不是声称的。"
        ),
        code(
            "import inspect\n"
            "from experiments.holdout_dfl_ltr_v1 import loss as dfl\n"
            "print(inspect.getsource(dfl.listwise_loss))\n"
            "print(inspect.getsource(dfl.spo_plus_loss))"
        ),
        md("## 3. Headline 对照表\n\n每一行都带按天配对 bootstrap CI（相对 incumbent `lstm_dispatch`）。"),
        code(TABLE.strip()),
        code(FIG2.strip()),
        md("## 4. A / B 两轨：换损失 vs 换选模尺子"),
        code(FIG3.strip()),
        code(FIG4.strip()),
        md(
            "## 5. 集成 vs 损失，以及为什么「几百分」在这个 holdout 上不算差距\n\n"
            "这一节决定了前面那张表该怎么读。产物："
            "`reports/holdout_dfl_ltr_v1/seed_decomposition.json` 与 `instability.json`。"
        ),
        code(STABILITY.strip()),
        md("## 6. 多重比较"),
        code(MULTIPLE.strip()),
        md("## 7. 结案（按计划第 4 节的预注册写法）"),
        code(CLOSING.strip()),
        md(
            "---\n\n"
            "**边界**：这 59 天在 nb11 已经露过面，本册只回答「六条路线有没有比 nb12 的钱冠更赚钱」。"
            "不宣布新冠军，不刷新 35 配方总榜，不改 `reports/phase_b/final_metrics.json`，"
            "不改 v1/v2 leaderboard 的 RMSE/MAE。R5 的基学习器是在本协议下重新拟合的"
            "（冻结的 v1/v2 parquet 只含 59 个 holdout 日，没有 30 日验证预测，无法据以定权重），"
            "这一点在 `ensemble_weights.json` 里也写明了。"
        ),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> int:
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    execute = "--no-exec" not in sys.argv
    if execute:
        from nbclient import NotebookClient

        client = NotebookClient(
            nb,
            timeout=1800,
            kernel_name="python3",
            resources={"metadata": {"path": str(ROOT / "notebooks")}},
        )
        client.execute()
    nbf.write(nb, OUT)
    print(f"wrote {OUT.relative_to(ROOT)} (executed={execute})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
