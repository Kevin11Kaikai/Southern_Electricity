"""Draw English vector figures from packaged CSV/JSON records; no model fitting.

Run from either Overleaf package: python scripts/build_figures.py
The method figure is a conceptual diagram; every quantitative figure reads data/.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter, MaxNLocator
import matplotlib.dates as mdates

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent if HERE.name == "scripts" else HERE
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=DEFAULT_ROOT / "data")
parser.add_argument("--out", type=Path, default=DEFAULT_ROOT / "figures")
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
data = args.data
M = pd.read_csv(data / "model_metrics.csv")
C = pd.read_csv(data / "correlations.csv")
P = pd.read_csv(data / "paired_daily.csv")
B = pd.read_csv(data / "ensemble_decomposition.csv")
I = pd.read_csv(data / "paired_intervals.csv")
S = json.loads((data / "summary.json").read_text(encoding="utf-8"))

NAVY, TEAL, ORANGE, GRAY, LIGHT = "#253B53", "#007D7A", "#C56728", "#788999", "#EDF3F6"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#9AA7B1", "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": NAVY, "ytick.color": NAVY, "grid.color": "#DFE6EB",
    "grid.linewidth": .6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white"})

def save(fig, name):
    fig.savefig(args.out / (name + ".pdf"), bbox_inches="tight", pad_inches=.07,
                metadata={"Creator": "Python / Matplotlib; build_figures.py", "CreationDate": None})
    fig.savefig(args.out / (name + ".png"), dpi=190, bbox_inches="tight", pad_inches=.07)
    plt.close(fig)

def base(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.8)

def method():
    fig, ax = plt.subplots(figsize=(8.2, 5.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    def box(x,y,w,h,label,fc=LIGHT,ec="#C7D4DE",bold=False,fs=9):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.025,rounding_size=0.08",fc=fc,ec=ec,lw=.8))
        ax.text(x+w/2,y+h/2,label,ha="center",va="center",fontsize=fs,fontweight="bold" if bold else "normal")
    def arrow(a,b,color=GRAY,dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=11,color=color,lw=1.05,
                                   linestyle="--" if dashed else "-", connectionstyle="arc3"))
    ax.text(.05,6.73,"A  TRAINING: learn relative window value",fontweight="bold",fontsize=11)
    box(.05,5.36,1.27,.73,"Available\nfeatures X")
    box(1.72,5.36,1.4,.73,"Predictor\n96 prices q",bold=True)
    box(3.54,5.36,1.72,.73,"Center each day\n89 block sums")
    box(5.68,5.36,1.85,.73,"3321 active\nwindow values")
    box(8.02,5.36,1.8,.73,"Predicted softmax\nτpred = 0.10",fc="#E2F1EF")
    for a,b in [((1.32,5.72),(1.72,5.72)),((3.12,5.72),(3.54,5.72)),((5.26,5.72),(5.68,5.72)),((7.53,5.72),(8.02,5.72))]: arrow(a,b)
    box(1.72,3.9,1.4,.73,"Training-only\ntrue prices A",fc="#FBF0E7",ec="#E7CBB6")
    box(3.54,3.9,1.72,.73,"True centered\nblock sums",fc="#FBF0E7",ec="#E7CBB6")
    box(5.68,3.9,1.85,.73,"True window\nvalues",fc="#FBF0E7",ec="#E7CBB6")
    box(8.02,3.9,1.8,.73,"Target softmax\nτtgt = 0.05",fc="#FBF0E7",ec="#E7CBB6")
    for a,b in [((3.12,4.26),(3.54,4.26)),((5.26,4.26),(5.68,4.26)),((7.53,4.26),(8.02,4.26))]: arrow(a,b,ORANGE)
    ax.text(4.4,5.0,"block MSE",ha="center",fontsize=8.5,color=TEAL)
    arrow((4.4,5.36),(4.4,5.16),TEAL); arrow((4.4,4.63),(4.4,4.87),TEAL)
    ax.text(8.92,5.0,"listwise CE",ha="center",fontsize=8.5,color=TEAL)
    arrow((8.92,5.36),(8.92,5.16),TEAL); arrow((8.92,4.63),(8.92,4.87),ORANGE)
    box(3.54,2.75,6.28,.65,"Minimize  block MSE + 2 × listwise CE",fc="#E2F1EF",ec="#A9D2CC",bold=True,fs=10)
    arrow((4.4,3.9),(4.4,3.40),TEAL); arrow((8.92,3.9),(8.92,3.40),TEAL)
    ax.plot([3.54,.65,.65,2.42],[3.075,3.075,4.95,4.95],color=TEAL,lw=1.05,ls="--")
    arrow((2.42,4.95),(2.42,5.36),TEAL,True)
    ax.text(.08,2.58,"Gradient update",color=TEAL,fontsize=8.5)
    ax.plot([0,10],[2.36,2.36],color="#C7D4DE",lw=.8)
    ax.text(.05,2.01,"B  EVALUATION: freeze the recipe, then settle a legal action",fontweight="bold",fontsize=11)
    box(.05,.79,2.03,.76,"Five fixed seeds\n42, 43, 44, 45, 46")
    box(2.61,.79,1.81,.76,"Mean price curve\nq̄ = mean(qk)",fc="#E2F1EF")
    box(4.94,.79,2.1,.76,"Exact argmax\n8 + 8 windows / idle",bold=True)
    box(7.57,.79,2.24,.76,"Lock action u*\nRealized score: Aᵀu*",fc="#FBF0E7",ec="#E7CBB6")
    for a,b in [((2.08,1.17),(2.61,1.17)),((4.42,1.17),(4.94,1.17)),((7.04,1.17),(7.57,1.17))]: arrow(a,b)
    ax.text(.06,.18,"Conceptual diagram  •  No evaluation labels enter the predictor or action choice.",fontsize=8.7,color=GRAY)
    save(fig,"fig01_method")

def scatter_panels(kind):
    fig, axs = plt.subplots(2,2,figsize=(7.4,5.55),sharey=True)
    if kind=="rmse":
        rows=[("rmse_same_prediction","point_mse","RMSE (same saved prediction)"),
              ("centered_rmse","centered_mse","Day-centered RMSE")]
        fname="fig02_rmse"
    else:
        rows=[("listwise_proxy","listwise_proxy","Fixed headline training proxy (lower is better)"),
              ("window_spearman","window_spearman","Mean daily window Spearman (higher is better)")]
        fname="fig03_proxy"
    for row,(xcol,key,label) in enumerate(rows):
        for col,v in enumerate(["v1","v2"]):
            ax=axs[row,col]; base(ax)
            d=M[M.source.eq("holdout_rmse_"+v)].copy()
            highlight=d.model.eq("transformer_contextual")
            ax.scatter(d.loc[~highlight,xcol],d.loc[~highlight,"realized_mean"],s=30,c=GRAY,alpha=.85,linewidth=.5,edgecolor="white")
            ax.scatter(d.loc[highlight,xcol],d.loc[highlight,"realized_mean"],s=68,c=TEAL,marker="D",linewidth=.7,edgecolor="white",zorder=4)
            r=C[C.version.eq(v)&C.metric.eq(key)&C.block_length.eq(7)].iloc[0]
            ax.set_title(f"{'ABCD'[row*2+col]}  {v}  |  {len(d)} fixed candidates",loc="left",pad=22)
            ax.text(0,1.035,f"Pearson r = {r.pearson:+.3f}     Spearman ρ = {r.spearman:+.3f}",transform=ax.transAxes,fontsize=8.2,color=GRAY)
            ax.set_xlabel(label,fontsize=8.5,labelpad=5)
            ax.set_ylim(4370,6490); ax.yaxis.set_major_locator(MaxNLocator(5))
            if col==0: ax.set_ylabel("Mean realized score")
            if kind=="rmse" and row==0:
                p=d.loc[highlight].iloc[0]
                ax.annotate("Transformer",(p[xcol],p.realized_mean),xytext=(14,10 if v=="v1" else -17),textcoords="offset points",fontsize=8,color=TEAL)
    fig.subplots_adjust(left=.10,right=.99,top=.90,bottom=.12,hspace=.72,wspace=.20)
    fig.text(.10,.015,"Each point: one fixed 59-day curve. Gray: other candidates. Teal diamond: contextual Transformer.",fontsize=8,color=GRAY)
    save(fig,fname)

def main_result():
    fig,axs=plt.subplots(1,2,figsize=(7.3,2.8),gridspec_kw={"width_ratios":[1.38,1]})
    ax=axs[0]; base(ax)
    labels=["Matched LightGBM","MSE Transformer reference","Frozen listwise ensemble"]
    values=[S["baseline"],S["reference"],S["headline"]]
    for y,(lab,x,color) in enumerate(zip(labels,values,[GRAY,NAVY,TEAL])):
        ax.scatter(x,y,s=63,c=color,zorder=3)
        ax.annotate(f"{x:,.2f}",(x,y),xytext=(7,5),textcoords="offset points",fontsize=9,color=color)
    ax.set_yticks(range(3),labels); ax.set_ylim(-.6,2.6); ax.set_xlim(5460,7140)
    ax.set_xlabel("Mean realized score (expanded dot scale)")
    ax.set_title("A  Absolute outcome",loc="left",pad=13)
    ax=axs[1]; base(ax)
    r=I[I.comparison.eq("headline_vs_reference")&I.block_length.eq(7)].iloc[0]
    ax.axhline(0,color=NAVY,lw=.85)
    ax.bar([0],[r.difference_mean],width=.36,color=TEAL,alpha=.85,zorder=3)
    ax.errorbar([0],[r.difference_mean],yerr=[[r.difference_mean-r.ci_low],[r.ci_high-r.difference_mean]],fmt="none",ecolor=NAVY,capsize=5,lw=1.4,zorder=4)
    ax.set_xlim(-.8,.8); ax.set_ylim(-440,1870)
    ax.set_xticks([0],["Headline − reference"]); ax.set_ylabel("Paired mean score difference")
    ax.set_title("B  Observed gain and uncertainty",loc="left",pad=13,fontsize=9.8)
    ax.text(.04,.93,f"+{r.difference_mean:.2f}  (+{S['relative_gain_pct']:.2f}%)",transform=ax.transAxes,fontsize=10,fontweight="bold",color=TEAL)
    ax.text(.04,.02,f"95% block interval: [{r.ci_low:.0f}, {r.ci_high:.0f}]",transform=ax.transAxes,fontsize=8,color=GRAY)
    fig.subplots_adjust(left=.25,right=.985,bottom=.21,top=.86,wspace=.67)
    save(fig,"fig04_main")

def ensemble():
    fig,axs=plt.subplots(1,2,figsize=(7.4,3.0),gridspec_kw={"width_ratios":[1.5,1]})
    ax=axs[0]; base(ax)
    bags=["window_ce_lstm","listwise_transformer","spo_lstm"]
    labels=["Window CE / LSTM","Listwise / Transformer","SPO+ / LSTM"]
    for j,bag in enumerate(bags):
        d=M[M.source.eq("holdout_seed_bag_v1") & M.model.str.match(bag+r"_seed4[2-6]$")].sort_values("model")
        color=TEAL if "listwise" in bag else GRAY
        ax.plot(d.realized_mean,[j]*len(d),color=color,alpha=.4,lw=2)
        ax.scatter(d.realized_mean,j+np.linspace(-.10,.10,5),s=29,c=color,alpha=.8,zorder=3)
        r=B[B.bag.eq(bag)].iloc[0]
        ax.scatter(r.single_seed_mean,j,marker="D",facecolor="white",edgecolor=NAVY,s=46,zorder=4)
        ax.scatter(r.curve_mean_score,j,marker="*",color=ORANGE,s=140,zorder=5)
    ax.set_yticks(range(3),labels); ax.set_ylim(2.4,-.5); ax.set_xlim(5560,6840)
    ax.set_xlabel("Mean realized score"); ax.set_title("A  Fixed seeds and curve averaging",loc="left",pad=12)
    ax.scatter([],[],c=GRAY,s=24,label="Seed")
    ax.scatter([],[],marker="D",facecolor="white",edgecolor=NAVY,s=30,label="Seed-score mean")
    ax.scatter([],[],marker="*",color=ORANGE,s=75,label="Curve mean")
    ax.legend(loc="upper center",bbox_to_anchor=(.48,-.27),ncol=3,frameon=False,fontsize=7.3,columnspacing=.9,handletextpad=.3)
    ax=axs[1]; base(ax)
    a,b=S["nonensemble_remainder"],S["ensemble_increment"]
    ax.bar([0],[a],width=.95,color=NAVY)
    ax.bar([0],[b],bottom=[a],width=.95,color=TEAL)
    ax.text(0,a/2,f"+{a:.2f}\nRecipe remainder",ha="center",va="center",color="white",fontsize=7.6)
    ax.text(0,a+b/2,f"+{b:.2f}\nCurve averaging",ha="center",va="center",color="white",fontsize=7.6)
    ax.text(0,a+b+18,f"Total +{a+b:.2f}",ha="center",fontsize=9,fontweight="bold")
    ax.axhline(0,color=NAVY,lw=.7); ax.set_xlim(-.68,.68); ax.set_ylim(0,570)
    ax.set_xticks([]); ax.set_ylabel("Observed score difference")
    ax.set_title("B  Arithmetic decomposition",loc="left",pad=12,fontsize=9.5)
    ax.text(.5,-.12,"Descriptive; not a causal loss effect",ha="center",transform=ax.transAxes,fontsize=8,color=GRAY)
    fig.subplots_adjust(left=.22,right=.98,bottom=.28,top=.87,wspace=.62)
    save(fig,"fig05_ensemble")

def daily():
    fig,axs=plt.subplots(2,1,figsize=(7.3,4.3),sharex=True)
    dates=pd.to_datetime(P.date); special=P.date.eq("2025-12-30")
    worst=P.date.eq(S["worst_paired_day"])
    ax=axs[0]; base(ax)
    ax.bar(dates,P.paired_gain,width=.83,color=np.where(special,ORANGE,np.where(worst,NAVY,GRAY)))
    w=P.index[worst][0]
    ax.annotate(f"{S['worst_paired_day'][5:]}: {P.loc[w,'paired_gain']:,.0f}\nlargest single-day loss",
                (dates.iloc[w],P.loc[w,"paired_gain"]),xytext=(10,-34),textcoords="offset points",
                arrowprops={"arrowstyle":"-","color":NAVY},fontsize=8.2,color=NAVY)
    ax.axhline(0,color=NAVY,lw=.8); ax.set_ylabel("Daily difference")
    ax.set_title("A  Headline − reference: all 59 days",loc="left")
    j=P.index[special][0]
    ax.annotate(f"Dec 30: +{P.loc[j,'paired_gain']:,.0f}\n{S['dec30_share_total_gain_pct']:.1f}% of total net gain",(dates.iloc[j],P.loc[j,"paired_gain"]),xytext=(-167,-5),textcoords="offset points",arrowprops={"arrowstyle":"-","color":ORANGE},fontsize=8.5,color=ORANGE)
    ax.set_ylim(min(-8000,P.paired_gain.min()*1.15),29000)
    ax=axs[1]; base(ax)
    cumulative=P.paired_gain.cumsum()
    ax.plot(dates,cumulative,c=TEAL,lw=1.8)
    ax.scatter(dates.iloc[j],cumulative.iloc[j],s=35,c=ORANGE,zorder=4)
    ax.axhline(0,color=NAVY,lw=.8); ax.set_ylabel("Cumulative difference")
    ax.set_title("B  Cumulative realized gain",loc="left")
    ax.text(.01,.90,f"All days: mean +{S['paired_gain']:.2f}   |   Excluding Dec 30: mean +{S['without_dec30_gain']:.2f}",transform=ax.transAxes,fontsize=8.2,color=GRAY)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlabel("Evaluation date, 2025")
    for ax in axs:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f"{x/1000:g}k"))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    fig.subplots_adjust(left=.11,right=.985,top=.94,bottom=.11,hspace=.45)
    save(fig,"fig06_daily")

method()
scatter_panels("rmse")
scatter_panels("proxy")
main_result()
ensemble()
daily()
print("Wrote six PDF + PNG figures from",data)
