"""Generate the same numeric tables for Chinese and English from exported records."""
from pathlib import Path
import argparse
import json
import pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
args=parser.parse_args()
root=args.root
packaged = not (root/'shared').is_dir()
data=root/'data' if packaged else root/'shared'/'data'
M=pd.read_csv(data/'model_metrics.csv'); C=pd.read_csv(data/'correlations.csv')
B=pd.read_csv(data/'ensemble_decomposition.csv'); I=pd.read_csv(data/'paired_intervals.csv')
D=pd.read_csv(data/'rmse_provenance.csv'); CP=pd.read_csv(data/'controlled_comparisons_machine.csv',encoding='utf-8-sig')
S=json.loads((data/'summary.json').read_text(encoding='utf-8'))
def esc(s):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$',
             '#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(mapping.get(c,c) for c in str(s))
def f(x,n=2):
    # Typeset a real minus sign; a bare ASCII hyphen renders as a short dash in text mode.
    s=f'{x:.{n}f}'
    return r'$-$'+s[1:] if s.startswith('-') else s
def rng(lo,hi,n=2): return '['+f(lo,n)+', '+f(hi,n)+']'
def emit(out,name,caption,label,columns,header,rows,note=''):
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    text=(r'\begin{table}[!htbp]'+'\n'+r'\centering\small'+'\n'+
          '\\caption{'+caption+'}\\label{'+label+'}\n'+
          '\\begin{tabularx}{\\linewidth}{'+columns+'}\n\\toprule\n'+
          ' & '.join(header)+r' \\'+'\n\\midrule\n'+body+'\n\\bottomrule\n\\end{tabularx}\n')
    if note: text+='\\par\\vspace{3pt}{\\footnotesize '+note+'}\n'
    text+='\\end{table}\n'
    assert not any(ord(ch)<32 and ch!='\n' for ch in text), (name, 'Control character in generated TeX')
    (out/(name+'.tex')).write_text(text,encoding='utf-8')
languages = [json.loads((root/'BUILD_INFO.json').read_text())['language']] if packaged else ['zh','en']
for lang in languages:
    zh=lang=='zh'; out=(root if packaged else root/lang)/'tables'; out.mkdir(parents=True,exist_ok=True)
    def L(a,b): return a if zh else b
    roles=[('holdout_rmse_v2','lgb_baseline',L('匹配 LightGBM 基线','Matched LightGBM')),
           ('holdout_rmse_v2','transformer_contextual',L('MSE Transformer 参考','MSE Transformer reference')),
           ('holdout_seed_bag_v1','listwise_transformer_mean',L('冻结 listwise 集成','Frozen listwise ensemble'))]
    rows=[]
    for src,model,label in roles:
        r=M[M.source.eq(src)&M.model.eq(model)].iloc[0]
        rows.append([label,f(r.predicted_mean),f(r.realized_mean),f(100*r.capture_aggregate)+r'\%',f(r.realized_mean-S['reference'])])
    emit(out,'main_results',L('相同 59 日与动作合同下的主结果。','Main outcomes under the same 59-day evaluation and action contract.'),'tab:main','@{}Xrrrr@{}',
         [L('方案','Configuration'),L('预测分','Predicted'),L('实现分','Realized'),r'$C_{\rm agg}$',L('对参考差','vs. ref.')],rows,
         L('所有分数均为日均原始调度分数。共同 oracle 为 9000.89；预测分与实现分分开列示，不把预测值当成结算结果。','Scores are daily means in the original scoring units. The shared oracle mean is 9000.89. Predicted values are not settlements.'))
    r=I[I.comparison.eq('headline_vs_reference')&I.block_length.eq(7)].iloc[0]
    rows=[[L('日均配对差','Paired mean difference'),f(S['paired_gain'])],
          [L('配对 IID 标准误（仅参考）','Paired IID standard error (reference only)'),f(S['paired_iid_se'])],
          [L('7 日块 95\% 条件重采样区间','7-day block 95\% conditional resampling interval'),rng(r.ci_low,r.ci_high)],
          [L('胜 / 平 / 负（容差 $10^{-6}$）','Wins / ties / losses (tolerance $10^{-6}$)'),f"{S['wins']} / {S['ties']} / {S['losses']}"],
          [L('逐日差中位数','Median daily difference'),f(S['paired_median'])],
          [L('12 月 30 日占总净增益','Dec 30 share of total net gain'),f(S['dec30_share_total_gain_pct'])+r'\%'],
          [L('剔除 12 月 30 日后的日均差','Mean difference excluding Dec 30'),f(S['without_dec30_gain'])],
          [L('胜日差值合计 / 负日差值合计','Winning-day total / losing-day total'),f(S['gross_winning_day_total'])+' / '+f(S['gross_losing_day_total'])],
          [L('最差单日差值及日期','Largest single-day loss and date'),f(S['worst_paired_difference'])+' ('+S['worst_paired_day']+')'],
          [L('最差单日实现分：主方案 / 参考','Worst realized day: headline / reference'),f(S['worst_day_headline'])+' / '+f(S['worst_day_reference'])],
          [L('负分日数：主方案 / 参考','Negative-score days: headline / reference'),f"{S['negative_days_headline']} / {S['negative_days_reference']}"]]
    emit(out,'paired',L('主比较的逐日证据、尾部与不确定性。','Paired evidence, loss tail, and uncertainty for the main comparison.'),'tab:paired','@{}Xr@{}',[L('统计量','Statistic'),L('结果','Value')],rows,
         L('剔除单日是事后敏感性分析；主结果始终使用全部 59 日。区间未校正此前对评价期的多次观察。最差单日与负分日描述本时段已实现的下行，不是尾部风险模型。','Date exclusion is a post hoc sensitivity analysis; the primary result retains all 59 days. Intervals do not correct for historical evaluation exposure. The worst-day and negative-day rows describe realized downside in this period; they are not a tail-risk model.'))
    rows=[]
    for _,r in B.iterrows():
        label={'window_ce_lstm':'Window CE / LSTM','listwise_transformer':'Listwise / Transformer','spo_lstm':'SPO+ / LSTM'}[r.bag]
        rows.append([label,f(r.single_seed_mean),f(r.single_seed_sd),f(r.curve_mean_score),f(r.ensemble_increment)])
    emit(out,'bags',L('固定角色的集成分解；按方案角色排列，不据此重选 headline。','Ensemble decomposition in fixed role order; no headline reselection.'),'tab:bags','@{}Xrrrr@{}',
         [L('配方','Recipe'),L('单种子均值','Seed mean'),L('种子 SD','Seed SD'),L('曲线平均','Curve mean'),L('集成增量','Increment')],rows,
         L('每条配方均为种子 42--46。单种子 SD 描述这五次训练的离散度，不是时间泛化误差。','Every recipe uses seeds 42--46. Seed SD describes these five fits, not uncertainty over future periods.'))
    rows=[]
    for family in ['Transformer','LSTM','DLinear']:
        def get(metric):
            selected=CP[CP.candidate.eq('codex_mse_'+family.lower()+'_'+metric)]
            assert len(selected)==1, (family,metric,'candidate must be unique')
            return float(selected.iloc[0].realized_profit_mean)
        a=get('rmse'); b=get('realized')
        rows.append([family,f(a),f(b),f(b-a)])
    emit(out,'selection_control',L('保持 MSE 训练、改变验证选模指标的现有对照。','Existing control: MSE training with a changed validation selection metric.'),'tab:selection','@{}Xrrr@{}',
         [L('模型族','Family'),L('按 RMSE 选','Select by RMSE'),L('按实现分选','Select by realized'),L('差值','Difference')],rows,
         L(r'来源为保存的对照实验；这些参考不是表~\ref{tab:main} 中所有历史模型的重新训练版本。',r'These are the recorded control experiment, not refits of every historical candidate in Table~\ref{tab:main}.'))
    rows=[[r.source.replace('holdout_rmse_',''),r'\texttt{'+esc(r.model)+'}',f(r.original_rmse,6),f(r.same_prediction_rmse,6),f(r.realized_mean)] for _,r in D[D.source.isin(['holdout_rmse_v1','holdout_rmse_v2'])].iterrows()]
    emit(out,'provenance',L('历史 RMSE 与实际结算曲线的 RMSE：保留原表，新增同源复算。','Historical RMSE versus the RMSE of the settled saved curve.'),'tab:provenance','@{}lXrrr@{}',
         [L('组','Set'),L('模型','Model'),L('历史表','Historical'),L('同源复算','Same curve'),L('实现分','Realized')],rows,
         L('本表不覆盖任何原始 RMSE/MAE 字段。其他点模型在 $10^{-5}$ 容差内吻合。','No original RMSE/MAE field is overwritten. Other point candidates agree within $10^{-5}$.'))
    rows=[]
    names={'point_mse':'RMSE','centered_mse':L('日中心 RMSE','Centered RMSE'),'listwise_proxy':L('Listwise 代理','Listwise proxy'),'window_spearman':L('窗口排序 $T$','Window rank $T$')}
    for _,r in C[C.block_length.eq(7)].iterrows():
        rows.append([r.version,names[r.metric],f(r.pearson,3),rng(r.pearson_ci_low,r.pearson_ci_high,3),f(r.spearman,3),rng(r.spearman_ci_low,r.spearman_ci_high,3)])
    emit(out,'correlations',L('跨候选关联及共同按天重采样区间。','Across-candidate associations and shared day-resampling intervals.'),'tab:corr','@{}lXrcr c@{}'.replace(' ',''),
         [L('组','Set'),L('横轴指标','Metric'),r'$r$',L('95\% 区间','95\% interval'),r'$\rho$',L('95\% 区间','95\% interval')],rows,
         L(r'$r$ 为 Pearson；$\rho$ 为跨候选 Spearman。窗口 $T$ 本身另含窗口内 Spearman。7 日块、10,000 次 percentile 重采样，仅改变日期、固定候选集合；区间不包含候选搜索或抽样不确定性。',r'$r$: Pearson; $\rho$: across-candidate Spearman. $T$ separately uses within-day Spearman. Percentile intervals: 7-day blocks, 10,000 replicates, dates resampled and candidates fixed. They exclude candidate-search and candidate-sampling uncertainty.'))
    rows=[]
    for _,r in I.iterrows():
        label=L('主方案－参考','Headline minus reference') if r.comparison=='headline_vs_reference' else L('曲线平均－单种子均值','Curve mean minus seed mean')
        rows.append([label,int(r.block_length),f(r.difference_mean),rng(r.ci_low,r.ci_high)])
    emit(out,'bootstrap',L('块长敏感性：同一冻结数据，三种重采样长度。','Block-length sensitivity on the same frozen evidence.'),'tab:bootstrap','@{}Xrrc@{}',
         [L('比较','Comparison'),L('块长','Days/block'),L('日均差','Mean diff.'),L('95\% 区间','95\% interval')],rows)
print('Generated bilingual numeric tables')
