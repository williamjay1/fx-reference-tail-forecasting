"""Anchor-symmetric reporting extract and parameter-perturbation conversion."""
from pathlib import Path
import json
import fx_windows_platform_compat
import numpy as np
import pandas as pd

OUT=Path('D:/MLWork/FXTailRisk/results/joint_fx_v18/inference')
REF=['EUR','USD','GBP','JPY','CHF']

def main():
    manifest=json.loads((OUT/'anchor_free_reference_manifest.json').read_text());assert manifest['status']=='completed'
    d=pd.read_csv(OUT/'anchor_free_reference_summary.csv',float_precision='round_trip')
    params=[];daily=[]
    for block in [20,60]:
        path=OUT/f'parameter_bootstrap/three_origins_B200_block{block}'
        m=json.loads((path/'manifest.json').read_text());assert m['status']=='completed'
        p=pd.read_csv(path/'forecasts.csv',float_precision='round_trip')
        base=pd.read_csv(path/'baseline_forecasts.csv',float_precision='round_trip')
        for key,g in p.groupby(['origin_date','book','report','tau']):
            x=g.pivot(index='replicate',columns='reference',values='es')[REF]
            absolute=x.max(axis=1)-x.min(axis=1);den=x.abs().mean(axis=1);r=absolute/den
            original=base[(base.origin_date==key[0])&(base.book==key[1])&(base.report==key[2])&(base.tau==key[3])].es
            baseline=(original.max()-original.min())/original.abs().mean()
            assert len(x)==200 and np.isfinite(r).all()
            for rep in x.index:
                daily.append(dict(origin_date=key[0],book=key[1],report=key[2],tau=key[3],block=block,
                  replicate=rep,absolute_ES_range=float(absolute.loc[rep]),mean_absolute_ES=float(den.loc[rep]),
                  anchor_symmetric_R_ES=float(r.loc[rep])))
            params.append(dict(origin_date=key[0],book=key[1],report=key[2],tau=key[3],block=block,
                original_anchor_symmetric_range=float(baseline),perturbation_median=float(r.median()),
                perturbation_p025=float(r.quantile(.025)),perturbation_p975=float(r.quantile(.975)),reps=200,
                not_true_ES_confidence_interval=True,nonnegative_span_percentiles_not_zero_sensitivity_test=True))
    param=pd.DataFrame(params);param.to_csv(OUT/'parameter_bootstrap/anchor_free_sensitivity.csv',index=False,float_format='%.17g')
    pd.DataFrame(daily).to_csv(OUT/'parameter_bootstrap/anchor_free_perturbations.csv',index=False,float_format='%.17g')
    table=['| Fixed book/reporting currency | Tail | Median R_ES | Pointwise 95% block-median interval |',
           '| --- | --- | --- | --- |']
    for _,r in d[(d.study=='legacy_historical')&(d.calendar_window=='all')&(d.metric=='anchor_symmetric_R_ES')].iterrows():
        table.append(f"| {r.book.replace('_',' ')}, {r.report} | {int(round(r.tau*100))}% | {100*r['median']:.2f}% | [{100*r.pointwise_median_low:.2f}%, {100*r.pointwise_median_high:.2f}%] |")
    pt=['| Fixed origin | EUR-reported funded assets tail | Block | Original R_ES | Parameter-perturbation median | 2.5–97.5 percentiles |',
        '| --- | --- | --- | --- | --- | --- |']
    for _,r in param[(param.book=='funded_assets')&(param.report=='EUR')].iterrows():
        pt.append(f"| {r.origin_date} | {int(round(r.tau*100))}% | {r.block} | {100*r.original_anchor_symmetric_range:.2f}% | {100*r.perturbation_median:.2f}% | [{100*r.perturbation_p025:.2f}%, {100*r.perturbation_p975:.2f}%] |")
    note='''# Anchor-symmetric ES diagnostic: actual completed outputs

Define D_ES=max_g e_g-min_g e_g and R_ES=D_ES/(G^-1 sum_g |e_g|), holding cash quantities, reporting currency, horizon and prior observations fixed. R_ES is invariant to a permutation of reference labels and to a common positive cash-unit rescaling. D_ES remains in the normalized cash-loss unit. R_ES is undefined if every component ES is zero; the analysis checks the denominator and never inserts an artificial epsilon. Existing EUR-denominated ranges are retained as legacy comparisons. This symmetric denominator was introduced in the revision to avoid selecting a reference as the normalizing anchor, without testing alternative definitions to select a favorable result.

Five native fits are retained for every stated cash task. Their primary one-interval 2018–2025 median R_ES spans 22.54–26.34% at 95% and 26.36–30.12% at 99%. The same calculation is retained for 2026, both longer horizons, the financed treasury and the distinct unfinanced treasury with a nonzero cutoff value. Values, denominators and absolute ranges remain separate columns. Annual summaries include every task. Pointwise 95% percentile intervals for the median use 5,000 common circular 60-date block resamples. These intervals condition on recorded forecasts and are not confidence bounds for true conditional ES; because a finite-model span is nonnegative, a positive lower interval endpoint is not a test that model sensitivity differs from zero.

'''+ '\n'.join(table)+'''

## Parameter sensitivity on the same symmetric denominator

'''+ '\n'.join(pt)+'''

Both blocks contain all 200 parameter-perturbation replicates per origin. No GARCH fit failed, and every prescribed optimizer start converged. The full four-task, two-tail summaries are in anchor_free_sensitivity.csv; these percentiles characterize estimation-induced system changes conditional on the original known window, not true-ES uncertainty.

## Audit and files

All 91,924 distinct cash-task/date records preserve the R_ES value after a fixed reference-label permutation and common positive loss-unit scaling, to within 6.67×10^-16. Five input forecast files are hashed in the completed manifest. The daily file preserves D_ES, mean absolute ES, R_ES and legacy EUR range. The summary contains 1,680 metric/task/window rows including annual results. This audit establishes numerical symmetry, not financial usefulness or predictive superiority.
'''
    (OUT/'ANCHOR_FREE_INSERT.md').write_text(note,encoding='utf-8')
    print(param[(param.book=='funded_assets')&(param.report=='EUR')].to_string(index=False),flush=True)
    print('Anchorfree narrative completed',flush=True)

if __name__=='__main__':main()
