from project_config import ROOT as PUBLIC_WORK_ROOT
"""Read audited comparison and prepare content inserts; no new forecast fit."""
from pathlib import Path
import json, hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd

P=(PUBLIC_WORK_ROOT/'results/joint_fx_v18/inference/multivariate_comparison')
PAIR=['IN_FHS_REF_POOL','SCALAR_BEKK_INTERNAL']

def main():
    manifest=json.loads((P/'manifest.json').read_text())
    assert manifest['status']=='completed'
    model=pd.read_csv(P/'model_summary.csv',float_precision='round_trip')
    contrasts=pd.read_csv(P/'paired_score_contrasts_whole_date.csv',float_precision='round_trip')
    task=pd.read_csv(P/'paired_score_contrasts_eight_task_tail_family.csv',float_precision='round_trip')
    sets=json.loads((P/'mcs_sets_35_42.json').read_text())
    audit=manifest['artifact_audit']
    annual=pd.read_csv(P/'annual_task_summary.csv',float_precision='round_trip').groupby(['year','model_key','tau']).mean_score.mean().unstack('model_key')
    annual['mixture_minus_BEKK']=annual[PAIR[0]]-annual[PAIR[1]]
    annual.reset_index().to_csv(P/'annual_whole_date_comparison.csv',index=False,float_format='%.17g')
    precision=pd.read_csv(P/'BEKK_precision_score_changes.csv',float_precision='round_trip')
    ps=precision.groupby('tau').agg(n_task_rows=('score_high_minus_low','size'),
        mean_score_change=('score_high_minus_low','mean'),
        mean_absolute_score_change=('score_high_minus_low',lambda v:v.abs().mean()),
        median_absolute_score_change=('score_high_minus_low',lambda v:v.abs().median()),
        max_absolute_score_change=('score_high_minus_low',lambda v:v.abs().max()),
        breach_changes=('breach_changed','sum')).reset_index()
    ps.to_csv(P/'BEKK_precision_score_summary.csv',index=False,float_format='%.17g')
    rows=[]
    for tau in [.95,.99]:
        old=float(model[(model.model_key==PAIR[0])&(model.tau==tau)].mean_score.iloc[0])
        new=float(model[(model.model_key==PAIR[1])&(model.tau==tau)].mean_score.iloc[0])
        for block in [20,60]:
            c=contrasts[(contrasts.tau==tau)&(contrasts.block==block)].iloc[0]
            assert np.isclose(old-new,c.mean_difference,atol=1e-13)
            rows.append(f'| {tau:.2f} | {block} | {old:.5f} | {new:.5f} | {c.mean_difference:.5f} | [{c.simultaneous_low:.5f}, {c.simultaneous_high:.5f}] |')
    q95=contrasts[(contrasts.tau==.95)&(contrasts.block==60)].iloc[0]
    q99=contrasts[(contrasts.tau==.99)&(contrasts.block==60)].iloc[0]
    s95=[x for x in sets if x['family_n']==35 and x['tau']==.95]
    s99=[x for x in sets if x['family_n']==35 and x['tau']==.99]
    assert task.loc[task.block==60].simultaneous_high.gt(0).all()
    text=f'''## Suggested main-text results insert

We add a covariance-targeted scalar BEKK control with a finite empirical internal radial law. Its full conditional covariance recursion, likelihood and radial coordinates transform consistently across the five currency bases. The reference mixture has lower mean FZ0 scores on the same 2,046 forecast origins: the mixture-minus-BEKK differences are {q95.mean_difference:.5f} at 0.95 and {q99.mean_difference:.5f} at 0.99. Common-date block bootstrap intervals simultaneously covering the two tail-level contrasts exclude zero for both block lengths. With 60-date blocks, the intervals are [{q95.simultaneous_low:.5f}, {q95.simultaneous_high:.5f}] and [{q99.simultaneous_low:.5f}, {q99.simultaneous_high:.5f}], respectively. These contrasts concern the four-task average. Simultaneous intervals for all eight task-by-tail contrasts cross zero with 60-date blocks; with 20-date blocks, only the four cashflow contrasts exclude zero. The annual average also favours BEKK in 2020 and 2021 at both levels, and in 2025 at 0.95. The findings therefore support an average-score improvement relative to this parsimonious multivariate control, rather than dominance for every holding, reporting currency or market period.

Including the control expands the restricted candidate family to 35 models. The 95% model confidence sets retain the mixture at both tail levels and block lengths, with inclusion pvalues {s95[0]['mixture_mcs_pvalue']:.3f}/{s95[1]['mixture_mcs_pvalue']:.3f} at 0.95 and {s99[0]['mixture_mcs_pvalue']:.3f}/{s99[1]['mixture_mcs_pvalue']:.3f} at 0.99 for 20/60-date blocks. BEKK is excluded at 0.95 under both block lengths and at 0.99 with 20-date blocks, but remains in the 0.99 set with 60-date blocks (p={s99[1]['BEKK_mcs_pvalue']:.3f}). The broader 42-model sensitivity, which additionally retains the unresolved standalone copula forecasts as diagnostics, gives the same inclusion pattern for the two focal systems. Set membership does not establish equality, economic equivalence or a uniquely best model. These results are conditional on the recorded historical forecasts and do not account for the paper's earlier design search.

Calibration remains a separate limitation. At 0.95, both systems are conservative for funded assets: their task-specific breach rates are about 3.4–3.5% against a 5% nominal rate. The mixture cashflow rates are 4.99–5.03%, compared with 4.45% for BEKK. At 0.99, the BEKK cashflow tasks each have 40 breaches (1.96%), versus 28 (1.37%) for the mixture and 20.46 nominally expected. The normalized ES identification means for BEKK cashflows are −0.483 and −0.492, with 60-lag HAC intervals excluding zero. Formal joint pvalues for tasks with fewer than 30 breaches are withheld. A lower proper score therefore does not establish complete tail calibration.

| Tail level | Block length | Mixture FZ0 | BEKK FZ0 | Mixture minus BEKK | Simultaneous 95% interval, two-tail family |
| --- | --- | --- | --- | --- | --- |
'''+ '\n'.join(rows)+'''

## Suggested methods insert

The score comparison keeps all four task forecasts from a date together and applies 5,000 circular fixed-length date-block resamples of lengths 20 and 60, truncating the last block to the original 2,046-date sample size. We use the maximum absolute standardized bootstrap deviation to construct simultaneous intervals, separately for the two whole-date tail contrasts and the eight task-by-tail contrasts. The model confidence set uses the Hansen–Lunde–Nason range statistic, recentered bootstrap pairwise score differences and coherent elimination of the model with the largest adverse studentized pairwise difference. Inclusion pvalues are monotonized over successive eliminations. Calibration moments and predictable lagged-hit instruments follow the existing positive-loss VaR/ES convention; the first two evaluation origins are omitted only from instrumented tests because their completed lagged targets are unavailable in the new forecast file.

## Suggested numerical/PIT paragraph

Numerical closure and precision checks address implementation error separately from prediction quality. All 24 annual BEKK optimizer starts converge. Across eight annual fits and five bases, the maximum normalized covariance-recursion and likelihood-offset discrepancies are 1.11×10⁻¹⁵ and 1.78×10⁻¹⁵. On the first, middle and last eligible origins of each year (24 origins, 192 task-by-tail checks), increasing the nested scrambled Sobol design from 65,536 to 262,144 paths changes BEKK ES by at most 0.448% at 0.95 and 1.071% at 0.99; none of these checks changes the realized breach classification. The expanded copula check covers the middle eligible origin of each calendar month (96 origins, 3,840 checks across five bases), with a maximum absolute relative ES change of 1.668%. These selected-origin comparisons are numerical sensitivities, rather than confidence intervals for true ES or a bound on error at every origin. The largest corresponding BEKK FZ0 perturbation is 0.226, so ES precision alone does not imply uniformly negligible score error. PITs are computed from the actual cash-loss scenarios for every task and origin. No realized loss is tied with an atom of the exact empirical laws, and all PIT upper-tail indicators agree with the stored breach classifications. Their descriptive distributions and support exceedances remain reported; no independent-observation uniformity pvalue is used. The five-reference empirical mixture contains 6,245 dependent scenario paths and is integrated exactly; that number is not an independent sample size.
'''
    (P/'MANUSCRIPT_INSERT.md').write_text(text,encoding='utf-8')
    capsule=dict(primary_score_difference={'q95':q95.mean_difference,'q99':q99.mean_difference},
        all_eight_contrast_intervals_cross_zero_with_block60=True,
        q99_BEKK_included_block60=True,any_claim_of_economic_equivalence=False,
        all_annual_differences=annual.reset_index()[['year','tau','mixture_minus_BEKK']].to_dict('records'),
        primary_outputs_manifest_sha256=hashlib.sha256((P/'manifest.json').read_bytes()).hexdigest(),
        reporting_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (P/'reporting_manifest.json').write_text(json.dumps(capsule,indent=2)+'\n',encoding='utf-8')
    readme='''Completed empirical reanalysis, conditional on recorded forecast configurations.

Read MANUSCRIPT_INSERT.md for the scoped main-text statements. model_summary.csv and task_summary.csv provide all focal scores and breach counts. paired_score_contrasts_whole_date.csv uses a simultaneous two-tail family; paired_score_contrasts_eight_task_tail_family.csv uses all eight tasks-by-tails. mcs_pvalues_35_42.csv and mcs_sets_35_42.json preserve both candidate-family sensitivities. calibration.csv keeps sparse-tail flags and unadjusted diagnostic pvalues, with no claim that non-rejection means calibration. annual_whole_date_comparison.csv records the years favouring each focal system.

Numerical checks are in precision_summary.csv, BEKK_precision_score_changes.csv, BEKK_precision_score_summary.csv, exact_empirical_risk_reconstruction_audit.csv and manifest.json. PIT summaries are descriptive; no iid KS pvalue is reported. Exact-empirical support/PIT ties, target dates, score formula, hit classifications, past-only fit cutoffs, all covariance bases, all optimizer starts and script/source hashes were checked. MCS implementation passes the existing independently coded scalar-loop toy, positive-score-rescaling/common-date-shift invariance and explicit-block-resampling audit.

The BEKK control is covariance-targeted and scalar in its ARCH/GARCH coefficients. It is a meaningful closed multivariate benchmark, not an exhaustive test of all full BEKK, DCC, dynamic-copula or multivariate machine-learning specifications. Its closure is a structural property verified numerically under transformed parameters/states; five independent nonlinear refits are not claimed. Numerical selected-origin stability is not a proof of negligible QMC error throughout the historical sample. The old 34/41-model results in the parent inference directory are preserved.

Reproduction: run joint18_inference_multivariate_compare.py once in an absent comparison directory, then joint18_inference_multivariate_reporting.py. Scientific Python312, fx_windows_platform_compat imported first, BLAS threads1 and D-only TEMP/TMP/cache. No raw-data file is changed.
'''
    (P/'AUDIT_README.md').write_text(readme,encoding='utf-8')
    print('Reporting outputs complete')

if __name__=='__main__':main()
