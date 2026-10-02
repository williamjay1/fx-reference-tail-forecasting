"""Actual parameter perturbation and extension MCS narrative, computed tables."""
from pathlib import Path
import json
import fx_windows_platform_compat
import numpy as np
import pandas as pd

OUT=Path('D:/MLWork/FXTailRisk/results/joint_fx_v18/inference')

def run():
    combined=[];signed=[]
    for block in [20,60]:
        d=OUT/f'parameter_bootstrap/three_origins_B200_block{block}'
        m=json.loads((d/'manifest.json').read_text());assert m['status']=='completed' and m['failed_coordinate_fits']==0
        x=pd.read_csv(d/'range_sensitivity.csv',float_precision='round_trip');x['block']=block;combined.append(x)
        s=pd.read_csv(d/'signed_reference_sensitivity.csv',float_precision='round_trip');s['block']=block;signed.append(s)
    r=pd.concat(combined);s=pd.concat(signed)
    r.to_csv(OUT/'parameter_bootstrap/three_origin_range_all_blocks.csv',index=False,float_format='%.17g')
    s.to_csv(OUT/'parameter_bootstrap/three_origin_signed_all_blocks.csv',index=False,float_format='%.17g')
    table=['| Origin | Book, EUR reporting | Tail | Date block | Original five-reference range | Perturbation median | Perturbation 2.5–97.5 percentiles |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for _,x in r[r.report=='EUR'].sort_values(['origin_date','book','tau','block']).iterrows():
        table.append(f"| {x.origin_date} | {x.book.replace('_',' ')} | {int(round(x.tau*100))}% | {x.block} | {100*x.original_reference_range:.2f}% | {100*x.perturbation_range_median:.2f}% | [{100*x.perturbation_range_p025:.2f}%, {100*x.perturbation_range_p975:.2f}%] |")
    extension=json.loads((OUT/'extended_mcs_sets.json').read_text())
    mtable=['| Evaluation | Horizon | Tail | Date block | Dates | Candidate count | Retained | Mixture MCS p-value |',
           '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for x in extension:
        mtable.append(f"| {x['study']} | {x['horizon']} | {int(round(x['tau']*100))}% | {x['block']} | {x['n_dates']} | {x['family_n']} | {x['included_n']} | {x['mixture_mcs_pvalue']:.4f} |")
    note='''# Completed parameter-estimation sensitivity and extension MCS

## Methods insert

To assess conditional estimation sensitivity, we refit the five-coordinate GARCH system at the first eligible origins in 2018, 2021 and 2025. Each origin uses its annual prior-only 1,250-return estimation window. For each block length of 20 or 60 returns, 200 circular joint-date resamples are shared across every training reference. This gives 24,000 coordinate fits. Each perturbed parameter set filters the original known return history and rebuilds the original known internally standardized empirical window; no bootstrap return is used as a future outcome. Adjacent two-node scenarios and fractional-mass ES are enumerated exactly. The resulting percentiles characterize parameter-induced forecasting-system variation conditional on the observed window. They are neither confidence intervals for true conditional ES nor a full time-series bootstrap of the research and forecasting procedure. The first/mid/latest-year calendar origins were fixed for this sensitivity, after historical development exposure.

## Results insert

All 24,000 coordinate fits converge from all three prescribed optimizer starts; no replicate is discarded. Independently reconstructed 720 VaR/ES pairs differ from stored values by at most 2.50×10^-16 and 7.81×10^-17, respectively. At the three selected origins, parameter perturbation changes both the level and ordering of reference-specific ES. For EUR-reported funded assets on 4 January 2021, the original 95% five-reference range is 39.79%; the perturbation medians are 18.50% with 20-return blocks and 27.52% with 60-return blocks, with 2.5–97.5 percentile ranges of 7.11–33.22% and 13.31–47.24%. Some signed native-reference contrasts reverse under perturbation. These results prevent interpreting the original reference range as a pure, irreducible coordinate effect: estimation and filtering choices can materially modulate the variation. The range itself is mechanically nonnegative, and its positive perturbation percentiles do not test a zero-sensitivity null.

## Full EUR-report range table

'''+ '\n'.join(table)+'''

The complete USD-report and signed four-versus-EUR perturbations are retained in the accompanying numerical files. Reference quantities, reporting currency, empirical raw observations and valuation definitions are fixed; only jointly block-resampled estimation windows affect the GARCH parameters. The displayed percentiles describe 200 parameter perturbations per origin/block.

## Temporal and horizon MCS sensitivity

The later 2026 evaluation gives block-sensitive MCS results in a short series of 189 origins. For the IN-FHS mixture, 95% inclusion p-values are 0.3131 with 20-date blocks and 0.0328 with 60-date blocks; 99% values are 0.1056 and 0.0100. Thus the mixture is excluded at the 5% threshold under the longer block choice. Only about three 60-date blocks fit this short period, and deep-tail outcomes are sparse. These results are reported as an unstable temporal sensitivity and cannot be used to establish successful generalization or full calibration. All seven long-horizon systems survive every reported MCS specification at 5 and 20 reference intervals. Their retained-set membership does not offset their independently reported calibration problems.

'''+ '\n'.join(mtable)+'''

## Parameter-file interpretation

The annual GARCH table contains 160 fits: eight years, five training references, four coordinate pairs. Pair labels are explicit because changing a basis changes which bilateral increments enter the marginal system. The median-by-reference parameter summary aggregates different coordinate pairs and cannot be interpreted as a same-pair causal effect. The internal-copula matrices contain the shrunk correlation of Student-t quantile transformed rank pseudo-observations. They are not unconditional covariance matrices, raw Spearman coefficients, or estimates of the entire joint cash law. All full parameters and pair labels remain available for audit.
'''
    (OUT/'PARAMETER_AND_EXTENSION_INSERT.md').write_text(note,encoding='utf-8')
    print('Wrote completed parameter and extension extracts',flush=True)

if __name__=='__main__':run()
