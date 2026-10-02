from project_config import ROOT as PUBLIC_WORK_ROOT
"""Anchor-symmetric reference ES diagnostic; preserve legacy EUR denominator.

All five native IN-FHS fits enter each declared cash task. R_ES is undefined
when mean absolute ES is zero; no reference is selected as denominator.
Date-block median intervals remain conditional, pointwise variation summaries.
"""
from pathlib import Path
import hashlib,json,time
import fx_windows_platform_compat
import numpy as np
import pandas as pd
from joint18_inference import OUT,REPS,SEED,REFERENCES

ROOT=PUBLIC_WORK_ROOT

def bootstrap_medians(x,block=60,seed=SEED):
    x=np.asarray(x,float);n,k=x.shape;nb=(n+block-1)//block
    rng=np.random.default_rng(seed);out=np.empty((REPS,k))
    for lo in range(0,REPS,40):
        size=min(40,REPS-lo);starts=rng.integers(0,n,size=(size,nb))
        idx=((starts[:,:,None]+np.arange(block))%n).reshape(size,-1)[:,:n]
        out[lo:lo+size]=np.median(x[idx],axis=1)
    return out

def main():
    started=time.perf_counter();daily=[];summary=[];sources=[]
    datasets=[('legacy_historical',ROOT/'results/joint_fx_20261002/internal_full/predictions.csv'),
       ('temporal_2026',ROOT/'results/joint_fx_v18/temporal/predictions.csv'),
       ('horizon_historical',ROOT/'results/joint_fx_v18/horizons/predictions.csv'),
       ('unfinanced_historical',ROOT/'results/joint_fx_v18/horizons/unfinanced_treasury/predictions.csv'),
       ('unfinanced_2026',ROOT/'results/joint_fx_v18/temporal/unfinanced_treasury/predictions.csv')]
    permutations_checked=0;max_identity_error=0.
    for study,source in datasets:
        sources.append(source)
        d=pd.read_csv(source,parse_dates=['origin_date'],float_precision='round_trip')
        d=d[d.model=='IN_FHS1250'].copy()
        if study=='legacy_historical':d=d[d.origin_date.dt.year.between(2018,2025)]
        if 'horizon' not in d:d['horizon']=1
        for horizon,group in d.groupby('horizon'):
            p=group.pivot(index=['origin_date','book','report','tau'],columns='train_reference',values='es')[REFERENCES]
            if p.isna().any().any():raise ValueError('Five complete references required')
            a=p.to_numpy();den=np.abs(a).mean(axis=1);spread=a.max(axis=1)-a.min(axis=1)
            value=np.divide(spread,den,out=np.full(len(den),np.nan),where=den>0)
            if not np.isfinite(value).all():raise ValueError('Undefined anchorfree diagnostic: denominator zero')
            check=a[:,[4,1,0,3,2]]
            perm=(check.max(axis=1)-check.min(axis=1))/np.abs(check).mean(axis=1)
            max_identity_error=max(max_identity_error,float(np.max(np.abs(perm-value))))
            assert np.allclose(perm,value,atol=1e-14,rtol=0)
            # Positive unit changes multiply D and denominator equally.
            scaled=a*37.;recovered=(scaled.max(axis=1)-scaled.min(axis=1))/np.abs(scaled).mean(axis=1)
            max_identity_error=max(max_identity_error,float(np.max(np.abs(recovered-value))))
            assert np.allclose(recovered,value,atol=1e-14,rtol=0)
            permutations_checked+=len(a)
            z=pd.DataFrame({'absolute_ES_range':spread,'mean_absolute_ES':den,'anchor_symmetric_R_ES':value,
                       'legacy_EUR_relative_range':spread/np.abs(a[:,0])},index=p.index).reset_index()
            z['study']=study;z['horizon']=horizon;daily.append(z)
            # Shared date blocks across every task/metric within each horizon
            # and evaluation or calendar-year window. No independent investors.
            for year in ['all']+sorted(z.origin_date.dt.year.unique().tolist()):
                window=z if year=='all' else z[z.origin_date.dt.year==year]
                metrics=['anchor_symmetric_R_ES','absolute_ES_range','mean_absolute_ES','legacy_EUR_relative_range']
                wide=window.pivot(index='origin_date',columns=['book','report','tau'],values=metrics).sort_index(axis=1)
                assert not wide.isna().any().any()
                boot=bootstrap_medians(wide.to_numpy());median=wide.median().to_numpy()
                for j,key in enumerate(wide.columns):
                    metric,book,report,tau=key
                    summary.append(dict(study=study,horizon=horizon,calendar_window=str(year),book=book,
                       report=report,tau=tau,metric=metric,n_dates=len(wide),median=float(median[j]),
                       p90=float(wide.iloc[:,j].quantile(.9)),p99=float(wide.iloc[:,j].quantile(.99)),
                       pointwise_median_low=float(np.quantile(boot[:,j],.025)),
                       pointwise_median_high=float(np.quantile(boot[:,j],.975)),
                       bootstrap_block=60,bootstrap_reps=REPS,
                       not_significance_test_of_nonnegative_span=True))
            print(f'{study} horizon {horizon}: {len(z)} task dates; {time.perf_counter()-started:.1f}s',flush=True)
    pd.concat(daily).to_csv(OUT/'anchor_free_reference_daily.csv',index=False,float_format='%.17g')
    pd.DataFrame(summary).to_csv(OUT/'anchor_free_reference_summary.csv',index=False,float_format='%.17g')
    manifest=dict(status='completed',definition='(max_native_ES-min_native_ES)/mean_reference(abs(native_ES))',
      numerator='absolute_ES_range',denominator='mean_absolute_ES',legacy_EUR_relative_range_retained=True,
      five_references=REFERENCES,zero_denominator='undefined; fail loudly, never replace with epsilon',
      label_permutation_invariant=True,positive_cash_unit_invariant=True,
      tested_task_dates=permutations_checked,max_label_or_unit_invariance_residual=max_identity_error,
      block=60,reps=REPS,seed=SEED,intervals='pointwise 95% percentiles for median conditional on stored forecasts',
      annual_all_groups=True,nonnegative_spread_percentiles_not_zero_sensitivity_test=True,
      sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_seconds=time.perf_counter()-started)
    (OUT/'anchor_free_reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':main()
