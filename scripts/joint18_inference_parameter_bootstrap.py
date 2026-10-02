from project_config import ROOT as PUBLIC_WORK_ROOT
"""Joint-date parameter-estimation perturbation at three fixed origins.

Refit each basis to the SAME circular-block resample of its annual prior
1250-return estimation window. Forecast filters and internal residual windows
then use original known returns, not pseudo-future paths. This isolates a
conditional parameter-estimation sensitivity; it is not a true-ES confidence
interval or a repeated full forecasting/process bootstrap.
"""
from pathlib import Path
import argparse, hashlib, json, time
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
from joint17_internal_scenarios import internal_shocks

ROOT=PUBLIC_WORK_ROOT
OUT=ROOT/'results/joint_fx_v18/inference/parameter_bootstrap'
OLD=ROOT/'results/joint_fx_20261002'
PANEL=ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv'
SEED=20261002

def scenarios(known_returns,train,reference,fits):
    a,others=m.basis_matrix(reference)
    rb=100*(known_returns@a.T)
    tr=100*(train@a.T)
    hs=[];zs=[]
    for j,f in enumerate(fits):
        # The appended zero is never consumed by the filter to calculate
        # h_next, and it never enters a residual or a fitting window.
        h=m.variance_path(np.append(rb[:,j],0.),f['omega'],f['alpha'],f['beta'],f['mu'],float(np.var(tr[:,j])))
        eps=rb[:,j]-f['mu']
        post=f['omega']+f['alpha']*eps*eps+f['beta']*h[:-1]
        zs.append(eps/np.sqrt(post));hs.append(h[-1])
    z=internal_shocks(np.column_stack(zs)[-1250:])
    pairs=np.stack([z[:-1],z[1:]],axis=1)
    path=m.to_eur(m.garch_paths(pairs,np.array(hs),fits),reference,others)
    loss,err,diag=m.cash_losses(path,diagnostics=True)
    forecasts=[]
    for (book,report),values in loss.items():
        for tau in [.95,.99]:
            q,e=m.risk_pair(values,tau)
            forecasts.append(dict(book=book,report=report,tau=tau,var=q,es=e,
                 max_scenario_loss=float(values.max()),cash_identity_error=err,
                 cash_relative_identity_error=diag['relative_identity_error']))
    return forecasts

def run(args):
    out=OUT/args.tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() or (out/'fits.csv').exists():raise FileExistsError('Use fresh tag')
    panel=pd.read_csv(PANEL,parse_dates=['date'])
    r=np.diff(np.log(panel[m.CCY].to_numpy()[:,1:]),axis=0)
    records=[];pred=[];baseline=[];started=time.perf_counter();failures=0
    for year in args.years:
        params=json.loads((OLD/f'scenario_v2/parameters_{year}.json').read_text())
        cutoff=int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(params['cutoff']))[0])
        t=int(np.flatnonzero(panel.date.dt.year.to_numpy()==year)[0])
        assert cutoff<t and t>=1250
        train=r[cutoff-1250:cutoff];known=r[:t]
        rng=np.random.default_rng(SEED+year)
        origin=str(panel.date.iloc[t].date());print(f'origin {origin}: B={args.reps}, block={args.block}',flush=True)
        for member in params['fits']:
            for p in scenarios(known,train,member['reference'],member['garch']):
                baseline.append(dict(origin_date=origin,year=year,reference=member['reference'],**p))
        for rep in range(args.reps):
            nb=(1250+args.block-1)//args.block
            starts=rng.integers(0,1250,size=nb)
            indices=((starts[:,None]+np.arange(args.block))%1250).ravel()[:1250]
            boot=train[indices]
            # Preserve hashes for a deterministic audit that every reference
            # was fitted using the same selected four-dimensional returns.
            draw_hash=hashlib.sha256(indices.tobytes()).hexdigest()
            for ref in m.CCY:
                a,others=m.basis_matrix(ref);rb=100*(boot@a.T);fits=[]
                for j,node in enumerate(others):
                    try:
                        f=m.fit_garch(rb[:,j]);fits.append(f)
                        records.append(dict(origin_date=origin,year=year,replicate=rep,reference=ref,
                             node=m.CCY[node],joint_indices_sha256=draw_hash,status='converged',**f))
                    except Exception as error:
                        fits.append(None);failures+=1
                        records.append(dict(origin_date=origin,year=year,replicate=rep,reference=ref,
                             node=m.CCY[node],joint_indices_sha256=draw_hash,status='failed',error=repr(error)))
                if any(f is None for f in fits):continue
                for p in scenarios(known,train,ref,fits):
                    pred.append(dict(origin_date=origin,year=year,replicate=rep,reference=ref,**p))
            if (rep+1)%5==0:print(f'  {year} {rep+1}/{args.reps}: {time.perf_counter()-started:.1f}s',flush=True)
        pd.DataFrame(records).to_csv(out/f'fits_through_{year}.csv',index=False,float_format='%.17g')
        pd.DataFrame(pred).to_csv(out/f'forecasts_through_{year}.csv',index=False,float_format='%.17g')
    f=pd.DataFrame(records);p=pd.DataFrame(pred);base=pd.DataFrame(baseline)
    f.to_csv(out/'fits.csv',index=False,float_format='%.17g')
    p.to_csv(out/'forecasts.csv',index=False,float_format='%.17g')
    base.to_csv(out/'baseline_forecasts.csv',index=False,float_format='%.17g')
    # No failed replicates are silently removed; summaries state complete
    # replicate counts and are unavailable if failures make fitting incomplete.
    summary=[];signed=[]
    for key,g in p.groupby(['origin_date','book','report','tau']):
        x=g.pivot(index='replicate',columns='reference',values='es').reindex(columns=m.CCY)
        complete=x.dropna()
        bg=base[(base.origin_date==key[0])&(base.book==key[1])&(base.report==key[2])&(base.tau==key[3])].set_index('reference').es
        span=(complete.max(axis=1)-complete.min(axis=1))/complete.EUR
        observed=float((bg.max()-bg.min())/bg.EUR)
        summary.append(dict(origin_date=key[0],book=key[1],report=key[2],tau=key[3],
             requested_replicates=args.reps,complete_replicates=len(complete),original_reference_range=observed,
             perturbation_range_median=float(span.median()),perturbation_range_p025=float(span.quantile(.025)),
             perturbation_range_p975=float(span.quantile(.975)),conditional_sensitivity_not_true_ES_CI=True))
        logratio=np.log(complete.iloc[:,1:].to_numpy()/complete.iloc[:,[0]].to_numpy())
        for j,ref in enumerate(m.CCY[1:]):
            y=logratio[:,j]
            signed.append(dict(origin_date=key[0],book=key[1],report=key[2],tau=key[3],reference=ref,
               complete_replicates=len(y),original_log_ES_ratio=float(np.log(bg[ref]/bg.EUR)),
               perturbation_log_ratio_median=float(np.median(y)),
               perturbation_log_ratio_p025=float(np.quantile(y,.025)),perturbation_log_ratio_p975=float(np.quantile(y,.975)),
               conditional_sensitivity_not_true_ES_CI=True))
    pd.DataFrame(summary).to_csv(out/'range_sensitivity.csv',index=False,float_format='%.17g')
    pd.DataFrame(signed).to_csv(out/'signed_reference_sensitivity.csv',index=False,float_format='%.17g')
    expected=len(args.years)*args.reps*20
    assert len(f)==expected
    assert f.groupby(['year','replicate']).joint_indices_sha256.nunique().eq(1).all()
    manifest=dict(status='completed',years=args.years,reps_per_origin=args.reps,block=args.block,seed=SEED,
      origins=sorted(base.origin_date.unique()),expected_fit_attempts=expected,recorded_fit_attempts=len(f),
      failed_coordinate_fits=failures,all_failures_retained=True,
      original_known_window_used_for_filter_and_empirical_residuals=True,
      jointly_resampled_five_reference_estimation_windows=True,
      bootstraps_process_or_true_ES=False,exposed_historical_origins=True,
      runtime_seconds=time.perf_counter()-started,
      source_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),
      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      maximum_identity_error=float(p.cash_relative_identity_error.max()))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--tag',required=True);a.add_argument('--reps',type=int,default=5)
    a.add_argument('--block',type=int,default=20);a.add_argument('--years',type=int,nargs='+',default=[2018])
    run(a.parse_args())
