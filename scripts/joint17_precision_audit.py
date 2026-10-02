from project_config import ROOT as PUBLIC_WORK_ROOT
"""Owned numerical/stress audit. Reuses saved parameters, never refits."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m

ROOT=PUBLIC_WORK_ROOT;REV=ROOT/'results/joint_fx_20261002'
OUT=REV/'math/precision'
DATES=['2015-01-13','2015-01-15','2015-01-16','2020-03-10','2020-03-17','2022-09-22',
       '2018-02-15','2019-08-15','2020-02-17','2021-08-16','2022-02-15','2023-08-15','2024-02-15','2025-08-15']
POWERS=(12,14,16);SEED=20261002

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def tail_max_share(x,tau,e):
    """Single largest scenario contribution to fractional tail integral."""
    mass=min(1/len(x),1-tau)
    return float(np.max(x)*mass/((1-tau)*e))

def summarize(year):
    risks=pd.read_csv(OUT/f'risks_{year}.csv')
    finite=risks[risks.power==0]
    mc=risks[risks.power>0]
    keys=['origin_date','book','report','tau','model','train_reference']
    base=mc[mc.power==12].set_index(keys)
    changes=[]
    for power in (14,16):
        other=mc[mc.power==power].set_index(keys)
        for key,row in base.iterrows():
            hi=other.loc[key]
            changes.append(dict(zip(keys,key),higher_power=power,var_difference=hi['var']-row['var'],
                es_difference=hi.es-row.es,score_difference=hi.score-row.score,
                var_relative_difference=(hi['var']-row['var'])/row['var'],es_relative_difference=(hi.es-row.es)/row.es))
    pd.DataFrame(changes).to_csv(OUT/f'changes_{year}.csv',index=False,float_format='%.17g')
    contrasts=[]
    for key,g in mc[mc.model=='TCOP1250'].groupby(['origin_date','book','report','tau']):
        values={power:g[g.power==power].set_index('train_reference') for power in POWERS}
        g12,g16=values[12],values[16]
        for ref in m.CCY[1:]:
            for metric in ('var','es','score'):
                lo=g12.loc[ref,metric]-g12.loc['EUR',metric]
                hi=g16.loc[ref,metric]-g16.loc['EUR',metric]
                contrasts.append(dict(zip(['origin_date','book','report','tau'],key),reference=ref,metric=metric,
                  contrast12=lo,contrast16=hi,contrast_change=hi-lo,sign_change=bool(lo*hi<0),
                  reference_spread12=float(g12[metric].max()-g12[metric].min()),
                  maximum_member_numerical_change=float((g16[metric]-g12[metric]).abs().max())))
    pd.DataFrame(contrasts).to_csv(OUT/f'contrasts_{year}.csv',index=False,float_format='%.17g')

def run(args):
    OUT.mkdir(parents=True,exist_ok=True)
    panel_path=ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv'
    panel=pd.read_csv(panel_path,parse_dates=['date']);prices=panel[m.CCY].to_numpy(float)
    dates=panel.date.to_numpy();r=np.diff(np.log(prices[:,1:]),axis=0)
    ewma=m.full_ewma_states(r);before=np.concatenate([ewma[[0]],ewma[:-1]],axis=0)
    radii=np.sqrt(np.maximum(np.einsum('ij,ijk,ik->i',r,np.linalg.inv(before),r),0))
    targets=pd.read_csv(REV/'data/outcome_tasks.csv',parse_dates=['origin_date']).set_index(['origin_date','book','report'])
    requests=[]
    for date in DATES:
        t=int(np.searchsorted(dates,np.datetime64(date)))
        if pd.Timestamp(dates[t]).year<=args.max_year:
            requests.append((pd.Timestamp(dates[t]).year,t,date))
    for year in sorted(set(x[0] for x in requests)):
        if (OUT/f'manifest_{year}.json').exists():
            print(f'SKIP completed precision {year}',flush=True);continue
        param_path=REV/'scenario_v2'/f'parameters_{year}.json'
        if not param_path.exists():
            print(f'PENDING annual parameters {year}',flush=True);continue
        params=json.loads(param_path.read_text(encoding='utf-8'))
        cutoff=int(np.searchsorted(dates,np.datetime64(params['cutoff'])))
        annual=[];started=time.perf_counter();rows=[];diagnostics=[]
        for member in params['fits']:
            ref=member['reference'];a,others=m.basis_matrix(ref);rb=100*(r@a.T)
            fits=member['garch'];h=[];z=[]
            for j,fit in enumerate(fits):
                train=rb[cutoff-1250:cutoff,j]
                variance=m.variance_path(rb[:,j],fit['omega'],fit['alpha'],fit['beta'],fit['mu'],float(np.var(train)))
                h.append(variance);z.append((rb[:,j]-fit['mu'])/np.sqrt(variance))
            cop=member['copula']['1250'];cop['cor']=np.asarray(cop['cor'])
            uniforms=m.copula_uniforms(cop,max(POWERS),SEED+year)
            annual.append((ref,others,fits,np.column_stack(h),np.column_stack(z),uniforms))
        def record(model,ref,power,paths,t,requested):
            # Default two-value API preserved in root model; do not alter it.
            losses,error,identity_diagnostics=m.cash_losses(paths,diagnostics=True)
            r1=paths[:,0,:];r2=paths[:,1,:]
            mag=float(np.max(np.abs(np.concatenate([r1,r1+r2],axis=1))))
            multiplier=float(np.max(np.exp(np.concatenate([r1,r1+r2],axis=1))))
            for (book,report),x in losses.items():
                y=float(targets.loc[(pd.Timestamp(dates[t]),book,report),'actual_loss'])
                for tau in (.95,.99):
                    q,e=m.risk_pair(x,tau)
                    rows.append(dict(origin_date=str(pd.Timestamp(dates[t]).date()),requested_origin=requested,
                     year=year,book=book,report=report,tau=tau,model=model,train_reference=ref,power=power,
                     scenario_n=len(x),actual_loss=y,var=q,es=e,score=float(m.fz0(y,q,e,tau)),
                     minimum_scenario_loss=float(np.min(x)),maximum_scenario_loss=float(np.max(x)),
                     maximum_loss_to_var=float(np.max(x)/q),maximum_single_scenario_ES_share=tail_max_share(x,tau,e),
                     max_abs_cash_identity_residual=error,max_relative_cash_identity_residual=identity_diagnostics['relative_identity_error'],
                     maximum_abs_log_increment_or_level=mag,maximum_price_multiplier=multiplier))
            return losses
        for _,t,requested in [x for x in requests if x[0]==year]:
            pools_fixed=[];pools_mc={p:[] for p in POWERS}
            for ref,others,fits,h,z,uniforms in annual:
                for w in (250,500,1250):
                    shocks=np.stack([z[t-w:t-1],z[t-w+1:t]],axis=1)
                    paths=m.to_eur(m.garch_paths(shocks,h[t],fits),ref,others)
                    record('FHS'+str(w),ref,0,paths,t,requested)
                    if w==1250:pools_fixed.append(paths)
                for power in POWERS:
                    shocks=m.empirical_copula_shocks(z[t-1250:t],uniforms[:2**power])
                    paths=m.to_eur(m.garch_paths(shocks,h[t],fits),ref,others)
                    record('TCOP1250',ref,power,paths,t,requested)
                    pools_mc[power].append(paths)
            record('FHS_REF_POOL','ALL_FIVE',0,np.concatenate(pools_fixed),t,requested)
            for power,paths in pools_mc.items():
                record('TCOP_REF_POOL','ALL_FIVE',power,np.concatenate(paths),t,requested)
                paths4=m.radial_paths(ewma[t-1],radii[t-1250:t],power,SEED+year)
                paths5=np.zeros((len(paths4),2,5));paths5[:,:,1:]=paths4
                record('EWMA_RADIAL','ALL_EQUIVARIANT',power,paths5,t,requested)
            for w in (500,1250):
                past=np.stack([r[t-w:t-1],r[t-w+1:t]],axis=1)
                full=np.zeros((len(past),2,5));full[:,:,1:]=past
                record('HS'+str(w),'ALL_EQUIVARIANT',0,full,t,requested)
            print(f'precision {year} {requested} complete elapsed {time.perf_counter()-started:.1f}s',flush=True)
        pd.DataFrame(rows).to_csv(OUT/f'risks_{year}.csv',index=False,float_format='%.17g')
        summarize(year)
        manifest={'status':'completed','year':year,'origins':[x[2] for x in requests if x[0]==year],
          'risk_rows':len(rows),'powers':POWERS,'seed':SEED+year,'parameter_sha256':sha(param_path),
          'model_script_sha256':sha(Path(m.__file__)),'panel_sha256':sha(panel_path),'no_parameter_refitting':True,
          'runtime_seconds':time.perf_counter()-started,'no_clipping':True,'role':'numerical/stress audit, post-exposure development'}
        (OUT/f'manifest_{year}.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(manifest),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--max-year',type=int,default=2025)
    run(p.parse_args())
