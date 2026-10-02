"""Calendar-stride integration checks for secondary covariance controls.

Exact IN-FHS main results do not have integration error. Precision checks here
do not select a model, change forecasts, or constitute an independent sample.
"""
import time,json
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
import joint17_internal_scenarios as ins
import joint18_temporal_common as c

def main():
 out=c.REV/'temporal';started=time.perf_counter();panel,r=c.panel_build()
 ids=np.flatnonzero((panel.date.dt.year.to_numpy()==2026)&(np.arange(len(panel))<len(panel)-2))[::6]
 c.write_json(out/'precision_specification.json',dict(recorded_before_precision_calculation=True,
  origin_selection='every sixth eligible2026 origin, starting first; calendar-only',n_origins=len(ids),
  powers=[14,16,18],models=['EWMA_GAUSS','EWMA_SELFNORMALIZED'],fixedseed=c.SEED+2026,
  main_exact_enumeration_unaffected=True,not_preregistered=True,not_model_selection=True))
 ewma=m.full_ewma_states(r);radii=np.sqrt(np.maximum(np.einsum('ij,ijk,ik->i',r,np.linalg.inv(ewma),r),0.))
 rows=[]
 for oi,t in enumerate(ids):
  for power in [14,16,18]:
   for name,p in [('EWMA_GAUSS',m.gaussian_paths(ewma[t-1],power,c.SEED+2026)),
                  ('EWMA_SELFNORMALIZED',ins.radial_unit_paths(ewma[t-1],radii[t-1250:t],power,c.SEED+2026)[0])]:
    eur=np.zeros((len(p),2,5));eur[:,:,1:]=p;loss,err,rel=c.cash_loss(eur,m.BOOKS)
    for (book,report),x in loss.items():
     for tau in [.95,.99]:
      q,e=m.risk_pair(x,tau);rows.append(dict(origin_date=str(panel.date.iloc[t].date()),model=name,
       power=power,book=book,report=report,tau=tau,var=q,es=e,identity_error=rel))
  if oi%8==0:print(f'precision {oi+1}/{len(ids)} {time.perf_counter()-started:.1f}s',flush=True)
 d=pd.DataFrame(rows);d.to_csv(out/'control_precision_predictions.csv',index=False,float_format='%.17g')
 a=d[d.power==16].merge(d[d.power==18],on=['origin_date','model','book','report','tau'],suffixes=('_16','_18'))
 checks=[]
 for (model,tau),g in a.groupby(['model','tau']):
  for metric in ['var','es']:
   change=np.abs(g[metric+'_18']-g[metric+'_16'])/np.abs(g[metric+'_18'])
   checks.append(dict(model=model,tau=float(tau),metric=metric,n=len(g),n_origins=len(ids),
    maximum_relative_change=float(change.max()),median_relative_change=float(change.median()),
    p90_relative_change=float(change.quantile(.9))))
 pd.DataFrame(checks).to_csv(out/'control_precision_summary.csv',index=False,float_format='%.17g')
 c.write_json(out/'control_precision_manifest.json',dict(status='completed',n_origins=len(ids),prediction_rows=len(d),
  primary_exact_model_has_no_integration_error=True,seed=c.SEED+2026,
  runtime_seconds=time.perf_counter()-started,prediction_sha256=c.sha(out/'control_precision_predictions.csv'),
  script_sha256=c.sha(__file__),not_a_uniform_error_bound=True,
  covariance_controls_remain_secondary=True))
 print(pd.DataFrame(checks).to_string(index=False),flush=True)

if __name__=='__main__':main()
