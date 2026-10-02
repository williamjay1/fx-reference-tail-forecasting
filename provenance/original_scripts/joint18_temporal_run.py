"""Specification-frozen 2026 extension, after acknowledged historical exposure."""
import argparse,time,json
from pathlib import Path
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
import joint17_internal_scenarios as ins
import joint18_temporal_common as c

def main(args):
 out=c.REV/'temporal';out.mkdir(parents=True,exist_ok=True)
 if (out/'predictions.csv').exists():raise FileExistsError('Keep completed outputs')
 panel,r=c.panel_build();started=time.perf_counter()
 cutoff,annual,param=c.annual_data(panel,r,2026,out)
 ewma=m.full_ewma_states(r)
 radius=np.sqrt(np.maximum(np.einsum('ij,ijk,ik->i',r,np.linalg.inv(ewma),r),0.))
 assert radius.max()**2<=1/.06+1e-6
 ids=np.flatnonzero((panel.date.dt.year.to_numpy()==2026)&(np.arange(len(panel))<len(panel)-2))
 records=[];checks=[]
 for oi,t in enumerate(ids):
  rows,diag=c.forecasts_exact(panel,r,t,1,m.BOOKS,annual,cutoff)
  actual={(x['book'],x['report']):x for x in diag.pop('actual')}
  for name,paths in [('EWMA_GAUSS',m.gaussian_paths(ewma[t-1],16,c.SEED+2026)),
                     ('EWMA_SELFNORMALIZED',ins.radial_unit_paths(ewma[t-1],radius[t-1250:t],16,c.SEED+2026)[0])]:
   eur=np.zeros((len(paths),2,5));eur[:,:,1:]=paths;loss,err,rel=c.cash_loss(eur,m.BOOKS)
   diag['scenario_identity_error']=max(diag['scenario_identity_error'],err)
   diag['scenario_relative_identity_error']=max(diag['scenario_relative_identity_error'],rel)
   for k,x in loss.items():
    y=actual[k]['actual_loss']
    for tau in [.95,.99]:
     q,e=m.risk_pair(x,tau)
     rows.append(dict(**actual[k],tau=tau,model=name,train_reference='ALL_EQUIVARIANT',var=q,es=e,
      score=float(m.fz0(y,q,e,tau)),strict_hit=int(y>q),tail_residual=e-q-max(y-q,0)/(1-tau),
      scenario_n=len(x),fit_cutoff_date=str(panel.date.iloc[cutoff].date()),cash_identity_error=err,
      cash_identity_relative_error=rel,minimum_scenario_loss=float(x.min()),maximum_scenario_loss=float(x.max())))
  records.extend(rows);checks.append(dict(origin_date=str(panel.date.iloc[t].date()),**diag))
  if oi%25==0:print(f'temporal {oi+1}/{len(ids)} elapsed {time.perf_counter()-started:.2f}s',flush=True)
 d=pd.DataFrame(records);d.to_csv(out/'predictions.csv',index=False,float_format='%.17g')
 pd.DataFrame(checks).to_csv(out/'identity_checks.csv',index=False,float_format='%.17g')
 c.write_json(out/'manifest.json',dict(status='completed',first_origin=d.origin_date.min(),last_origin=d.origin_date.max(),
  last_target=d.target_date.max(),origins=d.origin_date.nunique(),predictions=len(d),models=9,
  parameter_cutoff=str(panel.date.iloc[cutoff].date()),parameter_sha256=c.sha(param),
  frozen_specification_sha256=c.sha(out/'specification_freeze.json'),
  snapshot_panel_sha256=c.sha(c.PANEL),script_sha256=c.sha(__file__),helper_sha256=c.sha(c.__file__),
  runtime_seconds=time.perf_counter()-started,prediction_sha256=c.sha(out/'predictions.csv'),
  known_prior_exposure=True,label='specification-frozen temporal extension',confirmatory=False,
  actual_parameters_fixed_through_2026=True,state_updates_use_known_observations_only=True,
  maximum_realized_formula_error=max(x['realized_cash_formula_error'] for x in checks),
  maximum_scenario_identity_error=max(x['scenario_identity_error'] for x in checks),
  maximum_scenario_relative_identity_error=max(x['scenario_relative_identity_error'] for x in checks),
  monte_carlo_controls_power=16,seed=c.SEED,ordinary_fhs_not_rerun=True,
  no_score_based_configuration_selection=True))
 print(json.dumps(json.loads((out/'manifest.json').read_text())),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();main(parser.parse_args())
