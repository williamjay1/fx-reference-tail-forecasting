"""Same foreign-currency exposures, with explicit zero EUR cash financing.

The pre-existing financed portfolio is retained. This is a distinct economic
position with cutoff value0.6N, so reporting currency has first-order risk.
"""
import time,json
from pathlib import Path
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
import joint18_temporal_common as c

WEIGHTS=np.array([.5,.3,-.15,-.05]);EUR_WEIGHT=0.;BOOK='unfinanced_treasury'

def cash_loss_eurweight(paths,weights,eur_weight):
 r1=paths[:,0];rh=paths[:,1];out={};err=0.;rel=0.
 ba=eur_weight+np.sum(weights*np.exp(r1[:,1:]),axis=1)
 le=np.sum(weights*(np.exp(r1[:,1:])-np.exp(r1[:,1:]+rh[:,1:])),axis=1)
 for report in ['EUR','USD']:
  n=m.CCY.index(report)
  ln=ba*np.exp(-r1[:,n])-(ba-le)*np.exp(-r1[:,n]-rh[:,n])
  delta=ln*np.exp(r1[:,n]+rh[:,n])-le-ba*np.expm1(rh[:,n])
  scale=np.maximum(1.,np.abs(ln*np.exp(r1[:,n]+rh[:,n]))+np.abs(le)+np.abs(ba*np.expm1(rh[:,n])))
  err=max(err,float(np.max(np.abs(delta))));rel=max(rel,float(np.max(np.abs(delta)/scale)))
  out[(BOOK,report)]=ln
 assert rel<1e-12,(err,rel)
 return out,err,rel

def actual(panel,t,horizon):
 p=panel[m.CCY].to_numpy(float);qty=np.r_[EUR_WEIGHT,WEIGHTS/p[t,1:]]
 va=qty@p[t+1];vt=qty@p[t+1+horizon];rows={}
 for report in ['EUR','USD']:
  n=m.CCY.index(report)
  y=((va/p[t+1,n])-(vt/p[t+1+horizon,n]))*p[t,n]
  rows[(BOOK,report)]=dict(origin_date=str(panel.date.iloc[t].date()),start_date=str(panel.date.iloc[t+1].date()),
   target_date=str(panel.date.iloc[t+1+horizon].date()),horizon=horizon,book=BOOK,report=report,actual_loss=float(y),
   reference_notional=100_000_000./p[t,n],quantity_EUR=100_000_000.*EUR_WEIGHT,
   **{'quantity_'+ccy:100_000_000.*qty[j] for j,ccy in enumerate(m.CCY[1:],1)},
   holding_calendar_days=int((panel.date.iloc[t+1+horizon]-panel.date.iloc[t+1]).days),
   cutoff_book_value_over_N=float(qty@p[t]))
 return rows

def year_data(panel,r,year,out):
 if year<=2025:return c.annual_data(panel,r,year,out)
 source=c.REV/'temporal/parameters_2026.json';data=json.loads(source.read_text())
 cutoff=int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(data['cutoff']))[0]);members=[]
 for member in data['fits']:
  ref=member['reference'];a,others=m.basis_matrix(ref);rb=100*r@a.T;hs=[];zs=[]
  for j,f in enumerate(member['garch']):
   tr=rb[cutoff-1250:cutoff,j]
   h=m.variance_path(rb[:,j],f['omega'],f['alpha'],f['beta'],f['mu'],float(np.var(tr)))
   eps=rb[:,j]-f['mu'];post=f['omega']+f['alpha']*eps**2+f['beta']*h
   hs.append(h);zs.append(eps/np.sqrt(post))
  pars=np.array([[f['mu'],f['omega'],f['alpha'],f['beta']] for f in member['garch']])
  members.append((ref,others,pars,np.column_stack(hs),np.column_stack(zs)))
 return cutoff,members,source

def forecasts(panel,r,t,horizon,annual,cutoff):
 target=actual(panel,t,horizon);records=[];parts={};maxerr=0.;maxrel=0.
 # Compare realized compact formula with independent direct quantities.
 realized=np.zeros((1,2,5));realized[0,0,1:]=r[t];realized[0,1,1:]=r[t+1:t+1+horizon].sum(axis=0)
 obs,_,_=cash_loss_eurweight(realized,WEIGHTS,EUR_WEIGHT)
 realerr=max(abs(obs[k][0]-target[k]['actual_loss']) for k in target);assert realerr<1e-12
 def add(model,ref,loss,err,rel):
  nonlocal maxerr,maxrel
  maxerr=max(maxerr,err);maxrel=max(maxrel,rel)
  for key,x in loss.items():
   y=target[key]['actual_loss']
   for tau in [.95,.99]:
    q,e=m.risk_pair(x,tau)
    records.append(dict(**target[key],tau=tau,model=model,train_reference=ref,var=q,es=e,
     score=float(m.fz0(y,q,e,tau)),strict_hit=int(y>q),tail_residual=e-q-max(y-q,0)/(1-tau),scenario_n=len(x),
     fit_cutoff_date=str(panel.date.iloc[cutoff].date()),cash_identity_error=err,cash_identity_relative_error=rel,
     minimum_scenario_loss=float(x.min()),maximum_scenario_loss=float(x.max())))
 loss,err,rel=cash_loss_eurweight(c.common_historical_blocks(r,t,horizon),WEIGHTS,EUR_WEIGHT)
 add('HS1250','ALL_EQUIVARIANT',loss,err,rel)
 for ref,others,pars,h,z in annual:
  compressed=c.compressed_garch_blocks(c.normalized_window(z,t),h[t],pars,horizon)
  eur=m.to_eur(compressed,ref,others);loss,err,rel=cash_loss_eurweight(eur,WEIGHTS,EUR_WEIGHT)
  add('IN_FHS1250',ref,loss,err,rel)
  for k,x in loss.items():parts.setdefault(k,[]).append(x)
 add('IN_FHS_REF_POOL','ALL_FIVE',{k:np.concatenate(x) for k,x in parts.items()},maxerr,maxrel)
 return records,dict(realized_formula_error=realerr,maximum_identity_error=maxerr,maximum_relative_identity_error=maxrel)

def main():
 panel,r=c.panel_build();started=time.perf_counter()
 for section,years,horizons in [('horizons',range(2018,2026),[1,5,20]),('temporal',[2026],[1])]:
  out=c.REV/section/'unfinanced_treasury';out.mkdir(parents=True,exist_ok=True)
  if (out/'predictions.csv').exists():raise FileExistsError('Preserve completed output')
  c.write_json(out/'extension_specification.json',dict(recorded_before_new_outcomes_evaluation=True,
   historical_exposure_acknowledged=True,confirmatory=False,
   foreign_values_at_cutoff_over_N=WEIGHTS.tolist(),eur_quantity_over_N=EUR_WEIGHT,cutoff_value_over_N=.6,
   financing_variant='none; distinct position from EUR-financed treasury',horizons=horizons,
   original_financed_outputs_retained=True,not_selected_by_score=True,
   risk='future-start value loss; no carry/interest/options or capital saving',years=list(years)))
  rows=[];checks=[];parameters={}
  for year in years:
   cutoff,annual,source=year_data(panel,r,year,out);parameters[str(year)]=c.sha(source)
   for horizon in horizons:
    ids=np.flatnonzero((panel.date.dt.year.to_numpy()==year)&(np.arange(len(panel))<len(panel)-1-horizon))
    if year<2026:ids=np.array([t for t in ids if panel.date.iloc[t+1+horizon]<=pd.Timestamp('2025-12-31')])
    for oi,t in enumerate(ids):
     rr,diag=forecasts(panel,r,t,horizon,annual,cutoff);rows.extend(rr)
     checks.append(dict(origin_date=str(panel.date.iloc[t].date()),horizon=horizon,**diag))
     if oi%160==0:print(f'unfinanced {section} {year} h={horizon} {oi+1}/{len(ids)} elapsed {time.perf_counter()-started:.1f}s',flush=True)
  d=pd.DataFrame(rows);d.to_csv(out/'predictions.csv',index=False,float_format='%.17g')
  pd.DataFrame(checks).to_csv(out/'identity_checks.csv',index=False,float_format='%.17g')
  c.write_json(out/'manifest.json',dict(status='completed',books=1,models=7,
   origin_counts={str(h):int(g.origin_date.nunique()) for h,g in d.groupby('horizon')},predictions=len(d),
   all_daily_origins=True,cutoff_position_value_over_N=.6,eur_quantity_zero=True,
   parameter_hashes=parameters,prediction_sha256=c.sha(out/'predictions.csv'),script_sha256=c.sha(__file__),
   maximum_realized_formula_error=max(x['realized_formula_error'] for x in checks),
   maximum_relative_identity_error=max(x['maximum_relative_identity_error'] for x in checks),
   snapshot_non_executable_prices=True,no_score_selection=True,confirmatory=False,
   runtime_seconds_cumulative=time.perf_counter()-started))
  print(section,json.dumps(json.loads((out/'manifest.json').read_text())),flush=True)

if __name__=='__main__':main()
