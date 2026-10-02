from project_config import ROOT as PUBLIC_WORK_ROOT
"""Fixed-clock cash forecasts over arbitrary common-reference holding periods.

This is an extension of an already-exposed historical specification, never an
untouched confirmatory holdout. Existing v17 files and immutable raw data stay
unchanged. Calculation outputs remain on D.
"""
from pathlib import Path
import hashlib,json,time
import fx_windows_platform_compat
import numpy as np
import pandas as pd
from numba import njit
import joint17_models as m

ROOT=PUBLIC_WORK_ROOT
REV=ROOT/'results/joint_fx_v18'
OLD=ROOT/'results/joint_fx_20261002'
RAW=(PUBLIC_WORK_ROOT/'data/raw/ecb_snapshot.csv')
PANEL=ROOT/'datasets/joint_fx_v18_temporal/ecb_joint_panel_through_20260930.csv'
SEED=20261002
BOOKS={**m.BOOKS,'financed_treasury':np.array([.5,.3,-.15,-.05])}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def freeze():
 out=REV/'temporal';out.mkdir(parents=True,exist_ok=True)
 p=out/'specification_freeze.json'
 if p.exists():return
 write_json(p,dict(recorded_utc='2026-10-02T06:55:28Z',stage='before new v18 calculations, after prior exposure',
  exposed_2026=True,not_preregistered=True,not_independent_holdout=True,
  label='specification-frozen temporal extension',currencies=m.CCY,raw_sha256=sha(RAW),
  raw_observations_not_yet_read_by_extension_script=True,temporal_origin_year=2026,
  temporal_last_record_date='2026-09-30',temporal_holding_intervals=1,
  temporal_books={k:v.tolist() for k,v in m.BOOKS.items()},reports=['EUR','USD'],taus=[.95,.99],
  temporal_models=['HS1250','IN_FHS1250 all five references','IN_FHS_REF_POOL','EWMA_GAUSS','EWMA_SELFNORMALIZED'],
  parameter_cutoff='last common reference observation of 2025; fit only last1250 prior returns',
  online_states='known innovations through forecast cutoff only; empirical shocks update with known observations',
  no_score_based_model_selection=True,no_result_based_origin_or_book_selection=True,
  temporal_monte_carlo_power=16,temporal_seed=SEED,temporal_bootstrap_blocks=[20,60],
  horizon_origins='all eligible 2018-2025 common reference dates',horizons=[5,20],
  horizon_books={k:v.tolist() for k,v in BOOKS.items()},horizon_models=['HS1250','IN_FHS1250 all five references','IN_FHS_REF_POOL'],
  future_start='A=t+1',future_endpoint='T=t+1+h',horizon_bootstrap_blocks=[20,60,120],
  primary_long_horizon_uncertainty_blocks=[60,120],bootstrap_draws=5000,
  interpretation='reference ranges and calibration persistence; no confirmatory superiority',
  annual_fits_2018_2025='reuse verified v17 prior-only annual fits unchanged',
  immutable_raw_preserved=True,output_drive='D'))

def panel_build():
 freeze();PANEL.parent.mkdir(parents=True,exist_ok=True)
 if not PANEL.exists():
  d=pd.read_csv(RAW,parse_dates=['TIME_PERIOD'])
  assert set(d.CURRENCY)==set(m.CCY[1:]) and d.CURRENCY_DENOM.eq('EUR').all()
  assert not d.duplicated(['CURRENCY','TIME_PERIOD']).any()
  good=d[d.OBS_STATUS.eq('A')&d.OBS_VALUE.gt(0)]
  wide=good.pivot(index='TIME_PERIOD',columns='CURRENCY',values='OBS_VALUE').sort_index()
  wide=wide.loc[:'2026-09-30',m.CCY[1:]].dropna()
  p=1/wide;p.insert(0,'EUR',1.);p.index.name='date'
  p.reset_index().to_csv(PANEL,index=False,float_format='%.17g')
  old=pd.read_csv(ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv',float_precision='round_trip')
  same=p.reset_index();same=same[same.date<=pd.Timestamp('2025-12-31')]
  assert len(same)==len(old)
  err=float(np.max(np.abs(same[m.CCY].to_numpy()-old[m.CCY].to_numpy())))
  assert err<1e-14,err
  write_json(PANEL.parent/'panel_manifest.json',dict(status='completed',raw_sha256=sha(RAW),panel_sha256=sha(PANEL),
   first_date=str(p.index[0].date()),last_date=str(p.index[-1].date()),dates=len(p),
   v17_prefix_error=err,snapshot_not_vintage=True,raw_unchanged=True,filter_status='A',positive_only=True))
 panel=pd.read_csv(PANEL,parse_dates=['date'],float_precision='round_trip')
 r=np.diff(np.log(panel[m.CCY[1:]].to_numpy(float)),axis=0)
 return panel,r

def annual_data(panel,r,year,output,force_fit=False):
 cutoff=int(np.flatnonzero(panel.date.dt.year.to_numpy()<year)[-1])
 if year<=2025 and not force_fit:
  source=OLD/f'scenario_v2/parameters_{year}.json';data=json.loads(source.read_text())
  assert data['cutoff']==str(panel.date.iloc[cutoff].date())
 else:
  members=[];logs=[];started=time.perf_counter()
  for ref in m.CCY:
   a,others=m.basis_matrix(ref);rb=100*r@a.T;fits=[]
   for j,i in enumerate(others):
    fit=m.fit_garch(rb[cutoff-1250:cutoff,j]);fits.append(fit)
    logs.append(dict(year=year,training_reference=ref,node=m.CCY[i],cutoff_date=str(panel.date.iloc[cutoff].date()),**fit))
   members.append(dict(reference=ref,nodes=[m.CCY[i] for i in others],garch=fits))
  data=dict(cutoff=str(panel.date.iloc[cutoff].date()),fits=members,runtime_seconds=time.perf_counter()-started)
  output.mkdir(parents=True,exist_ok=True)
  write_json(output/f'parameters_{year}.json',data)
  pd.DataFrame(logs).to_csv(output/f'garch_fit_log_{year}.csv',index=False,float_format='%.17g')
  source=output/f'parameters_{year}.json'
 out=[]
 for member in data['fits']:
  ref=member['reference'];a,others=m.basis_matrix(ref);rb=100*r@a.T;fits=member['garch'];hs=[];zz=[]
  for j,fit in enumerate(fits):
   tr=rb[cutoff-1250:cutoff,j]
   h=m.variance_path(rb[:,j],fit['omega'],fit['alpha'],fit['beta'],fit['mu'],float(np.var(tr)))
   eps=rb[:,j]-fit['mu'];hpost=fit['omega']+fit['alpha']*eps**2+fit['beta']*h
   z=eps/np.sqrt(hpost);assert np.max(z*z)<=1/fit['alpha']+1e-6
   hs.append(h);zz.append(z)
  pars=np.array([[f['mu'],f['omega'],f['alpha'],f['beta']] for f in fits])
  out.append((ref,others,pars,np.column_stack(hs),np.column_stack(zz)))
 return cutoff,out,source

@njit(cache=True)
def compressed_garch_blocks(z,hnext,pars,horizon):
 """All historical adjacent h+1-node innovations; retain start and hold sums."""
 n=len(z)-horizon;out=np.zeros((n,2,4))
 for s in range(n):
  for j in range(4):
   v=hnext[j];mu,o,a,b=pars[j]
   for k in range(horizon+1):
    shock=np.sqrt(v)*z[s+k,j];ret=mu+shock
    if k==0:out[s,0,j]=ret/100.
    else:out[s,1,j]+=ret/100.
    v=o+a*shock*shock+b*v
 return out

def normalized_window(z,t):
 a=z[t-1250:t];sd=a.std(axis=0)
 assert np.all(sd>0)
 return (a-a.mean(axis=0))/sd

def common_historical_blocks(r,t,horizon):
 a=r[t-1250:t];n=1250-horizon
 out=np.zeros((n,2,5));out[:,0,1:]=a[:n]
 for k in range(1,horizon+1):out[:,1,1:]+=a[k:k+n]
 return out

def cash_loss(paths,books):
 """r1=t->A; rh=A->T. Same algebra for any horizon. Fixed units, no carry."""
 r1=paths[:,0];rh=paths[:,1];res={};err=0.;relative=0.
 for book,w in books.items():
  ba=np.sum(w*np.exp(r1[:,1:]),axis=1)-w.sum()
  le=np.sum(w*(np.exp(r1[:,1:])-np.exp(r1[:,1:]+rh[:,1:])),axis=1)
  for report in ['EUR','USD']:
   n=m.CCY.index(report)
   ln=ba*np.exp(-r1[:,n])-(ba-le)*np.exp(-r1[:,n]-rh[:,n])
   delta=ln*np.exp(r1[:,n]+rh[:,n])-le-ba*np.expm1(rh[:,n])
   denom=np.maximum(1.,np.abs(ln*np.exp(r1[:,n]+rh[:,n]))+np.abs(le)+np.abs(ba*np.expm1(rh[:,n])))
   err=max(err,float(np.max(np.abs(delta))));relative=max(relative,float(np.max(np.abs(delta)/denom)))
   res[(book,report)]=ln
 assert relative<1e-12,(err,relative)
 return res,err,relative

def actual_tasks(panel,t,horizon,books):
 p=panel[m.CCY].to_numpy(float);N=100_000_000.;rows={}
 a=t+1;end=t+1+horizon
 for book,w in books.items():
  qty=np.r_[-N*w.sum(),N*w/p[t,1:]]
  va=qty@p[a];vt=qty@p[end]
  for report in ['EUR','USD']:
   n=m.CCY.index(report);na=N/p[t,n]
   actual=(va/p[a,n]-vt/p[end,n])/na
   rows[(book,report)]=dict(actual_loss=float(actual),reference_notional=float(na),
    quantity_EUR=float(qty[0]),**{'quantity_'+c:float(qty[j]) for j,c in enumerate(m.CCY[1:],1)},
    origin_date=str(panel.date.iloc[t].date()),start_date=str(panel.date.iloc[a].date()),
    target_date=str(panel.date.iloc[end].date()),book=book,report=report,horizon=horizon,
    holding_calendar_days=int((panel.date.iloc[end]-panel.date.iloc[a]).days))
 # Direct valuation and the compact formula independently agree.
 rs=np.diff(np.log(p[:,1:]),axis=0);paths=np.zeros((1,2,5))
 paths[0,0,1:]=rs[t];paths[0,1,1:]=rs[t+1:t+1+horizon].sum(axis=0)
 calc,_,_=cash_loss(paths,books)
 maximum=max(abs(rows[k]['actual_loss']-calc[k][0]) for k in rows)
 assert maximum<1e-12,maximum
 return rows,maximum

def forecasts_exact(panel,r,t,horizon,books,annual,cutoff):
 actual,realerr=actual_tasks(panel,t,horizon,books);records=[];pools={};maxidentity=0.;maxrelative=0.
 def add(model,ref,losses,error,relative):
  nonlocal maxidentity,maxrelative
  maxidentity=max(maxidentity,error);maxrelative=max(maxrelative,relative)
  for (book,report),v in losses.items():
   y=actual[(book,report)]['actual_loss']
   for tau in [.95,.99]:
    q,e=m.risk_pair(v,tau)
    records.append(dict(**actual[(book,report)],tau=tau,model=model,train_reference=ref,var=q,es=e,
     score=float(m.fz0(y,q,e,tau)),strict_hit=int(y>q),
     tail_residual=e-q-max(y-q,0)/(1-tau),scenario_n=len(v),
     fit_cutoff_date=str(panel.date.iloc[cutoff].date()),cash_identity_error=error,
     cash_identity_relative_error=relative,minimum_scenario_loss=float(v.min()),maximum_scenario_loss=float(v.max())))
 loss,err,rel=cash_loss(common_historical_blocks(r,t,horizon),books)
 add('HS1250','ALL_EQUIVARIANT',loss,err,rel)
 for ref,others,pars,h,z in annual:
  shocks=normalized_window(z,t);compressed=compressed_garch_blocks(shocks,h[t],pars,horizon)
  eur=m.to_eur(compressed,ref,others);loss,err,rel=cash_loss(eur,books)
  add('IN_FHS1250',ref,loss,err,rel)
  for k,v in loss.items():pools.setdefault(k,[]).append(v)
 add('IN_FHS_REF_POOL','ALL_FIVE',{k:np.concatenate(v) for k,v in pools.items()},maxidentity,maxrelative)
 return records,dict(realized_cash_formula_error=realerr,scenario_identity_error=maxidentity,
  scenario_relative_identity_error=maxrelative,actual=list(actual.values()))
