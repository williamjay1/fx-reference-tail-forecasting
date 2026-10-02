"""Execute audited future-start joint FX scenarios with annual prior-only fits."""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m

ROOT=Path('D:/MLWork/FXTailRisk');REV=ROOT/'results/joint_fx_20261002'
PANEL=ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv'
WINDOWS=(250,500,1250)
TAUS=(.95,.99)
SEED=20261002

def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def store_json(path,data):path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')

def run(args):
 out=REV/args.tag;out.mkdir(parents=True,exist_ok=True)
 predfile=out/'predictions.csv'
 if predfile.exists():raise FileExistsError('Use a new tag; completed outputs are retained')
 panel=pd.read_csv(PANEL,parse_dates=['date']);dates=panel.date.to_numpy();p=panel[m.CCY].to_numpy(float)
 r=np.diff(np.log(p[:,1:]),axis=0);ewma=m.full_ewma_states(r)
 # Before-observation covariance; initial 20 returns precede every origin used.
 before=np.concatenate([ewma[[0]],ewma[:-1]],axis=0)
 rad=np.sqrt(np.maximum(np.einsum('ij,ijk,ik->i',r,np.linalg.inv(before),r),0))
 tasks=pd.read_csv(REV/'data/outcome_tasks.csv',parse_dates=['origin_date','start_date','target_date'])
 target=tasks.set_index(['origin_date','book','report'])
 records=[];fitslog=[];start=time.perf_counter();max_identity=0.;max_identity_relative=0.;max_hs_basis=0.;max_cov_basis=0.
 for year in range(args.start_year,args.end_year+1):
  ids=np.flatnonzero((panel.date.dt.year.to_numpy()==year)&(np.arange(len(p))>=1251)&(np.arange(len(p))<len(p)-2))
  if args.pilot:
   # Development dates fixed by calendar spacing, never selected by scores.
   ids=ids[::max(1,len(ids)//24)][:24]
  if not len(ids):continue
  cutoff=int(np.flatnonzero(panel.date.dt.year.to_numpy()<year)[-1]);assert cutoff<ids.min()
  annual=[]
  print(f'year {year}: {len(ids)} origins, fit cutoff {panel.date.iloc[cutoff].date()}',flush=True)
  for ref in m.CCY:
   a,others=m.basis_matrix(ref);rb=100*(r@a.T);fits=[];hp=[];z=[]
   for j,i in enumerate(others):
    tr=rb[cutoff-1250:cutoff,j];fit=m.fit_garch(tr)
    # Initialization variance is prior-only, not variance of the full future array.
    h=m.variance_path(rb[:,j],fit['omega'],fit['alpha'],fit['beta'],fit['mu'],float(np.var(tr)))
    fits.append(fit);hp.append(h);z.append((rb[:,j]-fit['mu'])/np.sqrt(h))
    fitslog.append(dict(year=year,training_reference=ref,node=m.CCY[i],cutoff_date=str(panel.date.iloc[cutoff].date()),
      train_last_date=str(panel.date.iloc[cutoff].date()),**fit))
   h=np.column_stack(hp);z=np.column_stack(z);cops={};us={}
   for w in WINDOWS:
    c=m.copula_fit(z[cutoff-w:cutoff]);cops[w]=c
    us[w]=m.copula_uniforms(c,args.power,SEED+year)
   annual.append((ref,others,a,fits,h,z,cops,us))
  for oi,t in enumerate(ids):
   actual={(b,n):float(target.loc[(pd.Timestamp(dates[t]),b,n),'actual_loss']) for b in m.BOOKS for n in ['EUR','USD']}
   pools={'FHS_REF_POOL':{},'TCOP_REF_POOL':{}}
   def add(model,ref,losses,err,diag=None):
    nonlocal max_identity,max_identity_relative
    max_identity=max(max_identity,err)
    rel=0. if diag is None else diag['relative_identity_error']
    max_identity_relative=max(max_identity_relative,rel)
    for (book,report),values in losses.items():
     for tau in TAUS:
      q,e=m.risk_pair(values,tau);y=actual[(book,report)]
      records.append(dict(origin_date=str(panel.date.iloc[t].date()),start_date=str(panel.date.iloc[t+1].date()),
       target_date=str(panel.date.iloc[t+2].date()),book=book,report=report,tau=tau,model=model,
       train_reference=ref,actual_loss=y,var=q,es=e,score=float(m.fz0(y,q,e,tau)),strict_hit=int(y>q),
       tail_residual=e-q-max(y-q,0)/(1-tau),scenario_n=len(values),fit_cutoff_date=str(panel.date.iloc[cutoff].date()),
       cash_identity_absolute_error=err,cash_identity_relative_error=rel,
       minimum_scenario_loss=float(np.min(values)),maximum_scenario_loss=float(np.max(values))))
   # Closed-family controls; independent of training-coordinate selection.
   for w in [500,1250]:
    past=np.stack([r[t-w:t-1],r[t-w+1:t]],axis=1)
    eur=np.zeros((len(past),2,5));eur[:,:,1:]=past
    loss,err,diag=m.cash_losses(eur,diagnostics=True);add('HS'+str(w),'ALL_EQUIVARIANT',loss,err,diag)
    for ref,others,a,*_ in annual:
     recovered=m.to_eur(past@a.T,ref,others)
     max_hs_basis=max(max_hs_basis,float(np.max(np.abs(recovered-eur))))
   for name,paths4 in [('EWMA_GAUSS',m.gaussian_paths(ewma[t-1],args.power,SEED+year)),
                       ('EWMA_RADIAL',m.radial_paths(ewma[t-1],rad[t-1250:t],args.power,SEED+year))]:
    eur=np.zeros((len(paths4),2,5));eur[:,:,1:]=paths4
    loss,err,diag=m.cash_losses(eur,diagnostics=True);add(name,'ALL_EQUIVARIANT',loss,err,diag)
    for ref,others,a,*_ in annual:
     transformed=a@ewma[t-1]@a.T
     recovered=np.linalg.solve(a,np.linalg.solve(a,transformed.T).T)
     max_cov_basis=max(max_cov_basis,float(np.max(np.abs(recovered-ewma[t-1]))))
   for ref,others,a,fits,h,z,cops,us in annual:
    assert t>cutoff and t-1250>=0
    for w in WINDOWS:
     # Pair endpoints are at or before the current known observation t.
     zp=np.stack([z[t-w:t-1],z[t-w+1:t]],axis=1)
     pf=m.to_eur(m.garch_paths(zp,h[t],fits),ref,others)
     lf,err,diag=m.cash_losses(pf,diagnostics=True);add('FHS'+str(w),ref,lf,err,diag)
     zc=m.empirical_copula_shocks(z[t-w:t],us[w])
     pc=m.to_eur(m.garch_paths(zc,h[t],fits),ref,others)
     lc,err,diag=m.cash_losses(pc,diagnostics=True);add('TCOP'+str(w),ref,lc,err,diag)
     if w==1250:
      for label,loss in [('FHS_REF_POOL',lf),('TCOP_REF_POOL',lc)]:
       for key,value in loss.items():pools[label].setdefault(key,[]).append(value)
   for label,items in pools.items():
    # All five references have the same scenario count within a family.
    merged={key:np.concatenate(value) for key,value in items.items()};add(label,'ALL_FIVE',merged,0.)
   if oi%40==0:print(f'  {year} origin {oi+1}/{len(ids)} elapsed {time.perf_counter()-start:.1f}s',flush=True)
  # Annual checkpoint is distinct from the final all-year prediction record.
  pd.DataFrame(records).to_csv(out/f'predictions_through_{year}.csv.gz',index=False,float_format='%.17g',compression='gzip')
  params=[{'reference':ref,'nodes':[m.CCY[i] for i in others],'garch':fits,
    'copula':{str(w):{'nu':v['nu'],'loglik':v['loglik'],'cor':v['cor'].tolist()} for w,v in cops.items()}}
      for ref,others,a,fits,h,z,cops,us in annual]
  store_json(out/f'parameters_{year}.json',{'cutoff':str(panel.date.iloc[cutoff].date()),'fits':params})
 d=pd.DataFrame(records);d.to_csv(predfile,index=False,float_format='%.17g')
 pd.DataFrame(fitslog).to_csv(out/'garch_fit_log.csv',index=False,float_format='%.17g')
 summary=d.groupby(['tau','model','train_reference'],sort=False).agg(score=('score','mean'),
   breach=('strict_hit','mean'),mean_var=('var','mean'),mean_es=('es','mean'),n=('score','size')).reset_index()
 summary.to_csv(out/'summary.csv',index=False,float_format='%.17g')
 store_json(out/'manifest.json',dict(status='completed',pilot=args.pilot,years=[args.start_year,args.end_year],
  origins=d.origin_date.nunique(),predictions=len(d),actual_garch_fits=len(fitslog),scenario_power=args.power,
  runtime_seconds=time.perf_counter()-start,maximum_cash_identity_residual=max_identity,
  maximum_scale_adjusted_cash_identity_residual=max_identity_relative,
  maximum_common_HS_coordinate_error=max_hs_basis,maximum_full_covariance_coordinate_error=max_cov_basis,
  source_sha256=hashfile(PANEL),script_sha256=hashfile(Path(__file__)),model_script_sha256=hashfile(Path(m.__file__)),
  prediction_sha256=hashfile(predfile),seed=SEED,post_exposure=True,prior_only_annual_fit=True,
  copied_quantile_predictions=False,student_t_log_marginals=False,reporting_currency_risk_not_invariant=True))
 assert max_identity_relative<1e-12 and max_hs_basis<1e-12 and max_cov_basis<1e-12
 print(f'COMPLETE {len(d)} forecasts in {time.perf_counter()-start:.1f}s',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--start-year',type=int,default=2013);p.add_argument('--end-year',type=int,default=2025)
 p.add_argument('--tag',default='scenario');p.add_argument('--power',type=int,default=12);p.add_argument('--pilot',action='store_true')
 run(p.parse_args())
