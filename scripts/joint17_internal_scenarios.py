from project_config import ROOT as PUBLIC_WORK_ROOT
"""Stress-driven redesign: internally standardized finite FX scenario laws.

Exploratory after engineering failure, not a retrospective preregistration.
Reuses actual prior-only GARCH estimates, changes the empirical innovation law.
"""
from pathlib import Path
import argparse,json,time,hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd
from scipy.stats import qmc,norm
import joint17_models as m

ROOT=PUBLIC_WORK_ROOT;REV=ROOT/'results/joint_fx_20261002'
PANEL=ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv'
FIXED_DATES=['2015-01-13','2015-01-15','2015-01-16','2018-05-02','2019-09-02',
 '2020-03-10','2020-03-17','2021-06-01','2022-09-22','2023-04-03','2024-08-01','2025-10-01']
SEED=20261002

def empirical_discrete(z,u):
 ordered=np.sort(z,axis=0);ids=np.minimum((u*len(z)).astype(int),len(z)-1)
 out=np.empty_like(u)
 for j in range(4):out[:,:,j]=ordered[ids[:,:,j],j]
 return out

def internal_shocks(z):
 mean=z.mean(axis=0);sd=z.std(axis=0)
 if not (sd>0).all():raise ValueError('Degenerate internal residual window')
 return (z-mean)/sd

def radial_unit_paths(cov,radii,power,seed):
 rescaled=radii/np.sqrt(np.mean(radii*radii)/4.)
 return m.radial_paths(cov,rescaled,power,seed),float(np.mean(radii*radii)/4.),float(rescaled.max())

def prepare():
 panel=pd.read_csv(PANEL,parse_dates=['date']);p=panel[m.CCY].to_numpy();r=np.diff(np.log(p[:,1:]),axis=0)
 ewma=m.full_ewma_states(r)
 # Post-innovation covariance, all historical innovation information is known.
 radius=np.sqrt(np.maximum(np.einsum('ij,ijk,ik->i',r,np.linalg.inv(ewma),r),0.))
 assert radius.max()**2<=1/.06+1e-6
 tasks=pd.read_csv(REV/'data/outcome_tasks.csv',parse_dates=['origin_date','start_date','target_date'])
 return panel,r,ewma,radius,tasks.set_index(['origin_date','book','report'])

def annual_data(panel,r,year,power):
 param=REV/f'scenario_v2/parameters_{year}.json'
 if not param.exists():raise FileNotFoundError(param)
 data=json.loads(param.read_text());cutoff=int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(data['cutoff']))[0])
 out=[]
 for member in data['fits']:
  ref=member['reference'];a,others=m.basis_matrix(ref);rb=100*r@a.T;fits=member['garch'];hs=[];zz=[]
  for j,fit in enumerate(fits):
   tr=rb[cutoff-1250:cutoff,j]
   h=m.variance_path(rb[:,j],fit['omega'],fit['alpha'],fit['beta'],fit['mu'],float(np.var(tr)))
   eps=rb[:,j]-fit['mu']
   h_post=fit['omega']+fit['alpha']*eps*eps+fit['beta']*h
   z=eps/np.sqrt(h_post)
   assert np.max(z*z)<=1/fit['alpha']+1e-6
   hs.append(h);zz.append(z)
  h=np.column_stack(hs);z=np.column_stack(zz)
  cop=m.copula_fit(internal_shocks(z[cutoff-1250:cutoff]));u=m.copula_uniforms(cop,power,SEED+year)
  out.append((ref,others,fits,h,z,cop,u))
 return cutoff,out

def forecasts_at(panel,r,ewma,radius,target,t,power,annual,cutoff):
 actual={(b,n):float(target.loc[(panel.date.iloc[t],b,n),'actual_loss']) for b in m.BOOKS for n in ['EUR','USD']}
 records=[];pools={'IN_FHS_REF_POOL':{},'IN_TCOP_REF_POOL':{}};diags=[]
 def add(name,ref,losses,identity,relative):
  for (book,report),x in losses.items():
   for tau in [.95,.99]:
    q,e=m.risk_pair(x,tau);y=actual[(book,report)]
    records.append(dict(origin_date=str(panel.date.iloc[t].date()),start_date=str(panel.date.iloc[t+1].date()),
     target_date=str(panel.date.iloc[t+2].date()),book=book,report=report,tau=tau,model=name,
     train_reference=ref,actual_loss=y,var=q,es=e,score=float(m.fz0(y,q,e,tau)),strict_hit=int(y>q),
     tail_residual=e-q-max(y-q,0)/(1-tau),scenario_n=len(x),fit_cutoff_date=str(panel.date.iloc[cutoff].date()),
     cash_identity_absolute_error=identity,cash_identity_relative_error=relative,
     minimum_scenario_loss=float(x.min()),maximum_scenario_loss=float(x.max())))
 def from_paths(name,ref,x):
  losses,err,diag=m.cash_losses(x,diagnostics=True);add(name,ref,losses,err,diag['relative_identity_error']);return losses
 # Emit reliable complete warmup for all six combination candidates.
 past=np.stack([r[t-1250:t-1],r[t-1250+1:t]],axis=1);eur=np.zeros((len(past),2,5));eur[:,:,1:]=past
 from_paths('HS1250','ALL_EQUIVARIANT',eur)
 gauss=m.gaussian_paths(ewma[t-1],power,SEED+int(panel.date.iloc[t].year))
 rad,scale,bound=radial_unit_paths(ewma[t-1],radius[t-1250:t],power,SEED+int(panel.date.iloc[t].year))
 for name,x in [('EWMA_GAUSS',gauss),('EWMA_SELFNORMALIZED',rad)]:
  eur=np.zeros((len(x),2,5));eur[:,:,1:]=x;from_paths(name,'ALL_EQUIVARIANT',eur)
 diags.append(dict(origin_date=str(panel.date.iloc[t].date()),radius_variance_scale=scale,
   normalized_radius_max=bound,raw_radius_squared_max=float(radius[t-1250:t].max()**2)))
 for ref,others,fits,h,z,cop,u in annual:
  normalized=internal_shocks(z[t-1250:t]);zp=np.stack([normalized[:-1],normalized[1:]],axis=1)
  diags.append(dict(origin_date=str(panel.date.iloc[t].date()),training_reference=ref,
   minimum_internal_residual_sd=float(z[t-1250:t].std(axis=0).min()),
   maximum_centered_standardized_shock=float(np.abs(normalized).max()),
   raw_internal_maximum=float(np.abs(z[t-1250:t]).max())))
  for name,shocks in [('IN_FHS1250',zp),('IN_TCOP1250',empirical_discrete(normalized,u))]:
   eur=m.to_eur(m.garch_paths(shocks,h[t],fits),ref,others);loss=from_paths(name,ref,eur)
   label='IN_FHS_REF_POOL' if name=='IN_FHS1250' else 'IN_TCOP_REF_POOL'
   for k,v in loss.items():pools[label].setdefault(k,[]).append(v)
 for name,parts in pools.items():add(name,'ALL_FIVE',{k:np.concatenate(v) for k,v in parts.items()},0.,0.)
 return records,diags

def main(args):
 out=REV/args.tag;out.mkdir(parents=True,exist_ok=True)
 if (out/'predictions.csv').exists():raise FileExistsError('Preserve outputs; use new tag')
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(m.__file__)]}
 panel,r,ewma,radius,target=prepare();records=[];diags=[];started=time.perf_counter()
 years=sorted(set(pd.Timestamp(x).year for x in FIXED_DATES)) if args.precision else range(args.start_year,args.end_year+1)
 for year in years:
  cutoff,annual=annual_data(panel,r,year,args.power)
  if args.precision:ids=np.flatnonzero(panel.date.isin(pd.to_datetime(FIXED_DATES)).to_numpy()&(panel.date.dt.year.to_numpy()==year))
  else:ids=np.flatnonzero((panel.date.dt.year.to_numpy()==year)&(np.arange(len(panel))<len(panel)-2))
  for k,t in enumerate(ids):
   rr,dd=forecasts_at(panel,r,ewma,radius,target,t,args.power,annual,cutoff);records.extend(rr);diags.extend(dd)
   if k%50==0:print(f'{year} {k+1}/{len(ids)} elapsed {time.perf_counter()-started:.1f}s',flush=True)
  pd.DataFrame(records).to_csv(out/f'checkpoint_{year}.csv.gz',index=False,float_format='%.17g')
  params=[dict(reference=ref,copula_df=cop['nu'],copula_corr=cop['cor'].tolist(),copula_loglik=cop['loglik']) for ref,_,_,_,_,cop,_ in annual]
  (out/f'copula_parameters_{year}.json').write_text(json.dumps(params,indent=2)+'\n')
 d=pd.DataFrame(records);d.to_csv(out/'predictions.csv',index=False,float_format='%.17g');pd.DataFrame(diags).to_csv(out/'radial_diagnostics.csv',index=False)
 manifest=dict(status='completed',precision=args.precision,scenario_power=args.power,origins=d.origin_date.nunique(),
   predictions=len(d),runtime_seconds=time.perf_counter()-started,source_hashes=hashes,
   maximum_identity_error=float(d.cash_identity_relative_error.max()),raw_radius_max=float(radius.max()),
   prediction_sha256=hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest(),
   post_stress_exploratory=True,original_data_clipped=False,parameters_reused=True,changed_innovation_law=True,
   copula_empirical_margins='discrete',seed=SEED)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--power',type=int,default=14)
 ap.add_argument('--precision',action='store_true');ap.add_argument('--start-year',type=int,default=2013)
 ap.add_argument('--end-year',type=int,default=2025);main(ap.parse_args())
