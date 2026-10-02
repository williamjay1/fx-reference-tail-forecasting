"""Covariance-targeted scalar BEKK with bounded internal elliptical shocks.

An established multivariate covariance recursion, not a new generic estimator.
The future empirical radial law is explicit, finite and coordinate closed.
"""
from pathlib import Path
import argparse,json,time,hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit,logit
from scipy.stats import qmc,norm
import joint17_models as m
from joint17_internal_scenarios import annual_data,internal_shocks

ROOT=Path('D:/MLWork/FXTailRisk');OLD=ROOT/'results/joint_fx_20261002'
OUT=ROOT/'results/joint_fx_v18/multivariate';SEED=20261003

@njit(cache=True)
def covariance_states(x,s,a,b):
 h=np.empty((len(x)+1,4,4));h[0]=s
 for k in range(len(x)):h[k+1]=(1-a-b)*s+a*np.outer(x[k],x[k])+b*h[k]
 return h

@njit(cache=True)
def objective_ab(x,s,a,b):
 h=s.copy();da=np.zeros((4,4));db=np.zeros((4,4));v=0.;ga=0.;gb=0.
 for k in range(len(x)):
  inv=np.linalg.inv(h);sign,ld=np.linalg.slogdet(h)
  if sign<=0:return 1e20,np.zeros(2)
  z=inv@x[k];v+=.5*(ld+x[k]@z)
  mat=inv-np.outer(z,z)
  ga+=.5*np.sum(mat*da);gb+=.5*np.sum(mat*db)
  old=h.copy();h=(1-a-b)*s+a*np.outer(x[k],x[k])+b*h
  da=-s+np.outer(x[k],x[k])+b*da;db=-s+old+b*db
 return v/len(x),np.array([ga,gb])/len(x)

def unpack(p):
 a=.999*expit(p[0]);b=(.999-a)*expit(p[1]);return float(a),float(b)

def objective(p,x,s):
 a,b=unpack(p);v,g=objective_ab(x,s,a,b)
 dap=a*(1-a/.999);dbp=b*(1-b/(.999-a))
 return float(v),np.array([dap*(g[0]-expit(p[1])*g[1]),dbp*g[1]])

def fit_joint(x):
 mu=x.mean(axis=0);eps=x-mu;s=eps.T@eps/len(eps);fits=[];attempts=[]
 for a,b in [(.06,.90),(.12,.80),(.03,.95)]:
  p=np.array([logit(a/.999),logit(b/(.999-a))])
  z=minimize(objective,p,args=(eps,s),jac=True,method='L-BFGS-B',bounds=[(-9,9)]*2,
   options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7,'maxls':40})
  attempts.append(dict(success=bool(z.success),nll=float(z.fun),message=str(z.message),iterations=int(z.nit)))
  if z.success and np.isfinite(z.fun):fits.append(z)
 if not fits:raise RuntimeError('All scalar BEKK starts failed')
 z=min(fits,key=lambda f:f.fun);a,b=unpack(z.x)
 return dict(alpha=a,beta=b,mu=mu.tolist(),target=s.tolist(),nll=float(z.fun),
  converged_starts=len(fits),attempts=attempts),eps,s

def units(power,seed):
 u=np.clip(qmc.Sobol(10,scramble=True,seed=seed).random_base2(power),1e-10,1-1e-10)
 v=[]
 for k in range(2):
  z=norm.ppf(u[:,5*k:5*k+4]);z/=np.linalg.norm(z,axis=1)[:,None]
  v.append((z,u[:,5*k+4]))
 return v

def paths(h,s,a,b,mu,radii,unit):
 rr=radii/np.sqrt(np.mean(radii*radii)/4.)
 z=[]
 for direction,u in unit:
  idx=np.minimum((u*len(rr)).astype(int),len(rr)-1);z.append(direction*rr[idx,None])
 e1=z[0]@np.linalg.cholesky(h).T
 h2=(1-a-b)*s[None,:,:]+b*h[None,:,:]+a*e1[:,:,None]*e1[:,None,:]
 e2=np.einsum('nij,nj->ni',np.linalg.cholesky(h2),z[1])
 eur=np.zeros((len(e1),2,5));eur[:,0,1:]=(mu+e1)/100.;eur[:,1,1:]=(mu+e2)/100.
 return eur,float(rr.max())

def closure_check(eps,s,a,b):
 base,_=objective_ab(eps,s,a,b);h=covariance_states(eps,s,a,b)
 rad=np.einsum('ij,ijk,ik->i',eps,np.linalg.inv(h[1:]),eps)
 records=[]
 for ref in m.CCY:
  A,_=m.basis_matrix(ref);xx=eps@A.T;ss=A@s@A.T
  hh=covariance_states(xx,ss,a,b);vv,_=objective_ab(xx,ss,a,b)
  sign,ld=np.linalg.slogdet(A);assert sign!=0
  scale=np.maximum(1.,np.abs(hh));target=np.einsum('ij,njk,lk->nil',A,h,A)
  residual=float(np.max(np.abs(hh-target)/scale))
  rr=np.einsum('ij,ijk,ik->i',xx,np.linalg.inv(hh[1:]),xx)
  records.append(dict(reference=ref,covariance_relative_error=residual,
   likelihood_offset_error=float(abs(vv-base-ld)),radius_squared_error=float(np.max(np.abs(rr-rad)))))
 assert max(q['covariance_relative_error'] for q in records)<1e-11
 assert max(q['likelihood_offset_error'] for q in records)<1e-9
 return records

def main(args):
 OUT.mkdir(parents=True,exist_ok=True);dest=OUT/args.tag
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir();t0=time.perf_counter()
 panel=pd.read_csv(ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv',parse_dates=['date'])
 r=np.diff(np.log(panel[m.CCY].to_numpy()[:,1:]),axis=0)
 tasks=pd.read_csv(OLD/'data/outcome_tasks.csv',parse_dates=['origin_date']).set_index(['origin_date','book','report'])
 rows=[];checks=[];pit=[];parameters=[];precision=[]
 for year in range(args.start_year,args.end_year+1):
  cutoff=int(np.flatnonzero(panel.date.dt.year.to_numpy()<year)[-1]);start=cutoff-1250
  tr=100*r[start:cutoff];fit,eps,s=fit_joint(tr);a,b=fit['alpha'],fit['beta'];mu=np.array(fit['mu'])
  # Analytic-gradient finite-difference audit and closure of the entire likelihood.
  pp=np.array([logit(a/.999),logit(b/(.999-a))]);val,grad=objective(pp,eps,s)
  numerical=np.array([(objective(pp+np.eye(2)[j]*1e-5,eps,s)[0]-objective(pp-np.eye(2)[j]*1e-5,eps,s)[0])/2e-5 for j in range(2)])
  assert np.max(np.abs(grad-numerical))<1e-5
  c=closure_check(eps,s,a,b)
  for q in c:q.update(year=year,gradient_absolute_error=float(np.max(np.abs(grad-numerical))))
  checks.extend(c)
  last=int(np.flatnonzero(panel.date.dt.year.to_numpy()==year)[-1]);xx=100*r[start:min(last+1,len(r))]-mu
  h=covariance_states(xx,s,a,b)
  radii=np.sqrt(np.maximum(np.einsum('ij,ijk,ik->i',xx,np.linalg.inv(h[1:]),xx),0.))
  assert np.max(radii*radii)<=1/a+1e-6
  ids=np.flatnonzero((panel.date.dt.year.to_numpy()==year)&(np.arange(len(panel))<len(panel)-2))
  calendar=set(ids[[0,len(ids)//2,-1]])
  if args.pilot:ids=ids[:4]
  unit=units(args.power,SEED+year);hiunit=units(args.power+2,SEED+year)
  parameters.append(dict(year=year,cutoff=str(panel.date.iloc[cutoff].date()),**fit))
  for k,t in enumerate(ids):
   rr=radii[t-1250-start:t-start];p,bound=paths(h[t-start],s,a,b,mu,rr,unit)
   loss,err,d=m.cash_losses(p,diagnostics=True)
   his=None
   if t in calendar:
    p2,_=paths(h[t-start],s,a,b,mu,rr,hiunit);his,_=m.cash_losses(p2)
   for (book,report),x in loss.items():
    y=float(tasks.loc[(panel.date.iloc[t],book,report),'actual_loss'])
    pit.append(dict(origin_date=str(panel.date.iloc[t].date()),book=book,report=report,pit=float(np.mean(x<=y)),
     below_support=int(y<float(x.min())),above_support=int(y>float(x.max())),model='SCALAR_BEKK_INTERNAL'))
    for tau in [.95,.99]:
     q,e=m.risk_pair(x,tau)
     rows.append(dict(origin_date=str(panel.date.iloc[t].date()),start_date=str(panel.date.iloc[t+1].date()),target_date=str(panel.date.iloc[t+2].date()),
      model='SCALAR_BEKK_INTERNAL',train_reference='ALL_EQUIVARIANT',book=book,report=report,tau=tau,actual_loss=y,var=q,es=e,
      score=float(m.fz0(y,q,e,tau)),strict_hit=int(y>q),scenario_n=len(x),fit_cutoff_date=str(panel.date.iloc[cutoff].date()),
      cash_identity_relative_error=d['relative_identity_error'],maximum_normalized_radius=bound))
     if his is not None:
      q2,e2=m.risk_pair(his[(book,report)],tau)
      precision.append(dict(origin_date=str(panel.date.iloc[t].date()),book=book,report=report,tau=tau,var_low=q,var_high=q2,
       es_low=e,es_high=e2,relative_es_change=(e2-e)/e2))
   if k%60==0:print(year,k+1,len(ids),'elapsed',round(time.perf_counter()-t0,1),flush=True)
 pd.DataFrame(rows).to_csv(dest/'predictions.csv',index=False,float_format='%.17g')
 pd.DataFrame(pit).to_csv(dest/'pit.csv',index=False,float_format='%.17g')
 pd.DataFrame(checks).to_csv(dest/'closure_checks.csv',index=False,float_format='%.17g')
 pd.DataFrame(precision).to_csv(dest/'precision.csv',index=False,float_format='%.17g')
 (dest/'parameters.json').write_text(json.dumps(parameters,indent=2)+'\n')
 manifest=dict(status='completed',origins=pd.DataFrame(rows).origin_date.nunique(),predictions=len(rows),power=args.power,
  runtime_seconds=time.perf_counter()-t0,pilot=args.pilot,seed=SEED,
  law='covariance-targeted scalar BEKK plus post-innovation internal empirical radial rescaled to mean squared radius4',
  references='joint law fitted in EUR and transformed; full likelihood/state closure numerically checked across allfivebasis',
  minimum_converged_starts=min(p['converged_starts'] for p in parameters),max_closure_error=max(c['covariance_relative_error'] for c in checks),
  max_likelihood_offset_error=max(c['likelihood_offset_error'] for c in checks),
  max_precision_relative_es_change=max(abs(p['relative_es_change']) for p in precision),
  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--power',type=int,default=16)
 ap.add_argument('--start-year',type=int,default=2018);ap.add_argument('--end-year',type=int,default=2025);ap.add_argument('--pilot',action='store_true')
 main(ap.parse_args())
