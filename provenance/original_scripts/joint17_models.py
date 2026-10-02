"""Finite joint FX paths, reference-basis fits, and upper-tail pair scoring.

Student-t GARCH is a marginal filter likelihood, never an exponentiated
unbounded log-return distribution. t-copula uses bounded empirical margins.
"""
from __future__ import annotations
import math
import fx_windows_platform_compat
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit,gammaln,logit,stdtr,stdtrit
from scipy.stats import rankdata,qmc,norm,chi2

CCY=['EUR','USD','GBP','JPY','CHF']
BOOKS={'funded_assets':np.array([.25,.25,.25,.25]),'net_cashflows':np.array([.4,.1,-.3,-.2])}
DF_GRID=(4.,8.,16.,32.)

@njit(cache=True)
def variance_path(r,omega,alpha,beta,mu,h0=-1.):
 h=np.empty(len(r));h[0]=max(np.var(r) if h0<0 else h0,1e-8)
 for i in range(1,len(r)):h[i]=max(omega+alpha*(r[i-1]-mu)**2+beta*h[i-1],1e-10)
 return h

def unpack(p):
 mu=float(p[0]);omega=float(np.exp(p[1]));alpha=.999*float(expit(p[2]))
 beta=(.999-alpha)*float(expit(p[3]));nu=2.05+97.95*float(expit(p[4]))
 return mu,omega,alpha,beta,nu

def nll(p,r):
 mu,o,a,b,nu=unpack(p);h=variance_path(r,o,a,b,mu)
 c=gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(np.pi*(nu-2))
 return float(-np.mean(c-.5*np.log(h)-(nu+1)/2*np.log1p((r-mu)**2/(h*(nu-2)))))

def fit_garch(r):
 """MLE on percent-scale cash or FX observations; three fixed starts."""
 r=np.asarray(r,float);assert len(r)>=250 and np.isfinite(r).all()
 v=max(float(np.var(r)),1e-8);mu=float(np.mean(r));fits=[]
 bounds=[(mu-max(5*np.sqrt(v),.25),mu+max(5*np.sqrt(v),.25)),(np.log(v)-18,np.log(v)+8),(-9,9),(-9,9),(-9,9)]
 for a,b,nu in [(0.06,.90,8.),(.10,.82,10.),(.04,.93,6.)]:
  p0=np.array([mu,np.log(max(v*(1-a-b),1e-8)),logit(a/.999),logit(b/(.999-a)),logit((nu-2.05)/97.95)])
  z=minimize(nll,p0,args=(r,),method='L-BFGS-B',bounds=bounds,
    options={'maxiter':1500,'ftol':1e-11,'gtol':1e-6,'maxls':40})
  if z.success and np.isfinite(z.fun):fits.append(z)
 if not fits:raise RuntimeError('No converged GARCH fit among three starts')
 z=min(fits,key=lambda x:x.fun);mu,o,a,b,nu=unpack(z.x)
 return dict(mu=mu,omega=o,alpha=a,beta=b,nu=nu,nll=float(z.fun),iterations=int(z.nit),converged_starts=len(fits),train_n=len(r))

def basis_matrix(reference):
 ref=CCY.index(reference);others=[i for i in range(5) if i!=ref]
 a=np.zeros((4,4))
 for j,i in enumerate(others):
  if i:a[j,i-1]+=1
  if ref:a[j,ref-1]-=1
 return a,others

def to_eur(paths,reference,others):
 """r_i^EUR=(r_i^ref-r_EUR^ref); map whole two-node paths."""
 y=np.zeros((*paths.shape[:-1],5));y[...,others]=paths
 y=y-y[...,[0]]
 assert np.max(np.abs(y[...,0]))==0
 return y

def cash_losses(paths,diagnostics=False):
 """Two path increments -> four dimensionless future-start cash losses."""
 assert paths.shape[1:]==(2,5)
 r1=paths[:,0,:];r2=paths[:,1,:];out={};err=0.;relative=0.
 for book,w in BOOKS.items():
  ba=np.sum(w*np.exp(r1[:,1:]),axis=1)-w.sum()
  le=np.sum(w*(np.exp(r1[:,1:])-np.exp(r1[:,1:]+r2[:,1:])),axis=1)
  for report in ['EUR','USD']:
   n=CCY.index(report);ln=ba*np.exp(-r1[:,n])-(ba-le)*np.exp(-r1[:,n]-r2[:,n])
   residual=ln*np.exp(r1[:,n]+r2[:,n])-le-ba*np.expm1(r2[:,n])
   err=max(err,float(np.max(np.abs(residual))))
   scale=np.maximum(1.,np.abs(ln*np.exp(r1[:,n]+r2[:,n]))+np.abs(le)+np.abs(ba*np.expm1(r2[:,n])))
   relative=max(relative,float(np.max(np.abs(residual)/scale)))
   out[(book,report)]=ln
 assert relative<1e-12,(err,relative)
 if diagnostics:return out,err,{'relative_identity_error':relative}
 return out,err

def risk_pair(loss,tau,weights=None):
 x=np.asarray(loss,float);assert np.isfinite(x).all()
 if weights is None:
  # Inverse empirical CDF, with fractional tail mass in ES.
  q=float(np.partition(x,max(0,int(math.ceil(tau*len(x)))-1))[max(0,int(math.ceil(tau*len(x)))-1)])
  e=q+float(np.maximum(x-q,0).mean())/(1-tau)
 else:
  w=np.asarray(weights,float);assert np.all(w>=0) and abs(w.sum()-1)<1e-9
  ids=np.argsort(x);cs=np.cumsum(w[ids]);q=float(x[ids[min(np.searchsorted(cs,tau,side='left'),len(x)-1)]])
  e=q+float(w@np.maximum(x-q,0))/(1-tau)
 if not (np.isfinite(q) and np.isfinite(e) and e>=q-1e-12 and e>0):
  raise ValueError(f'FZ0 domain invalid q={q} e={e}')
 return q,e

def fz0(y,q,e,tau):
 y=np.asarray(y);q=np.asarray(q);e=np.asarray(e)
 if np.any(e<=0) or np.any(e<q-1e-10):raise ValueError('FZ0 domain')
 return np.maximum(y-q,0)/((1-tau)*e)+q/e+np.log(e)-1

def copula_fit(z):
 n,d=z.shape;u=np.column_stack([rankdata(z[:,j],method='average')/(n+1.) for j in range(d)])
 best=None
 for nu in DF_GRID:
  v=stdtrit(nu,u);cor=np.corrcoef(v,rowvar=False);cor=.98*cor+.02*np.eye(d)
  sign,ld=np.linalg.slogdet(cor);assert sign>0
  inv=np.linalg.inv(cor);quad=np.einsum('ij,jk,ik->i',v,inv,v)
  joint=gammaln((nu+d)/2)-gammaln(nu/2)-d/2*np.log(nu*np.pi)-ld/2-(nu+d)/2*np.log1p(quad/nu)
  margins=gammaln((nu+1)/2)-gammaln(nu/2)-.5*np.log(nu*np.pi)-(nu+1)/2*np.log1p(v*v/nu)
  ll=float(np.mean(joint-np.sum(margins,axis=1)))
  if best is None or ll>best['loglik']:best=dict(nu=nu,cor=cor,loglik=ll)
 return best

def copula_uniforms(fit,power,seed):
 # Independent future innovation steps with within-step tail dependence.
 u=qmc.Sobol(d=10,scramble=True,seed=seed).random_base2(power)
 u=np.clip(u,1e-10,1-1e-10);out=np.empty((len(u),2,4));chol=np.linalg.cholesky(fit['cor']);nu=fit['nu']
 for step in range(2):
  z=norm.ppf(u[:,step*5:step*5+4])@chol.T
  z=z/np.sqrt(chi2.ppf(u[:,step*5+4],nu)[:,None]/nu)
  out[:,step,:]=np.clip(stdtr(nu,z),1e-10,1-1e-10)
 return out

def empirical_copula_shocks(z,uniforms):
 out=np.empty_like(uniforms);n=len(z)
 for j in range(4):
  # Piecewise-linear empirical quantile; endpoints bounded by past shocks.
  ordered=np.sort(z[:,j]);pos=uniforms[:,:,j]*(n-1)
  lo=np.floor(pos).astype(int);hi=np.minimum(lo+1,n-1)
  out[:,:,j]=ordered[lo]+(ordered[hi]-ordered[lo])*(pos-lo)
 return out

def garch_paths(shocks,h_next,fits):
 out=np.empty_like(shocks)
 for j,p in enumerate(fits):
  mu=p['mu'];h=h_next[j];r1=mu+np.sqrt(h)*shocks[:,0,j]
  h2=p['omega']+p['alpha']*(r1-mu)**2+p['beta']*h
  r2=mu+np.sqrt(h2)*shocks[:,1,j]
  out[:,0,j]=r1/100;out[:,1,j]=r2/100
 return out

def full_ewma_states(returns):
 h=np.cov(returns[:20].T);out=np.empty((len(returns),4,4))
 for i,r in enumerate(returns):
  h=.94*h+.06*np.outer(r,r);out[i]=h
 return out

def gaussian_paths(cov,power,seed):
 u=np.clip(qmc.Sobol(8,scramble=True,seed=seed).random_base2(power),1e-10,1-1e-10)
 z=norm.ppf(u).reshape((-1,2,4));chol=np.linalg.cholesky(cov)
 return z@chol.T

def radial_paths(cov,radii,power,seed):
 """Bounded empirical radial, full-covariance elliptical control.

Historical Mahalanobis radii are invariant to invertible basis changes.
Finite radius support gives finite exponential cash moments.
"""
 u=np.clip(qmc.Sobol(10,scramble=True,seed=seed).random_base2(power),1e-10,1-1e-10)
 out=np.empty((len(u),2,4));chol=np.linalg.cholesky(cov)
 for k in range(2):
  z=norm.ppf(u[:,5*k:5*k+4]);z=z/np.linalg.norm(z,axis=1)[:,None]
  idx=np.minimum((u[:,5*k+4]*len(radii)).astype(int),len(radii)-1)
  out[:,k,:]=(z*radii[idx,None])@chol.T
 return out
