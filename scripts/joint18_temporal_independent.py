from project_config import ROOT as PUBLIC_WORK_ROOT
"""Independent vectorized path recursion and direct cash valuation checks.

Does not call forecast helpers, cash-loss algebra, risk-pair or scoring helpers.
Recorded fitted parameters and panel data are input; results are reconstructed.
"""
import json,math,time
from pathlib import Path
import fx_windows_platform_compat
import numpy as np
import pandas as pd

ROOT=PUBLIC_WORK_ROOT;REV=ROOT/'results/joint_fx_v18'
CCY=['EUR','USD','GBP','JPY','CHF']
BOOKS={'funded_assets':np.array([.25,.25,.25,.25]),'net_cashflows':np.array([.4,.1,-.3,-.2]),
       'financed_treasury':np.array([.5,.3,-.15,-.05])}
FIXED=['2018-01-02','2019-06-03','2020-03-16','2021-12-01','2022-09-22','2023-05-02','2024-08-01','2025-01-02']

def independent_cash(pnow,rstart,rhold,books):
 pa=pnow*np.exp(rstart);pt=pnow*np.exp(rstart+rhold);res={}
 for book,w in books.items():
  qty=np.r_[-w.sum(),w/pnow[1:]]
  va=pa@qty;vt=pt@qty
  for report in ['EUR','USD']:
   n=CCY.index(report)
   res[(book,report)]=(va/pa[:,n]-vt/pt[:,n])*pnow[n]
 return res

def risk(x,tau):
 ordered=np.sort(x);idx=math.ceil(len(x)*tau)-1;q=ordered[idx]
 # Tail integral: fractional mass at the VaR atom and full upper observations.
 e=(ordered[idx+1:].sum()+(len(x)*(1-tau)-(len(x)-idx-1))*q)/(len(x)*(1-tau))
 return q,e

def reconstruct(panel,t,h,parameters,books):
 p=panel[CCY].to_numpy(float);r=np.diff(np.log(p[:,1:]),axis=0);results={};pool={}
 past=r[t-1250:t];n=1250-h
 start=np.zeros((n,5));hold=np.zeros((n,5));start[:,1:]=past[:n]
 for k in range(1,h+1):hold[:,1:]+=past[k:k+n]
 results[('HS1250','ALL_EQUIVARIANT')]=independent_cash(p[t],start,hold,books)
 cutoff=int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(parameters['cutoff']))[0])
 for member in parameters['fits']:
  ref=member['reference'];others=[i for i in range(5) if CCY[i]!=ref];refidx=CCY.index(ref)
  # Compute direct log cross prices, never call basis_matrix/to_eur.
  cross=100*np.diff(np.log(p[:,others]/p[:,[refidx]]),axis=0)
  zz=[];hn=[]
  for j,f in enumerate(member['garch']):
   eps=cross[:,j]-f['mu'];variance=np.empty(t+1);variance[0]=np.var(cross[cutoff-1250:cutoff,j])
   for k in range(1,t+1):variance[k]=max(f['omega']+f['alpha']*eps[k-1]**2+f['beta']*variance[k-1],1e-10)
   post=f['omega']+f['alpha']*eps[:t]**2+f['beta']*variance[:t]
   z=eps[t-1250:t]/np.sqrt(post[t-1250:t]);z=(z-z.mean())/z.std()
   zz.append(z);hn.append(variance[t])
  shocks=np.column_stack(zz);v=np.broadcast_to(np.array(hn),(n,4)).copy();rs=np.zeros((n,4));rh=np.zeros((n,4))
  for k in range(h+1):
   innovations=np.sqrt(v)*shocks[k:k+n]
   mu=np.array([f['mu'] for f in member['garch']]);ret=mu+innovations
   if k==0:rs=ret/100
   else:rh+=ret/100
   omega=np.array([f['omega'] for f in member['garch']]);alpha=np.array([f['alpha'] for f in member['garch']]);beta=np.array([f['beta'] for f in member['garch']])
   v=omega+alpha*innovations**2+beta*v
  rstart=np.zeros((n,5));rhold=np.zeros((n,5));rstart[:,others]=rs;rhold[:,others]=rh
  rstart-=rstart[:,[0]];rhold-=rhold[:,[0]]
  loss=independent_cash(p[t],rstart,rhold,books);results[('IN_FHS1250',ref)]=loss
  for key,x in loss.items():pool.setdefault(key,[]).append(x)
 results[('IN_FHS_REF_POOL','ALL_FIVE')]={k:np.concatenate(v) for k,v in pool.items()}
 return results

def main():
 started=time.perf_counter();panel=pd.read_csv(ROOT/'datasets/joint_fx_v18_temporal/ecb_joint_panel_through_20260930.csv',parse_dates=['date'],float_precision='round_trip')
 checks=[];maximum={k:0. for k in ['var','es','actual_loss','score']}
 for section,dates,horizons,books in [('horizons',FIXED,[5,20],BOOKS),('temporal',['2026-01-02','2026-04-01','2026-08-03'],[1],{k:v for k,v in BOOKS.items() if k!='financed_treasury'})]:
  d=pd.read_csv(REV/section/'predictions.csv',float_precision='round_trip');d=d[d.origin_date.isin(dates)]
  for date in dates:
   t=int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(date))[0]);year=int(date[:4])
   param=(ROOT/f'results/joint_fx_20261002/scenario_v2/parameters_{year}.json') if year<2026 else REV/'temporal/parameters_2026.json'
   parameters=json.loads(param.read_text())
   for h in horizons:
    rec=reconstruct(panel,t,h,parameters,books)
    pp=panel[CCY].to_numpy(float);pa=pp[t+1];pt=pp[t+1+h]
    for (model,ref),loss in rec.items():
     for (book,report),x in loss.items():
      w=books[book];qty=np.r_[-w.sum(),w/pp[t,1:]];n=CCY.index(report)
      actual=((qty@pa)/pa[n]-(qty@pt)/pt[n])*pp[t,n]
      for tau in [.95,.99]:
       q,e=risk(x,tau);score=max(actual-q,0)/((1-tau)*e)+q/e+np.log(e)-1
       found=d[(d.origin_date==date)&(d.horizon==h)&(d.model==model)&(d.train_reference==ref)&(d.book==book)&(d.report==report)&(d.tau==tau)]
       assert len(found)==1,(section,date,h,model,ref,book,report,tau)
       row=found.iloc[0];errs={'var':abs(q-row['var']),'es':abs(e-row.es),'actual_loss':abs(actual-row.actual_loss),'score':abs(score-row.score)}
       for k,v in errs.items():maximum[k]=max(maximum[k],float(v))
       checks.append(dict(section=section,origin_date=date,horizon=h,model=model,reference=ref,book=book,report=report,tau=tau,**{k+'_error':v for k,v in errs.items()}))
  # Every emitted proper score and breach flag also independently recomputed.
  y=d.actual_loss.to_numpy();q=d['var'].to_numpy();e=d.es.to_numpy();tau=d.tau.to_numpy()
  scores=np.maximum(y-q,0)/((1-tau)*e)+q/e+np.log(e)-1
  assert np.max(np.abs(scores-d.score.to_numpy()))<1e-12
  assert np.array_equal((y>q).astype(int),d.strict_hit.to_numpy())
 assert max(maximum[k] for k in ['var','es','actual_loss'])<1e-11 and maximum['score']<1e-8,maximum
 pd.DataFrame(checks).to_csv(REV/'temporal/independent_reconstruction_checks.csv',index=False,float_format='%.17g')
 result=dict(status='passed',records_checked=len(checks),cash_valuation='actual quantities and numeraire prices; no compact loss formula',
  garch_recursion='plain Python/NumPy from recorded parameters, no compiled forecasting helper',
  es='sorted tail integral with fractional VaR atom, not hinge helper',maximum_errors=maximum,
  check_dates=dict(horizons=FIXED,temporal=['2026-01-02','2026-04-01','2026-08-03']),runtime_seconds=time.perf_counter()-started)
 (REV/'temporal/independent_reconstruction_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True)

if __name__=='__main__':main()
