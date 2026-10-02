"""Exact-mixture PIT and expanded calendar-stratified numerical checks."""
from pathlib import Path
import argparse,json,time,hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
import joint17_internal_scenarios as old

ROOT=Path('D:/MLWork/FXTailRisk');OUT=ROOT/'results/joint_fx_v18/pit_precision'

def main(args):
 dest=OUT/args.tag
 if dest.exists():raise FileExistsError(dest)
 dest.mkdir(parents=True);t0=time.perf_counter()
 panel,r,ewma,radius,targets=old.prepare();rows=[];precision=[];lev=[]
 for year in range(2018,2026):
  cutoff,annual=old.annual_data(panel,r,year,16)
  ids=np.flatnonzero((panel.date.dt.year.to_numpy()==year)&(np.arange(len(panel))<len(panel)-2))
  # Monthly middle eligible origin, without conditioning on outcome or forecast.
  strata=[]
  for month in range(1,13):
   monthids=ids[panel.date.iloc[ids].dt.month.to_numpy()==month]
   if len(monthids):strata.append(int(monthids[len(monthids)//2]))
  selected=set(strata)
  if args.precision_only:ids=np.array(strata)
  highu={ref:m.copula_uniforms(cop,18,old.SEED+year) for ref,_,_,_,_,cop,_ in annual}
  for kk,t in enumerate(ids):
   parts={};native={}
   for ref,others,fits,h,z,cop,u in annual:
    normalized=old.internal_shocks(z[t-1250:t]);shock=np.stack([normalized[:-1],normalized[1:]],axis=1)
    eur=m.to_eur(m.garch_paths(shock,h[t],fits),ref,others);loss,_=m.cash_losses(eur)
    for k,v in loss.items():parts.setdefault(k,[]).append(v)
    if ref=='EUR':native=loss
    if t in selected:
     lo=m.to_eur(m.garch_paths(old.empirical_discrete(normalized,u),h[t],fits),ref,others)
     hi=m.to_eur(m.garch_paths(old.empirical_discrete(normalized,highu[ref]),h[t],fits),ref,others)
     ll,_=m.cash_losses(lo);hl,_=m.cash_losses(hi)
     for key in ll:
      for tau in [.95,.99]:
       q,e=m.risk_pair(ll[key],tau);qh,eh=m.risk_pair(hl[key],tau)
       precision.append(dict(origin_date=str(panel.date.iloc[t].date()),model='IN_TCOP1250',reference=ref,
        book=key[0],report=key[1],tau=tau,var_low=q,var_high=qh,es_low=e,es_high=eh,relative_es_change=(eh-e)/eh))
   if not args.precision_only:
    common=np.zeros((1249,2,5));common[:,0,1:]=r[t-1250:t-1];common[:,1,1:]=r[t-1249:t]
    hs,_=m.cash_losses(common)
    for model,losses in [('IN_FHS_REF_POOL',{k:np.concatenate(v) for k,v in parts.items()}),('IN_FHS1250_EUR',native),('HS1250',hs)]:
     for (book,report),x in losses.items():
      y=float(targets.loc[(panel.date.iloc[t],book,report),'actual_loss'])
      left=float(np.mean(x<y));right=float(np.mean(x<=y))
      rows.append(dict(origin_date=str(panel.date.iloc[t].date()),book=book,report=report,model=model,
       pit=.5*(left+right),cdf_left=left,cdf_right=right,below_support=int(y<x.min()),above_support=int(y>x.max())))
      for tau in [.95,.99]:
       q,e=m.risk_pair(x,tau);excess=np.maximum(x-q,0)
       concentration=float(excess.max()/excess.sum()) if excess.sum()>0 else 0.
       lev.append(dict(origin_date=str(panel.date.iloc[t].date()),book=book,report=report,model=model,tau=tau,
        maximum_scenario_loss=float(x.max()),var=q,es=e,largest_tail_excess_share=concentration,scenario_n=len(x)))
   if kk%100==0:print(year,kk+1,len(ids),'elapsed',round(time.perf_counter()-t0,1),flush=True)
 for name,values in [('pit',rows),('precision',precision),('tail_leverage',lev)]:
  pd.DataFrame(values).to_csv(dest/f'{name}.csv',index=False,float_format='%.17g')
 manifest=dict(status='completed',precision_origins=pd.DataFrame(precision).origin_date.nunique(),precision_rows=len(precision),
  pit_origins=pd.DataFrame(rows).origin_date.nunique() if rows else 0,pit_rows=len(rows),runtime_seconds=time.perf_counter()-t0,
  origin_selection='middle eligible forecast date of every calendar month2018–2025',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  max_native_tcop_relative_es_change=max(abs(x['relative_es_change']) for x in precision),
  pit_definition='mid distribution function at observed cashloss; realized losses are not tied with model atoms',
  exact_mixture_paths=6245,numerical_sizes=[65536,262144],inference='exploratory conditional on frozen recorded specs')
 (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--precision-only',action='store_true');main(ap.parse_args())
