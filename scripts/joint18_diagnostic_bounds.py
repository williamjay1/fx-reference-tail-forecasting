from project_config import ROOT as PUBLIC_WORK_ROOT
"""Exact cash-Wasserstein bound for reference-specific ES differences."""
from pathlib import Path
import json,time,hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
import joint17_internal_scenarios as old

ROOT=PUBLIC_WORK_ROOT;OUT=ROOT/'results/joint_fx_v18/diagnostic_bounds'

def main():
 if OUT.exists():raise FileExistsError(OUT)
 OUT.mkdir(parents=True);t0=time.perf_counter();panel,r,_,_,_=old.prepare();records=[]
 for year in range(2018,2026):
  cutoff,annual=old.annual_data(panel,r,year,10)
  ids=np.flatnonzero((panel.date.dt.year.to_numpy()==year)&(np.arange(len(panel))<len(panel)-2))
  for kk,t in enumerate(ids):
   values={}
   for ref,others,fits,h,z,_,_ in annual:
    zz=old.internal_shocks(z[t-1250:t]);shock=np.stack([zz[:-1],zz[1:]],axis=1)
    x=m.to_eur(m.garch_paths(shock,h[t],fits),ref,others);loss,_=m.cash_losses(x)
    for key,v in loss.items():values.setdefault(key,[]).append(v)
   for (book,report),v in values.items():
    ordered=np.sort(np.array(v),axis=1)
    distance=max(float(np.mean(np.abs(ordered[g]-ordered[j]))) for g in range(5) for j in range(g+1,5))
    for tau in [.95,.99]:
     es=np.array([m.risk_pair(x,tau)[1] for x in v]);spread=float(np.ptp(es));avg=float(np.mean(np.abs(es)))
     rho=(1-tau)*spread/distance if distance>0 else 0.
     assert rho<=1+1e-10
     records.append(dict(origin_date=str(panel.date.iloc[t].date()),book=book,report=report,tau=tau,
      absolute_es_range=spread,mean_reference_es=avg,symmetric_relative_range=spread/avg,
      maximum_cash_wasserstein1=distance,es_lipschitz_bound=distance/(1-tau),tail_disagreement_ratio=rho))
   if kk%150==0:print(year,kk+1,len(ids),'elapsed',round(time.perf_counter()-t0,1),flush=True)
 d=pd.DataFrame(records);d.to_csv(OUT/'daily_diagnostics.csv',index=False,float_format='%.17g')
 summ=d.groupby(['book','report','tau']).agg(n=('origin_date','size'),median_symmetric_range=('symmetric_relative_range','median'),
  median_absolute_es_range=('absolute_es_range','median'),median_w1=('maximum_cash_wasserstein1','median'),
  median_tail_disagreement_ratio=('tail_disagreement_ratio','median'),maximum_tail_disagreement_ratio=('tail_disagreement_ratio','max')).reset_index()
 summ.to_csv(OUT/'summary.csv',index=False,float_format='%.17g')
 (OUT/'manifest.json').write_text(json.dumps(dict(status='completed',origins=d.origin_date.nunique(),records=len(d),
  runtime_seconds=time.perf_counter()-t0,exact_paths_per_reference=1249,
  bound_violations=int((d.tail_disagreement_ratio>1+1e-10).sum()),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
 print(summ.to_string(index=False),flush=True)

if __name__=='__main__':main()
