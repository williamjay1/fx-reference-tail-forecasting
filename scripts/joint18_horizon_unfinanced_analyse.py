"""Shared-date uncertainty for v18 temporal and longer-horizon extensions.

Conditional on fitted algorithms. Blocks preserve contemporaneous currency,
book and report dependence. 20-node blocks are a short-block sensitivity at
h20; primary longer-horizon uncertainty uses60 and120 nodes.
"""
import json,time
from pathlib import Path
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint18_temporal_common as c

B=5000

def hac(x,lags):
 x=np.asarray(x,float)
 if x.ndim==1:x=x[:,None]
 n=len(x);z=x-x.mean(axis=0);v=z.T@z/n
 for k in range(1,min(lags,n-1)+1):
  a=z[k:].T@z[:-k]/n;v+=(1-k/(lags+1))*(a+a.T)
 return v/n

def circular_means(x,block):
 x=np.asarray(x,float)
 if x.ndim==1:x=x[:,None]
 n,d=x.shape;rng=np.random.default_rng(c.SEED);nb=(n+block-1)//block
 sums=np.zeros((n,d))
 for k in range(block):sums+=np.roll(x,-k,axis=0)
 tail=n-(nb-1)*block;partial=np.zeros((n,d))
 for k in range(tail):partial+=np.roll(x,-k,axis=0)
 out=np.empty((B,d))
 for lo in range(0,B,100):
  size=min(100,B-lo);starts=rng.integers(0,n,size=(size,nb))
  out[lo:lo+size]=(sums[starts[:,:-1]].sum(axis=1)+partial[starts[:,-1]])/n
 return out

def contrast_summary(x,block,labels,meta):
 estimate=x.mean(axis=0);boot=circular_means(x,block);se=boot.std(axis=0,ddof=1)
 standardized=np.divide(boot-estimate,se,out=np.zeros_like(boot),where=se>0)
 critical=float(np.quantile(np.max(np.abs(standardized),axis=1),.95));rows=[]
 for j,label in enumerate(labels):
  rows.append(dict(**meta,contrast=label,block=block,delta=float(estimate[j]),
   point_low=float(np.quantile(boot[:,j],.025)),point_high=float(np.quantile(boot[:,j],.975)),
   simultaneous_low=float(estimate[j]-critical*se[j]),simultaneous_high=float(estimate[j]+critical*se[j]),
   family_size=len(labels),critical=critical))
 return rows

def analyse(section):
 started=time.perf_counter();out=c.REV/section/'unfinanced_treasury';source=out/'predictions.csv'
 d=pd.read_csv(source,parse_dates=['origin_date','start_date','target_date'],float_precision='round_trip')
 key=['origin_date','horizon','book','report','tau','model','train_reference']
 assert not d.duplicated(key).any()
 assert np.isfinite(d[['actual_loss','var','es','score']].to_numpy()).all()
 assert (d.es>0).all() and (d.es>=d['var']-1e-12).all()
 d['model_key']=d.model
 keep=d.model.eq('IN_FHS1250');d.loc[keep,'model_key']=d.loc[keep,'model']+'@'+d.loc[keep,'train_reference']
 target=['origin_date','horizon','book','report','tau']
 assert d.groupby(target).actual_loss.agg(lambda x:x.max()-x.min()).max()<1e-12
 assert d.groupby(target).target_date.nunique().max()==1
 tasks=d.groupby(['horizon','tau','model_key','book','report']).agg(n=('score','size'),
  mean_score=('score','mean'),breaches=('strict_hit','sum'),breach_rate=('strict_hit','mean'),
  mean_var=('var','mean'),mean_es=('es','mean'),median_es=('es','median')).reset_index()
 tasks.to_csv(out/'task_summary.csv',index=False,float_format='%.17g')
 summary=d.groupby(['horizon','tau','model_key']).agg(n_dates=('origin_date','nunique'),n_tasks=('score','size'),
  mean_score=('score','mean'),breach_rate=('strict_hit','mean'),mean_es=('es','mean'),median_es=('es','median')).reset_index()
 summary.to_csv(out/'model_summary.csv',index=False,float_format='%.17g')
 d.groupby([d.origin_date.dt.year,'horizon','tau','model_key']).agg(mean_score=('score','mean'),
  breach_rate=('strict_hit','mean'),mean_es=('es','mean')).reset_index().to_csv(out/'yearly_summary.csv',index=False,float_format='%.17g')
 contrasts=[];reference_contrasts=[];ranges=[];ranges_summary=[];cal=[];reporting=[]
 for (horizon,tau),g in d.groupby(['horizon','tau']):
  daily=g.groupby(['origin_date','model_key']).score.mean().unstack('model_key')
  assert not daily.isna().any().any()
  comparators=['IN_FHS1250@EUR','HS1250']
  if section=='temporal' and 'EWMA_GAUSS' in daily.columns:comparators+=['EWMA_GAUSS','EWMA_SELFNORMALIZED']
  delta=daily['IN_FHS_REF_POOL'].to_numpy()[:,None]-daily[comparators].to_numpy()
  blocks=[20,60] if section=='temporal' else [20,60,120]
  for block in blocks:
   contrasts+=contrast_summary(delta,block,comparators,dict(horizon=int(horizon),tau=float(tau),n_dates=len(daily),
    sign='mixture minus comparator; lower is better',block_primary=bool(section=='temporal' or block in [60,120])))
  gg=g[g.model.eq('IN_FHS1250')]
  for (book,report),z in gg.groupby(['book','report']):
   x=z.pivot(index='origin_date',columns='train_reference',values='es').sort_index()
   assert set(x.columns)==set(c.m.CCY) and x.notna().all().all()
   span=x.max(axis=1)-x.min(axis=1);rel=span/x.EUR
   dd=pd.DataFrame(dict(origin_date=x.index,absolute_range=span.to_numpy(),range_over_EUR=rel.to_numpy(),
    horizon=int(horizon),tau=float(tau),book=book,report=report));ranges.append(dd)
   entry=dict(horizon=int(horizon),tau=float(tau),book=book,report=report,n_dates=len(x),
    median_relative=float(rel.median()),p90_relative=float(rel.quantile(.9)),median_absolute=float(span.median()))
   for block in blocks:
    # Median uncertainty, actual indices shared across tasks by common seed.
    rng=np.random.default_rng(c.SEED);n=len(x);nb=(n+block-1)//block;med=[];a=rel.to_numpy()
    for lo in range(0,B,100):
     size=min(100,B-lo);starts=rng.integers(0,n,size=(size,nb))
     idx=(starts[:,:,None]+np.arange(block)).reshape(size,-1)[:,:n]%n
     med.extend(np.median(a[idx],axis=1).tolist())
    ranges_summary.append(dict(**entry,block=block,median_low=float(np.quantile(med,.025)),
     median_high=float(np.quantile(med,.975)),nonnegative_range_not_a_significance_test=True))
    pairs=[(a,b) for j,a in enumerate(c.m.CCY) for b in c.m.CCY[j+1:]]
    signed=np.column_stack([(x[a]-x[b]).to_numpy()/x.EUR.to_numpy() for a,b in pairs])
    reference_contrasts+=contrast_summary(signed,block,[a+' minus '+b for a,b in pairs],
     dict(horizon=int(horizon),tau=float(tau),book=book,report=report,n_dates=len(x),
      units='daily ES difference divided by same-date nativeEUR ES',not_causal_effect=True))
  for (model,book,report),z in g.groupby(['model_key','book','report']):
   z=z.sort_values('origin_date');y=z.actual_loss.to_numpy();q=z['var'].to_numpy();e=z.es.to_numpy()
   hit=(y>q).astype(float);u=(e-q-np.maximum(y-q,0)/(1-tau))/e
   for lag in blocks:
    hitse=float(np.sqrt(max(0,hac(hit,lag)[0,0])));use=float(np.sqrt(max(0,hac(u,lag)[0,0])))
    cal.append(dict(horizon=int(horizon),tau=float(tau),model_key=model,book=book,report=report,n=len(z),
     breaches=int(hit.sum()),nominal_expected=len(z)*(1-tau),lags=lag,breach_rate=float(hit.mean()),
     rate_low=float(hit.mean()-1.96*hitse),rate_high=float(hit.mean()+1.96*hitse),
     es_identification_mean=float(u.mean()),es_identification_low=float(u.mean()-1.96*use),
     es_identification_high=float(u.mean()+1.96*use),fewer_than_30_breaches=bool(hit.sum()<30),
     no_asymptotic_tail_pvalues=True,normal_intervals_descriptive=True))
  actual=g[g.model.eq('HS1250')][['origin_date','horizon','book','report','actual_loss']].drop_duplicates()
  for book,z in actual.groupby('book'):
   x=z.pivot(index='origin_date',columns='report',values='actual_loss');diff=x.USD-x.EUR
   reporting.append(dict(horizon=int(horizon),book=book,n_dates=len(x),
    sd_EUR=float(x.EUR.std()),sd_USD=float(x.USD.std()),mean_abs_reporting_difference=float(diff.abs().mean()),
    maximum_abs_reporting_difference=float(diff.abs().max()),tau=float(tau)))
 pd.DataFrame(contrasts).to_csv(out/'score_contrasts.csv',index=False,float_format='%.17g')
 pd.DataFrame(reference_contrasts).to_csv(out/'pairwise_reference_es_contrasts.csv',index=False,float_format='%.17g')
 pd.concat(ranges).to_csv(out/'reference_es_range_daily.csv',index=False,float_format='%.17g')
 pd.DataFrame(ranges_summary).to_csv(out/'reference_es_range_summary.csv',index=False,float_format='%.17g')
 pd.DataFrame(cal).to_csv(out/'calibration.csv',index=False,float_format='%.17g')
 pd.DataFrame(reporting).to_csv(out/'reporting_currency_effects.csv',index=False,float_format='%.17g')
 c.write_json(out/'analysis_manifest.json',dict(status='completed',source_sha256=c.sha(source),script_sha256=c.sha(__file__),
  bootstrap_draws=B,seed=c.SEED,blocks=[20,60] if section=='temporal' else [20,60,120],
  resampling_unit='common origin date, all currencies/books/reportstogether',
  bootstrap_refits_parameters=False,exposed_historical_data=True,confirmatory=False,
  score_is_joint_var_es_not_full_distribution=True,random_books_not_independent_investors=True,
  twenty_interval_hold_overlap='60/120 primary blocks;20 short-block sensitivity',
  temporal_small_sample='Only189 origins;99expected1.89breaches/task;unfinanced treasury exact systems only. No tail asymptotic pvalues.',
  runtime_seconds=time.perf_counter()-started,models=d.model_key.nunique(),origin_support_per_horizon={str(h):int(g.origin_date.nunique()) for h,g in d.groupby('horizon')}))
 print(section,summary.to_string(index=False),flush=True)
 print(pd.DataFrame(ranges_summary).query('block==60').to_string(index=False),flush=True)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('section',choices=['temporal','horizons']);a=p.parse_args();analyse(a.section)
