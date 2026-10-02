"""Date-clustered financial evaluation. No performance-based configuration choice."""
from pathlib import Path
import json,hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd
from scipy.stats import chi2

ROOT=Path('D:/MLWork/FXTailRisk');REV=ROOT/'results/joint_fx_20261002'
OUT=REV/'analysis';OUT.mkdir(parents=True,exist_ok=True)
PRIMARY=['IN_FHS1250@EUR','FHS1250@EUR','EWMA_GAUSS','EWMA_SELFNORMALIZED','TAYLOR_OPT']
SEED=20261002;B=5000

def hac(x,lags=20):
 x=np.asarray(x,float)
 if x.ndim==1:x=x[:,None]
 n=len(x);z=x-x.mean(axis=0);v=z.T@z/n
 for k in range(1,min(lags,n-1)+1):
  a=z[k:].T@z[:-k]/n;v+=(1-k/(lags+1))*(a+a.T)
 return v/n

def wald(x,lags=20):
 cov=hac(x,lags);mu=np.mean(x,axis=0);rank=int(np.linalg.matrix_rank(cov))
 if rank!=len(mu):return np.nan,np.nan,rank
 stat=float(mu@np.linalg.solve(cov,mu));return stat,float(chi2.sf(stat,rank)),rank

def circular_means(x,block,seed=SEED):
 """Fixed-length circular blocks; shared indices across every contrast/task."""
 x=np.asarray(x,float)
 if x.ndim==1:x=x[:,None]
 n,d=x.shape;rng=np.random.default_rng(seed);nb=(n+block-1)//block
 # The last block is truncated to exactly n observations.
 sums=np.zeros((n,d))
 for k in range(block):sums+=np.roll(x,-k,axis=0)
 tail=n-(nb-1)*block
 partial=np.zeros((n,d))
 for k in range(tail):partial+=np.roll(x,-k,axis=0)
 out=np.empty((B,d))
 for lo in range(0,B,100):
  size=min(100,B-lo);starts=rng.integers(0,n,size=(size,nb))
  out[lo:lo+size]=(sums[starts[:,:-1]].sum(axis=1)+partial[starts[:,-1]])/n
 return out

def read_all():
 sources=[REV/'internal_full/predictions.csv',REV/'scenario_v2/predictions.csv',
          REV/'direct/warmup_2013_2017/predictions.csv',REV/'direct/full_prior_aligned/predictions.csv',
          REV/'combination/full_reliable_v2/predictions.csv']
 records=[]
 for source in sources:
  if not source.exists():raise FileNotFoundError(source)
  d=pd.read_csv(source,parse_dates=['origin_date','start_date','target_date'],float_precision='round_trip');d['source_file']=str(source)
  if source==REV/'scenario_v2/predictions.csv':
   # Unresolved ordinary-copula/radial MC results are failure diagnostics only.
   d=d[d.model.str.match(r'^FHS\d+$')|(d.model=='HS500')].copy()
  if '/direct/' in str(source).replace('\\','/'):
   d['tail_residual']=d.es-d['var']-np.maximum(d.actual_loss-d['var'],0)/(1-d.tau)
  if 'tail_residual' not in d:d['tail_residual']=d.es-d['var']-np.maximum(d.actual_loss-d['var'],0)/(1-d.tau)
  if 'train_reference' not in d:d['train_reference']='not_applicable'
  d['model_key']=d.model
  mask=d.model.str.match(r'^(IN_)?(FHS|TCOP)\d+$')
  d.loc[mask,'model_key']=d.loc[mask,'model']+'@'+d.loc[mask,'train_reference']
  records.append(d)
 d=pd.concat(records,ignore_index=True)
 key=['origin_date','book','report','tau','model_key']
 if d.duplicated(key).any():raise ValueError('Duplicate forecast keys')
 ev=d[d.origin_date.dt.year.between(2018,2025)].copy()
 if ev[['var','es','score','actual_loss']].isna().any().any():raise ValueError('Missing mandatory prediction')
 if not (ev.es>0).all() or not (ev.es>=ev['var']-1e-10).all():raise ValueError('Scoring domain')
 task=['origin_date','book','report','tau']
 if ev.groupby(task).actual_loss.agg(lambda v:v.max()-v.min()).max()>1e-12:raise ValueError('Target mismatch')
 if not ev.groupby(['origin_date','tau','model_key']).size().eq(4).all():raise ValueError('Each date/model must contain exactly four cash tasks')
 if ev.groupby(task).start_date.nunique().max()!=1 or ev.groupby(task).target_date.nunique().max()!=1:raise ValueError('Clock mismatch')
 return d,ev,sources

def main():
 d,ev,sources=read_all()
 daily=ev.groupby(['origin_date','tau','model_key']).score.mean().unstack('model_key')
 if daily.isna().any().any():raise ValueError('Unequal date support across comparators')
 summaries=ev.groupby(['tau','model_key']).agg(n_dates=('origin_date','nunique'),n_tasks=('score','size'),
   mean_score=('score','mean'),breach_rate=('strict_hit','mean'),mean_var=('var','mean'),
   mean_es=('es','mean'),median_es=('es','median')).reset_index()
 summaries['rank']=summaries.groupby('tau').mean_score.rank(method='min')
 summaries.to_csv(OUT/'model_summary.csv',index=False)
 component=ev.groupby(['tau','model_key','book','report']).agg(n=('score','size'),mean_score=('score','mean'),
  breaches=('strict_hit','sum'),breach_rate=('strict_hit','mean'),mean_var=('var','mean'),mean_es=('es','mean')).reset_index()
 component.to_csv(OUT/'task_summary.csv',index=False)
 yearly=ev.groupby([ev.origin_date.dt.year,'tau','model_key']).agg(score=('score','mean'),
  breach_rate=('strict_hit','mean'),mean_es=('es','mean')).reset_index()
 yearly.to_csv(OUT/'yearly_summary.csv',index=False)
 contrast=[]
 for tau in [.95,.99]:
  a=daily.xs(tau,level='tau');delta=a['IN_FHS_REF_POOL'].to_numpy()[:,None]-a[PRIMARY].to_numpy()
  estimate=delta.mean(axis=0)
  for block in [20,60]:
   boot=circular_means(delta,block);se=boot.std(axis=0,ddof=1)
   if not np.all(se>0):raise ValueError('Degenerate primary bootstrap contrast')
   critical=float(np.quantile(np.max(np.abs((boot-estimate)/se),axis=1),.95))
   for j,comp in enumerate(PRIMARY):
    contrast.append(dict(tau=tau,comparator=comp,block=block,delta=float(estimate[j]),
     point_low=float(np.quantile(boot[:,j],.025)),point_high=float(np.quantile(boot[:,j],.975)),
     simultaneous_low=float(estimate[j]-critical*se[j]),simultaneous_high=float(estimate[j]+critical*se[j]),
     family_size=len(PRIMARY),critical=critical))
 pd.DataFrame(contrast).to_csv(OUT/'primary_score_contrasts.csv',index=False)
 # All component references retained, never select the worst as sole comparator.
 sensitivity=[];long=[]
 sc=ev[ev.model.str.match(r'^(IN_)?(FHS|TCOP)\d+$')]
 for (model,tau),g in sc.groupby(['model','tau']):
  idx=['origin_date','book','report']
  for metric in ['var','es']:
   x=g.pivot(index=idx,columns='train_reference',values=metric)
   span=x.max(axis=1)-x.min(axis=1);relative=span/x['EUR'].abs()
   ratio=x.max(axis=1)/x.min(axis=1)
   r=pd.DataFrame({'absolute_range':span,'range_over_EUR':relative,'max_over_min':ratio}).reset_index()
   r['model']=model;r['tau']=tau;r['metric']=metric;long.append(r)
   for (book,report),z in r.groupby(['book','report']):
    sensitivity.append(dict(model=model,tau=tau,metric=metric,book=book,report=report,n=len(z),
      median_relative=float(z.range_over_EUR.median()),p90_relative=float(z.range_over_EUR.quantile(.90)),
      p99_relative=float(z.range_over_EUR.quantile(.99)),median_ratio=float(z.max_over_min.median()),
      median_absolute=float(z.absolute_range.median())))
 pd.DataFrame(sensitivity).to_csv(OUT/'reference_sensitivity_summary.csv',index=False)
 pd.concat(long).to_csv(OUT/'reference_sensitivity_daily.csv',index=False)
 # Formal calibration is marginal per predefined task, not pooled independent investors.
 calibration=[]
 for (model,tau,book,report),g in d.groupby(['model_key','tau','book','report']):
  g=g.sort_values('origin_date');fullhit=(g.actual_loss.to_numpy()>g['var'].to_numpy()).astype(float)
  valid_lag=g.target_date.shift(2)<=g.origin_date
  if not valid_lag.iloc[2:].all():raise ValueError('Lagged hit not completed at current cutoff')
  # Future-start outcome becomes known two common observations after its origin.
  lag=np.concatenate([np.full(2,np.nan),fullhit[:-2]])
  g=g.copy();g['known_lag_hit']=lag
  g=g[g.origin_date.dt.year.between(2018,2025)]
  n=len(g);alpha=1-tau;y=g.actual_loss.to_numpy();q=g['var'].to_numpy();e=g.es.to_numpy()
  h=(y>q).astype(float)-alpha;u=(e-q-np.maximum(y-q,0)/alpha)/e
  for lags in [20,60]:
   rate=(h+alpha).mean();se=float(np.sqrt(hac(h,lags)[0,0]))
   stat,pv,rank=wald(np.column_stack([h,u]),lags)
   # Three predictable instruments for VaR; lag hit uses a completed target.
   instruments=np.column_stack([np.ones(n),g.known_lag_hit.to_numpy()-alpha,np.log(e)])
   cond=h[:,None]*instruments
   cond=cond[np.isfinite(cond).all(axis=1)]
   cv,cp,cr=wald(cond,lags)
   joint_cond=np.column_stack([h[:,None]*instruments,u[:,None]*instruments])
   joint_cond=joint_cond[np.isfinite(joint_cond).all(axis=1)]
   jc,jp,jr=wald(joint_cond,lags)
   sparse=bool((y>q).sum()<30)
   if sparse:pv=np.nan;cp=np.nan;jp=np.nan
   use=float(u.mean());use_se=float(np.sqrt(hac(u,lags)[0,0]))
   calibration.append(dict(model_key=model,tau=tau,book=book,report=report,n=n,breaches=int((y>q).sum()),
    nominal_expected=alpha*n,lags=lags,breach_rate=float(rate),rate_low=float(rate-1.96*se),rate_high=float(rate+1.96*se),
    es_identification_mean=use,es_identification_low=use-1.96*use_se,es_identification_high=use+1.96*use_se,
    joint_statistic=stat,joint_pvalue=pv,joint_rank=rank,conditional_var_statistic=cv,
    conditional_var_pvalue=cp,conditional_var_rank=cr,conditional_var_n=len(cond),
    conditional_joint_statistic=jc,conditional_joint_pvalue=jp,conditional_joint_rank=jr,
    fewer_than_30_breaches=sparse,pvalues_withheld_sparse_tail=sparse))
 pd.DataFrame(calibration).to_csv(OUT/'calibration.csv',index=False)
 # Stress support is reported, not trimmed from performance measures.
 if 'maximum_scenario_loss' in ev:
  stress=ev[ev.maximum_scenario_loss.notna()].copy()
  stress['max_support_over_es']=stress.maximum_scenario_loss/stress.es
  stress.sort_values('maximum_scenario_loss',ascending=False).head(100).to_csv(OUT/'largest_scenario_support.csv',index=False)
  stress.groupby(['tau','model_key']).agg(maximum_support=('maximum_scenario_loss','max'),
   p99_support=('maximum_scenario_loss',lambda x:x.quantile(.99)),maximum_es=('es','max'),
   median_support_over_es=('max_support_over_es','median')).reset_index().to_csv(OUT/'scenario_support_summary.csv',index=False)
 # Currency reporting effects are economic objects, shown for the same holdings.
 actual=ev[ev.model_key=='HS1250'][['origin_date','book','report','actual_loss']].drop_duplicates()
 actual=actual.pivot(index=['origin_date','book'],columns='report',values='actual_loss').reset_index()
 actual['USD_minus_EUR']=actual.USD-actual.EUR
 actual.groupby('book').agg(n=('EUR','size'),mean_EUR=('EUR','mean'),mean_USD=('USD','mean'),
  sd_EUR=('EUR','std'),sd_USD=('USD','std'),mean_abs_reporting_difference=('USD_minus_EUR',lambda x:np.abs(x).mean()),
  maximum_abs_reporting_difference=('USD_minus_EUR',lambda x:np.abs(x).max())).reset_index().to_csv(OUT/'reporting_effects.csv',index=False)
 manifest={'status':'completed','n_dates':ev.origin_date.nunique(),'n_models':ev.model_key.nunique(),
  'evaluation_first':str(ev.origin_date.min().date()),'evaluation_last':str(ev.origin_date.max().date()),
  'primary_contrasts':PRIMARY,'bootstrap_draws':B,'blocks':[20,60],'seed':SEED,
  'historical_reanalysis':True,'bootstrap_refits_algorithms':False,
  'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
  'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print(summaries[summaries.model_key.isin(['IN_TCOP_REF_POOL','IN_FHS_REF_POOL']+PRIMARY+['GARCH_CASH_T_2STEP','TW_RELATIVE','TW_MIN_RIDGE','IN_TCOP_REF_PAST_BEST','IN_FHS_REF_PAST_BEST'])].sort_values(['tau','rank']).to_string(index=False),flush=True)
 print(pd.DataFrame(contrast).to_string(index=False),flush=True)

if __name__=='__main__':main()
