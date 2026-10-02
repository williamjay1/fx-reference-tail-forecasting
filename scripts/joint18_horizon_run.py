"""Exact finite adjacent-block risk at five and twenty future intervals."""
import argparse,time,json
import fx_windows_platform_compat
import numpy as np
import pandas as pd
import joint17_models as m
import joint18_temporal_common as c

def pilot():
 panel,r=c.panel_build();out=c.REV/'horizons';out.mkdir(parents=True,exist_ok=True)
 cutoff,annual,source=c.annual_data(panel,r,2025,out)
 ids=np.flatnonzero(panel.date.isin(pd.to_datetime(['2025-01-02','2025-03-03','2025-06-02','2025-09-01','2025-12-01'])))
 start=time.perf_counter();rr=[];checks=[]
 for t in ids:
  for horizon in [1,5,20]:
   rows,diag=c.forecasts_exact(panel,r,t,horizon,c.BOOKS,annual,cutoff);rr.extend(rows)
   diag.pop('actual');checks.append(dict(date=str(panel.date.iloc[t].date()),horizon=horizon,**diag))
 old=pd.read_csv(c.OLD/'internal_full/predictions.csv',float_precision='round_trip')
 new=pd.DataFrame(rr)
 key=['origin_date','book','report','tau','model','train_reference']
 lhs=new[(new.horizon==1)&new.book.isin(m.BOOKS)]
 joined=lhs.merge(old,on=key,suffixes=('_new','_old'))
 assert len(joined)==len(lhs), (len(joined),len(lhs))
 errors={field:float(np.abs(joined[field+'_new']-joined[field+'_old']).max()) for field in ['var','es','score','actual_loss']}
 # Proper-score divisions amplify machine-level cash discrepancies; separately
 # enforce economic-unit and score-unit numerical tolerances, not model ranking.
 assert max(errors[k] for k in ['var','es','actual_loss'])<1e-12 and errors['score']<1e-9,errors
 per=time.perf_counter()-start
 result=dict(status='passed',pilot_dates=[str(panel.date.iloc[t].date()) for t in ids],
  horizons=[1,5,20],predictions=len(rr),runtime_seconds=per,v17_h1_reproduction_errors=errors,
  maximum_general_horizon_realized_error=max(x['realized_cash_formula_error'] for x in checks),
  maximum_general_horizon_scenario_identity_error=max(x['scenario_relative_identity_error'] for x in checks),
  estimated_full_exact_seconds=per*4092/15,not_score_selected=True,
  note='Runtime estimate includes loading old full predictions; compiled full run should be faster.')
 c.write_json(out/'pilot.json',result);print(json.dumps(result),flush=True)

def full(args):
 panel,r=c.panel_build();out=c.REV/'horizons';out.mkdir(parents=True,exist_ok=True)
 if (out/'predictions.csv').exists():raise FileExistsError('Keep completed results')
 start=time.perf_counter();records=[];checks=[];params={}
 for year in range(2018,2026):
  cutoff,annual,source=c.annual_data(panel,r,year,out);params[str(year)]=c.sha(source)
  for horizon in [5,20]:
   ids=np.flatnonzero((panel.date.dt.year.to_numpy()==year)&(np.arange(len(panel))<len(panel)-1-horizon))
   # Historical comparison endpoint stays within2025 even though later data available.
   ids=np.array([t for t in ids if panel.date.iloc[t+1+horizon]<=pd.Timestamp('2025-12-31')])
   for oi,t in enumerate(ids):
    rows,diag=c.forecasts_exact(panel,r,t,horizon,c.BOOKS,annual,cutoff)
    records.extend(rows);diag.pop('actual');checks.append(dict(origin_date=str(panel.date.iloc[t].date()),horizon=horizon,**diag))
    if oi%80==0:print(f'h={horizon} year={year} {oi+1}/{len(ids)} elapsed {time.perf_counter()-start:.1f}s',flush=True)
  pd.DataFrame([x for x in records if x['origin_date'][:4]==str(year)]).to_csv(out/f'checkpoint_{year}.csv.gz',index=False,float_format='%.17g')
 d=pd.DataFrame(records);d.to_csv(out/'predictions.csv',index=False,float_format='%.17g')
 pd.DataFrame(checks).to_csv(out/'identity_checks.csv',index=False,float_format='%.17g')
 counts={str(h):int(x.origin_date.nunique()) for h,x in d.groupby('horizon')}
 c.write_json(out/'manifest.json',dict(status='completed',horizon_origins=counts,predictions=len(d),models=7,
  full_daily_origins=True,books={k:v.tolist() for k,v in c.BOOKS.items()},
  treasury_definition='Fixed at cutoff: +0.5N USD, +0.3N GBP, -0.15N JPY, -0.05N CHF values and -0.6N EUR financing; cash-flow risk, no interest/carry/delta.',
  future_start='t+1',endpoint='t+1+h',sample_window=1250,
  scenario_per_reference={str(h):1250-h for h in [5,20]},mixture_scenarios={str(h):5*(1250-h) for h in [5,20]},
  preserve_adjacent_shock_blocks=True,sequential_variance_recursion=True,
  runtime_seconds=time.perf_counter()-start,parameter_source_sha256=params,
  panel_sha256=c.sha(c.PANEL),prediction_sha256=c.sha(out/'predictions.csv'),
  script_sha256=c.sha(__file__),helper_sha256=c.sha(c.__file__),
  maximum_realized_formula_error=max(x['realized_cash_formula_error'] for x in checks),
  maximum_scenario_identity_error=max(x['scenario_identity_error'] for x in checks),
  maximum_relative_scenario_identity_error=max(x['scenario_relative_identity_error'] for x in checks),
  exact_enumeration=True,no_monte_carlo=True,no_configuration_selection=True,
  historical_reanalysis=True,confirmatory=False,cutoff='planned18UTC; no first-release vintage or executable prices'))
 print(json.dumps(json.loads((out/'manifest.json').read_text())),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');args=p.parse_args()
 if args.pilot:pilot()
 else:full(args)
