"""Read immutable official ECB data, construct future-start cash-loss tasks."""
from pathlib import Path
import hashlib, json
import fx_windows_platform_compat
import numpy as np
import pandas as pd

ROOT=Path('D:/MLWork/FXTailRisk')
RAW=Path('F:/AcademicData/FXTailRisk/raw/joint_revision_20261002/ecb_EXR_USD_GBP_JPY_CHF_retrieved_20261002_v1.csv')
OUT=ROOT/'datasets/joint_fx_20261002'
RESULT=ROOT/'results/joint_fx_20261002/data'
CCY=['EUR','USD','GBP','JPY','CHF']
BOOKS={'funded_assets':np.array([.25,.25,.25,.25]),'net_cashflows':np.array([.4,.1,-.3,-.2])}

def main():
 OUT.mkdir(parents=True,exist_ok=True);RESULT.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(RAW,parse_dates=['TIME_PERIOD'])
 assert set(d.CURRENCY)==set(CCY[1:]) and d.CURRENCY_DENOM.eq('EUR').all()
 assert not d.duplicated(['CURRENCY','TIME_PERIOD']).any()
 good=d[d.OBS_STATUS.eq('A') & d.OBS_VALUE.gt(0)].copy()
 # Evaluation is deliberately bounded to the already-exposed original period.
 wide=good.pivot(index='TIME_PERIOD',columns='CURRENCY',values='OBS_VALUE').sort_index()
 wide=wide.loc[:'2025-12-31',CCY[1:]].dropna()
 p=1.0/wide;p.insert(0,'EUR',1.0);p.index.name='date'
 p.reset_index().to_csv(OUT/'ecb_joint_panel.csv',index=False,float_format='%.17g')
 r=np.diff(np.log(p.to_numpy(float)),axis=0)
 dates=p.index;records=[];max_err=0.0
 for t in range(1,len(p)-2):
  r1,r2=r[t],r[t+1]
  for book,w in BOOKS.items():
   a=N*w/p.iloc[t,1:].to_numpy(float) if (N:=100_000_000.) else None
   quantities=np.r_[-N*w.sum(),a]
   va=float(quantities@p.iloc[t+1].to_numpy(float));vt=float(quantities@p.iloc[t+2].to_numpy(float))
   le=float(np.sum(w*(np.exp(r1[1:])-np.exp(r1[1:]+r2[1:]))))
   ba=float(np.sum(w*np.exp(r1[1:]))-w.sum())
   for report in ['EUR','USD']:
    n=CCY.index(report);pn=float(p.iloc[t,n]);na=N/pn
    observed=(va/float(p.iloc[t+1,n])-vt/float(p.iloc[t+2,n]))/na
    formula=ba*np.exp(-r1[n])-(ba-le)*np.exp(-r1[n]-r2[n])
    identity=observed*np.exp(r1[n]+r2[n])-le-ba*np.expm1(r2[n])
    max_err=max(max_err,abs(observed-formula),abs(identity))
    records.append(dict(origin_date=dates[t].date().isoformat(),
      cutoff_utc=dates[t].date().isoformat()+'T18:00:00Z',start_date=dates[t+1].date().isoformat(),
      target_date=dates[t+2].date().isoformat(),book=book,report=report,actual_loss=observed,
      reference_notional=na,quantity_EUR=quantities[0],**{'quantity_'+c:quantities[j] for j,c in enumerate(CCY[1:],1)},
      identity_residual=identity,lead_calendar_days=(dates[t+1]-dates[t]).days,
      holding_calendar_days=(dates[t+2]-dates[t+1]).days))
 assert max_err<1e-12,max_err
 tasks=pd.DataFrame(records);tasks.to_csv(RESULT/'outcome_tasks.csv',index=False,float_format='%.17g')
 manifest={'status':'built_and_cash_identity_checked','raw':str(RAW),'raw_sha256':hashlib.sha256(RAW.read_bytes()).hexdigest(),
   'source_dates':len(p),'first_date':str(dates[0].date()),'last_date':str(dates[-1].date()),
   'currencies':CCY,'outcome_rows':len(tasks),'max_cash_formula_or_identity_error':max_err,
   'cutoff':'planned 18:00 UTC after expected publication, not observed historical first-release timestamps',
   'target':'future start next common reference date, endpoint subsequent date, fixed quantities at cutoff',
   'snapshot_not_realtime_vintage':True,'exposed_period':True,
   'panel_sha256':hashlib.sha256((OUT/'ecb_joint_panel.csv').read_bytes()).hexdigest(),
   'tasks_sha256':hashlib.sha256((RESULT/'outcome_tasks.csv').read_bytes()).hexdigest()}
 (RESULT/'panel_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(manifest),flush=True)
if __name__=='__main__':main()
