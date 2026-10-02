from project_config import ROOT as PUBLIC_WORK_ROOT
"""Common-date MCS sensitivity for newly completed temporal/horizon forecasts."""
from pathlib import Path
import hashlib,json,time
import fx_windows_platform_compat
import numpy as np
import pandas as pd
from joint18_inference import circular_means,mcs_r,OUT,REPS,SEED

ROOT=(PUBLIC_WORK_ROOT/'results/joint_fx_v18')

def run():
    started=time.perf_counter();records=[];summaries=[];sources=[]
    for study in ['temporal','horizons']:
        source=ROOT/study/'predictions.csv';sources.append(source)
        d=pd.read_csv(source,parse_dates=['origin_date'],float_precision='round_trip')
        d['model_key']=d.model
        mask=d.model.eq('IN_FHS1250')
        d.loc[mask,'model_key']=d.loc[mask,'model']+'@'+d.loc[mask,'train_reference']
        for horizon,dh in d.groupby('horizon'):
            blocks=[20,60] if horizon<=5 else [20,60,120]
            task_count=dh.book.nunique()*dh.report.nunique()
            group=dh.groupby(['origin_date','tau','model_key'])
            assert group.size().eq(task_count).all()
            daily=group.score.mean().unstack('model_key')
            assert not daily.isna().any().any()
            for tau in [.95,.99]:
                x=daily.xs(tau,level='tau')
                for block in blocks:
                    boot=circular_means(x.to_numpy(),block)
                    r=mcs_r(x.to_numpy(),boot,list(x.columns))
                    r['study']=study;r['horizon']=horizon;r['tau']=tau;r['block']=block
                    r['n_dates']=len(x);r['tasks_per_date']=task_count;r['family_n']=len(x.columns)
                    records.append(r)
                    summaries.append(dict(study=study,horizon=int(horizon),tau=tau,block=block,
                      n_dates=len(x),tasks_per_date=task_count,family_n=len(x.columns),
                      included_n=int(r.included_95.sum()),
                      included=sorted(r.loc[r.included_95,'model_key']),
                      mixture_mcs_pvalue=float(r.set_index('model_key').loc['IN_FHS_REF_POOL','mcs_pvalue'])))
    pd.concat(records).to_csv(OUT/'extended_mcs_pvalues.csv',index=False,float_format='%.17g')
    (OUT/'extended_mcs_sets.json').write_text(json.dumps(summaries,indent=2)+'\n')
    manifest=dict(status='completed',method='HLN2011 T_R coherent elimination',reps=REPS,seed=SEED,
       retrospective_extensions_not_independent_confirmatory=True,
       bootstrap_conditions_on_recorded_forecasts=True,
       horizon20_blocks_include120=True,
       temporal_short_sample_and_sparse_q99_not_validation_of_calibration=True,
       source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_seconds=time.perf_counter()-started)
    (OUT/'extended_mcs_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':run()
