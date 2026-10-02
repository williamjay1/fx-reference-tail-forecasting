"""Actual two-origin forecast reconstruction; optionally regenerate main statistics."""
import fx_windows_platform_compat
from pathlib import Path
import argparse, gzip, json, subprocess, sys
import numpy as np
import pandas as pd
from project_config import ROOT, REPO_ROOT
def frozen(relative):
    row=next(x for x in json.loads((REPO_ROOT/'replication_manifest.json').read_text())['frozen_inputs'] if x['work_path']==relative)
    return pd.read_csv(REPO_ROOT/row['stored_path'],float_precision='round_trip')
def compare(a,b,name):
    assert list(a.columns)==list(b.columns),(name,'columns')
    assert a.shape==b.shape,(name,'shape',a.shape,b.shape)
    errors={}
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x=a[c].to_numpy(dtype=float);y=b[c].to_numpy(dtype=float)
            assert np.allclose(x,y,atol=1e-10,rtol=1e-10,equal_nan=True),(name,c)
            finite=np.isfinite(x)&np.isfinite(y);errors[c]=float(np.max(np.abs(x[finite]-y[finite]))) if finite.any() else 0.
        else:assert a[c].fillna('<NA>').astype(str).equals(b[c].fillna('<NA>').astype(str)),(name,c)
    return {'rows':len(a),'numeric_max_absolute_errors':errors}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--analysis',action='store_true');args=ap.parse_args()
    import joint17_internal_scenarios as ins
    subprocess.run([sys.executable,'-B',str(REPO_ROOT/'scripts/joint17_build_data.py')],check=True)
    panel_check=compare(pd.read_csv(ROOT/'datasets/joint_fx_20261002/ecb_joint_panel.csv',float_precision='round_trip'),frozen('datasets/joint_fx_20261002/ecb_joint_panel.csv'),'rebuilt panel')
    task_check=compare(pd.read_csv(ROOT/'results/joint_fx_20261002/data/outcome_tasks.csv',float_precision='round_trip'),frozen('results/joint_fx_20261002/data/outcome_tasks.csv'),'rebuilt outcomes')
    panel,r,ewma,radius,targets=ins.prepare();records=[]
    dates=['2018-01-02','2020-03-17']
    for date in dates:
        cutoff,annual=ins.annual_data(panel,r,int(date[:4]),16)
        t=int(np.flatnonzero(panel.date.to_numpy()==np.datetime64(date))[0])
        part,_=ins.forecasts_at(panel,r,ewma,radius,targets,t,16,annual,cutoff);records+=part
    a=pd.DataFrame(records);b=frozen('results/joint_fx_20261002/internal_full/predictions.csv');b=b[b.origin_date.isin(dates)]
    keys=['origin_date','book','report','tau','model','train_reference'];cols=keys+['actual_loss','var','es','score','strict_hit','scenario_n']
    a=a[cols].sort_values(keys).reset_index(drop=True);b=b[cols].sort_values(keys).reset_index(drop=True)
    forecast_check=compare(a,b,'two-origin forecasts')
    result={'status':'passed','origins':dates,'scenario_power':16,'panel_reconstruction':panel_check,'outcome_reconstruction':task_check,'forecast_reconstruction':forecast_check,'full_history_refitted':False,'all_SI_refitted':False,'analysis_recomputed':False}
    if args.analysis:
        run=subprocess.run([sys.executable,'-B',str(REPO_ROOT/'scripts/joint17_analyse.py')],text=True,capture_output=True)
        (ROOT/'replication_analysis.log').write_text(run.stdout+'\n'+run.stderr,encoding='utf-8');assert run.returncode==0,run.stderr[-2000:]
        result['analysis_tables']={}
        for name in ['model_summary.csv','primary_score_contrasts.csv','calibration.csv']:
            relative='results/joint_fx_20261002/analysis/'+name
            result['analysis_tables'][name]=compare(pd.read_csv(ROOT/relative,float_precision='round_trip'),frozen(relative),name)
        result['analysis_recomputed']=True
    path=ROOT/'replication_validation.json';path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
