"""Descriptive VaR/ES coordinate dispersion from the audited stored forecasts."""
from pathlib import Path
import json,hashlib
import fx_windows_platform_compat
import numpy as np
import pandas as pd
ROOT=Path('D:/MLWork/FXTailRisk'); OUT=ROOT/'results/jof_v19'; OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'results/joint_fx_20261002/internal_full/predictions.csv'
d=pd.read_csv(SOURCE)
references=['EUR','USD','GBP','JPY','CHF']
d=d.loc[(d.origin_date>='2018-01-01')&(d.model=='IN_FHS1250')&d.train_reference.isin(references)].copy()
keys=['origin_date','book','report','tau']
assert not d.duplicated(keys+['train_reference']).any()
assert d.groupby(keys).size().eq(5).all()
rows=[]
for key,g in d.groupby(keys,sort=True):
    r=dict(zip(keys,key))
    for col,label in [('var','VaR'),('es','ES')]:
        v=g[col].to_numpy(); denominator=np.mean(np.abs(v))
        assert denominator>0
        absolute=np.max(v)-np.min(v)
        r[f'D_{label}']=absolute; r[f'R_{label}']=absolute/denominator
        assert np.isclose(absolute*37/(np.mean(np.abs(v*37))),r[f'R_{label}'],atol=1e-14)
    rows.append(r)
daily=pd.DataFrame(rows)
assert len(daily)==2046*4*2
old=pd.read_csv(ROOT/'results/joint_fx_v18/diagnostic_bounds/daily_diagnostics.csv')
print('previous_fields',old.columns.tolist())
old=old.rename(columns={'anchor_symmetric_R_ES':'R_ES'})
if 'R_ES' not in old.columns:
    candidate=[c for c in old if 'symmetric' in c or c in ('relative_es_range','symmetric_relative_es_range')]
    assert len(candidate)==1,candidate
    old=old.rename(columns={candidate[0]:'R_ES'})
merged=daily.merge(old[keys+['R_ES']],on=keys,suffixes=('_new','_audited'),validate='one_to_one')
error=float(np.max(np.abs(merged.R_ES_new-merged.R_ES_audited)))
assert error<2e-13,error
summary=daily.groupby(['book','report','tau'])[['R_VaR','R_ES','D_VaR','D_ES']].median().reset_index()
daily.to_csv(OUT/'forecast_dispersion_daily.csv',index=False,float_format='%.17g')
summary.to_csv(OUT/'forecast_dispersion_summary.csv',index=False,float_format='%.17g')
labels={'funded_assets':'Assets','net_cashflows':'Flows'}
table=['| Fixed cash task | VaR span 95% | ES span 95% | VaR span 99% | ES span 99% |', '| --- | --- | --- | --- | --- |']
for (book,report),g in summary.groupby(['book','report']):
    vals=[]
    for tau in [.95,.99]:
        z=g.loc[np.isclose(g.tau,tau)].iloc[0]
        vals.extend([f'{100*z.R_VaR:.2f}%',f'{100*z.R_ES:.2f}%'])
    table.append('| '+ ' | '.join([f'{labels[book]}, {report}']+vals)+' |')
(OUT/'FORECAST_DISPERSION_TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
manifest=dict(status='completed',origins=2046,native_forecasts=len(d),diagnostic_rows=len(daily),
    input_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    existing_ES_diagnostic_max_difference=error,
    minimum_native_VaR=float(d['var'].min()),minimum_native_ES=float(d.es.min()),
    interpretation='descriptive re-analysis; symmetric mean-absolute-functional denominator; not independent validation',
    no_model_fitting=True,no_new_data=True)
(OUT/'forecast_dispersion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
print(summary.to_string(index=False));print(json.dumps(manifest))

if __name__=='__main__': pass
