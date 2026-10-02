from project_config import ROOT as PUBLIC_WORK_ROOT
from pathlib import Path
import json,hashlib
import fx_windows_platform_compat
import pandas as pd
REV=(PUBLIC_WORK_ROOT/'results/joint_fx_20261002')
def main():
 out=REV/'internal_full';out.mkdir(exist_ok=True)
 if (out/'predictions.csv').exists():raise FileExistsError('Already merged')
 sources=[REV/f'internal_full_{a}' for a in 'abcd'];frames=[];rad=[];man=[]
 for source in sources:
  m=json.loads((source/'manifest.json').read_text());assert m['status']=='completed' and m['scenario_power']==16
  frames.append(pd.read_csv(source/'predictions.csv'));rad.append(pd.read_csv(source/'radial_diagnostics.csv'));man.append(m)
 d=pd.concat(frames,ignore_index=True).sort_values(['origin_date','book','report','tau','model','train_reference'])
 assert not d.duplicated(['origin_date','book','report','tau','model','train_reference']).any()
 assert d.origin_date.nunique()==3324 and len(d)==398880
 d.to_csv(out/'predictions.csv',index=False,float_format='%.17g');pd.concat(rad,ignore_index=True).to_csv(out/'radial_diagnostics.csv',index=False)
 manifest=dict(status='completed',origins=d.origin_date.nunique(),predictions=len(d),scenario_power=16,
  component_manifests=man,maximum_member_identity_error=float(d.cash_identity_relative_error.max()),
  pool_identity_entries='Inherited member identities; zero entries are not independently recomputed measurements',
  prediction_sha256=hashlib.sha256((out/'predictions.csv').read_bytes()).hexdigest())
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
