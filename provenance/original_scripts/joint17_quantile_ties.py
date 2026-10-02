"""Independent serialized-outcome tie check for the standard hit diagnostics."""
from pathlib import Path
import csv,json
ROOT=Path('D:/MLWork/FXTailRisk/results/joint_fx_20261002')
manifest=json.loads((ROOT/'analysis/manifest.json').read_text())
records=[]
for file in manifest['sources']:
 n=ties=near=0;closest=float('inf')
 with open(file,encoding='utf-8',newline='') as handle:
  for row in csv.DictReader(handle):
   if not '2018'<=row['origin_date'][:4]<='2025':continue
   if 'scenario_v2' in file and not ((row['model'].startswith('FHS') and row['model'][3:].isdigit()) or row['model']=='HS500'):continue
   y,q=float(row['actual_loss']),float(row['var']);gap=abs(y-q)
   n+=1;ties+=int(y==q);near+=int(gap<=1e-12);closest=min(closest,gap)
 records.append(dict(file=file,n=n,exact_ties=ties,within_1e_12=near,minimum_gap=closest if n else None))
output={'status':'completed','n':sum(r['n'] for r in records),'exact_ties':sum(r['exact_ties'] for r in records),
 'within_1e_12':sum(r['within_1e_12'] for r in records),'records':records,
 'scope':'Realized ties only; does not prove the unknown conditional loss law has no atom at a future quantile.'}
assert output['n']==41*2046*8
(ROOT/'analysis/quantile_tie_check.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
print(json.dumps(output))
