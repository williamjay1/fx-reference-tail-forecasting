"""Verify and materialize frozen inputs, with lossless gzip decompression."""
from pathlib import Path
import argparse, gzip, hashlib, json, os, shutil
REPO=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work-dir',default=os.environ.get('FX_WORK_ROOT',str(REPO/'.work')))
    ap.add_argument('--data-only',action='store_true',help='Fresh numerical training: install data/style/table inputs, not saved model results')
    ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    work=Path(args.work_dir).expanduser().resolve();count=0;copied=0
    assert work!=REPO and not work.is_relative_to(REPO/'replication_inputs')
    manifest=json.loads((REPO/'replication_manifest.json').read_text(encoding='utf-8'))
    for row in manifest['frozen_inputs']:
        src=REPO/row['stored_path'];assert digest(src)==row['stored_sha256'],src
        count+=1
        if args.verify_only:continue
        rel=row['work_path']
        if args.data_only and not (rel.startswith(('datasets/','data/raw/','table_inputs/')) or '/style_sources/' in rel or rel=='results/joint_fx_20261002/data/outcome_tasks.csv'):continue
        dst=work/rel;assert dst.resolve().is_relative_to(work)
        if dst.exists():
            assert digest(dst)==row['sha256'],f'Refusing to overwrite different output: {dst}'
            continue
        dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.partial')
        opener=gzip.open if row['compression']=='gzip' else open
        with opener(src,'rb') as f,tmp.open('wb') as out:shutil.copyfileobj(f,out)
        assert digest(tmp)==row['sha256'] and tmp.stat().st_size==row['bytes'],src
        tmp.rename(dst);copied+=1
    print(json.dumps({'stored_inputs_verified':count,'work_files_materialized':copied,'work_dir':str(work)}))
if __name__=='__main__':main()
