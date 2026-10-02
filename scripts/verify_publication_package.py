"""Standard-library checks for publication metadata and scientific source integrity.

This does not claim model refitting or independent-platform numerical validation.
"""
from pathlib import Path
import ast, hashlib, json

REPO = Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((REPO/'replication_manifest.json').read_text(encoding='utf-8'))
    metadata=json.loads((REPO/'.zenodo.json').read_text(encoding='utf-8'))
    assert metadata['version']==manifest['version']=='1.0.0'
    assert metadata['creators'][0]['name']=='Zhang, Junjie'
    assert metadata['creators'][0]['orcid']=='0009-0004-8821-4018'
    assert 'doi' not in metadata
    filenames={p.stem for p in (REPO/'scripts').glob('*.py')}
    for record in manifest['scientific_script_provenance']:
        p=REPO/record['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==record['published_sha256'],p
        original=REPO/'provenance/original_scripts'/p.name
        assert hashlib.sha256(original.read_bytes()).hexdigest()==record['source_sha256'],original
    for p in (REPO/'scripts').glob('*.py'):
        source=p.read_text(encoding='utf-8');compile(source,str(p),'exec')
        tree=ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): names=[x.name.split('.')[0] for x in node.names]
            elif isinstance(node,ast.ImportFrom):names=[(node.module or '').split('.')[0]]
            else:continue
            for name in names:
                if name.startswith(('joint17','joint18','jof19','jof20','fx_windows','project_config')):
                    assert name in filenames,(p,name,'missing local helper')
    assert not any(p.suffix.lower() in ['.docx','.tex'] for p in REPO.rglob('*') if '.git' not in p.parts)
    maximum=max((p.stat().st_size for p in REPO.rglob('*') if p.is_file() and '.git' not in p.parts),default=0)
    assert maximum<100_000_000
    print(json.dumps({'status':'passed','version':'1.0.0','scientific_scripts':len(manifest['scientific_script_provenance']),'frozen_inputs':len(manifest['frozen_inputs']),'largest_file_bytes':maximum,'numerical_rerun_claimed_by_this_check':False}))

if __name__=='__main__':main()
