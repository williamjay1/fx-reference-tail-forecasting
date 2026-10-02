"""Public, portable working directory. Immutable package inputs stay read only."""
from pathlib import Path
import os
import hashlib
REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('FX_WORK_ROOT', str(REPO_ROOT/'.work'))).expanduser().resolve()
ROOT.mkdir(parents=True, exist_ok=True)

def recorded_script_path(name, expected_digest):
    """Resolve the exact code version named by a frozen or newly generated manifest."""
    for directory in [REPO_ROOT/'scripts', REPO_ROOT/'provenance/original_scripts']:
        path=directory/name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==expected_digest:
            return path
    raise ValueError(f'No matching archived or portable source for {name}: {expected_digest}')

def legacy_work_input(value):
    """Map historical provenance paths to the configured, isolated working copy."""
    value=str(value).replace('\\','/')
    prefix='D:/MLWork/FXTailRisk/'
    path=ROOT/value[len(prefix):] if value.startswith(prefix) else Path(value)
    path=path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'Input outside configured working directory: {value}')
    return path
