"""Paths and JSON helpers. Training outputs always live outside source files."""
from pathlib import Path
import hashlib
import json
import os
import sys

PROJECT = Path(__file__).resolve().parents[1]
CODE = PROJECT / 'runtime'
VENDOR = PROJECT / 'vendor/llamafactory/src'
sys.path.insert(0, str(VENDOR))
ODB_VENDOR = PROJECT / 'vendor/odb/src'
sys.path.insert(0, str(ODB_VENDOR))

def run_root():
    return Path(os.environ['PPO4_RUN_ROOT']).expanduser().resolve()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def source_manifest():
    paths = [PROJECT / 'run.py', *CODE.glob('*.py'), *VENDOR.rglob('*.py'), *ODB_VENDOR.rglob('*.py'),
             * (PROJECT / 'configs').glob('*.yaml'), * (PROJECT / 'configs').glob('*.json')]
    return {p.relative_to(PROJECT).as_posix(): sha(p) for p in sorted(paths)}

def runtime_options():
    return read(run_root() / 'PROFILE.json')['runtime']
