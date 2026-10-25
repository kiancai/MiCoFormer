"""Compute-node verification of static inputs and unchanged real head adapters."""
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
import sklearn
import joblib
from shared import *

assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_NODELIST')
m = load_manifest()
verify_record(m['environment']); verify_record(m['baseline_record'])
for task in m['tasks'].values():
    verify_record(task['manifest'])
    t = json.loads((EXP/'tasks/resp/audit/manifest.json').read_text())
    for p, digest in {**t['input_hashes'], **t['source_hashes']}.items(): assert sha(p) == digest, p
    assert [f['key'] for f in t['folds']]==[f['key'] for f in task['folds']]
baseline = json.loads((EXP/'audit/baseline_embedding.json').read_text())
assert sha(EXP/'tasks/resp/cache/weak_unifrac_e50.npy') == baseline['sha256']
x = np.load(EXP/'tasks/resp/cache/weak_unifrac_e50.npy', mmap_mode='r')
assert x.shape == (90764, 512) and x.dtype == np.float32 and np.isfinite(x).all()
assert len(m['tasks']['resp']['folds']) == 52 and m['fits_per_model'] == 104
versions = dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__)
expected = json.loads((EXP/'audit/expected_environment.json').read_text())
for key in ['torch', 'numpy', 'sklearn', 'joblib']:
    assert versions[key] == expected[key], (key, versions[key], expected[key])
cache = EXP/'tasks/resp/cache'; alias = cache/'preflight_baseline.npy'
if not alias.exists(): alias.symlink_to('weak_unifrac_e50.npy')
dump(EXP/'tasks/resp/audit/embeddings/preflight_baseline.json', dict(status='PASS', sha256=baseline['sha256'],
    manifest_sha256=sha(EXP/'tasks/resp/audit/manifest.json'), precision='accepted BF16 baseline; smoke only'))
seen = set()
for fold in m['tasks']['resp']['folds']:
    if fold['family'] in seen: continue
    seen.add(fold['family'])
    for head in ['linear', 'mlp']:
        subprocess.run([sys.executable, str(EXP/'tasks/resp/scripts/fit.py'), '--model', 'preflight_baseline',
                        '--fold', fold['key'], '--head', head, '--smoke'], check=True)
dump(EXP/'audit/static_acceptance.json', dict(status='PASS', time=now(), job=os.environ['SLURM_JOB_ID'],
    versions=versions, manifest_sha256=sha(MANIFEST), real_head_smokes=2*len(seen), fits_per_model=104))
