#!/usr/bin/env python3
"""Arm a detached finite controller after all preparation receipts exist."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parents[1]
STAGE=ROOT/'tmp/20261024_respiratory_peft_mechanism_pilot/method_screen'
receipt=STAGE/'audit/controller_launch.json'
if receipt.exists():raise RuntimeError('already armed; inspect existing controller instead of launching twice')
git='/home/cml_lab/anaconda3/bin/git'
assert not subprocess.check_output([git,'-C',str(WORK),'status','--porcelain'],text=True)
for family in ['adapter','loraplus','dora']:
    assert (STAGE/'families'/family/'audit/manifest.sha256').exists()
assert (STAGE/'evaluation/audit/execution_manifest.json').exists()
accepted=json.loads((STAGE/'audit/cpu_acceptance.json').read_text())
assert accepted['status']=='PASS'
assert accepted['source_commit']==subprocess.check_output([git,'-C',str(WORK),'rev-parse','HEAD'],text=True).strip()
env=dict(os.environ,PYTHONPATH=str(WORK),PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
         OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
command=['/home/cml_lab/anaconda3/envs/MiCoFormerV2/bin/python','-m','micoformer.domain_method_adaptation.controller']
with (STAGE/'controller.log').open('xb') as output:
    process=subprocess.Popen(['nohup',*command],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
receipt.write_text(json.dumps(dict(pid=process.pid,sid=process.pid,command=command,
    source_commit=subprocess.check_output([git,'-C',str(WORK),'rev-parse','HEAD'],text=True).strip(),
    finite_configurations=6,preserve_placeholder=True),indent=2)+'\n')
time.sleep(2)
if process.poll() is not None:raise RuntimeError('controller exited on startup; inspect log')
print(receipt.read_text())
