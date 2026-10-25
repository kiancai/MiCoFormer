"""Direct H100 -> CmlLab delivery, bounded subprocesses, no scheduler actions."""
from pathlib import Path
import json
import shlex
import subprocess

from .prepare import RUN,ARM,now
from micoformer.mpc_unifrac import workflow as common

LOCAL='/home/cml_lab/caiqy/project/MCFProjet/tmp/20261025_gut_loraplus_h100_long'
SSH=['ssh','-F','/home/wyn/.ssh/config','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
     '-o','ForwardAgent=no','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15']


def sync(source,destination,*,exclude=None,timeout=180):
    command=['rsync','-a','--partial','--timeout=90','-e',shlex.join(SSH)]
    for pattern in exclude or []:command+=['--exclude',pattern]
    subprocess.run(command+[str(source),'cml_lab:'+destination],check=True,timeout=timeout)


def snapshot():
    for folder in ['audit','results','logs']:
        sync(str(RUN/'runs'/ARM/folder)+'/',LOCAL+'/runs/'+ARM+'/'+folder+'/',exclude=['*.pt','*.pt.*','*.lock'])
    sync(str(RUN/'audit')+'/',LOCAL+'/audit/h100/',exclude=['*.pt','*.pt.*','*.lock'])


def deliver_checkpoints():
    directory=RUN/'runs'/ARM/'run'
    records=[]
    # Stable immutable milestones first. Last/best are sent only after train exits.
    for name in ['epoch00.pt','epoch10.pt','epoch20.pt','epoch30.pt','epoch50.pt','last.pt','best.pt']:
        p=directory/name
        if p.exists():
            records.append(dict(file=name,bytes=p.stat().st_size,sha256=common.sha256_file(p)))
            sync(p,LOCAL+'/runs/'+ARM+'/run/',timeout=1800)
    if records:
        p=RUN/'audit/checkpoint_delivery.json'
        common._atomic_json(p,dict(status='TRANSFERRED',time=now(),source_machine='H100/NVH1003',files=records))
        sync(p,LOCAL+'/audit/')
