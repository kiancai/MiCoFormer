"""Finite detached queues; numerical gates precede each formal 50-epoch arm."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from .prepare import RUN,NAMES,verify,now,configure_tree
from micoformer.mpc_unifrac import workflow as common

QUEUES={0:['loraplus_r64','lora_r128'],1:['dora_r64','adapter_b256']}
UUIDS={0:'GPU-3d8a1781-fc8e-1855-bff0-6bda65635a07',1:'GPU-a576c223-bd1d-ae30-dc26-ab90d9053d61'}


def pipeline(slot):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUIDS[slot] or os.environ.get('MCF_TARGET_GPU_UUID')!=UUIDS[slot]:raise RuntimeError('queue UUID mismatch')
    for name in NAMES:
        for folder in ['audit','logs','results','run']:(RUN/'runs'/name/folder).mkdir(parents=True,exist_ok=True)
    status=RUN/f'audit/queue{slot}_status.json'
    with (RUN/f'audit/queue{slot}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,sha=verify();stop=threading.Event()
        def heartbeat():
            from .transfer import snapshot
            while not stop.is_set():
                try:snapshot()
                except Exception as error:common._atomic_json(RUN/f'audit/queue{slot}_transfer_error.json',dict(time=now(),error=repr(error)))
                stop.wait(45)
        worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
        complete=[]
        try:
            for arm in QUEUES[slot]:
                common._atomic_json(status,dict(status='RUNNING',time=now(),pid=os.getpid(),sid=os.getsid(0),
                    queue=slot,uuid=UUIDS[slot],current=arm,completed=complete,manifest_sha256=sha))
                for phase in ['gate','train']:
                    subprocess.run([sys.executable,'-m',__package__+'.runner',phase,'--arm',arm],check=True)
                complete.append(arm)
            common._atomic_json(status,dict(status='COMPLETED',time=now(),queue=slot,completed=complete,manifest_sha256=sha))
        except BaseException as error:
            common._atomic_json(status,dict(status='FAILED',time=now(),queue=slot,current=arm,completed=complete,error=repr(error)))
            common._atomic_json(RUN/f'audit/queue{slot}_failure.json',dict(time=now(),error=repr(error),traceback=traceback.format_exc()))
            raise
        finally:
            stop.set();worker.join(timeout=10)
            from .transfer import snapshot
            try:snapshot()
            except Exception as error:print('snapshot error',repr(error),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pipeline','gate','train','extract','verify']);parser.add_argument('--arm',choices=NAMES);parser.add_argument('--epoch',type=int,choices=[10,20,30,50]);parser.add_argument('--slot',type=int,choices=[0,1]);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='pipeline':return pipeline(args.slot)
    manifest,sha=verify()
    configure_tree()
    if args.action=='verify':print(json.dumps(dict(status='PASS',manifest_sha256=sha)),flush=True);return
    if not args.arm:parser.error('--arm required')
    if args.action=='extract':
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('no single CUDA device; no CPU fallback')
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        from .extraction import extract
        return extract(args.arm+f'_e{args.epoch:02d}',torch.device('cuda:0'))
    from .workflow import preflight,train
    arm=manifest['arms'][args.arm]
    device=preflight(Path(arm['output'])/'audit'/f'{args.action}_preflight.json')
    if args.action=='gate':
        from .gates import run
        return run(manifest,sha,device,args.arm)
    gate=json.loads((Path(arm['output'])/'audit/cuda_gates.json').read_text())
    if gate['status']!='PASS' or gate['manifest_sha256']!=sha:raise RuntimeError('missing exact recipe gate')
    if not train(arm,manifest,sha,device,resume=args.resume):raise RuntimeError('training interrupted; preserve checkpoint and allocation')


if __name__=='__main__':main()
