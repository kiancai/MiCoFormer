"""One finite, UUID-bound gut run; checkpoint delivery preserves all allocations."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import traceback

from .prepare import RUN,ARM,UUID,verify,now,configure_tree
from micoformer.mpc_unifrac import workflow as common


def endpoint(manifest,sha):
    import torch
    from micoformer.domain_long_adaptation.workflow import load_model
    from micoformer.domain_method_adaptation.modules import frozen_sha256
    torch.set_num_threads(2)
    arm=manifest['arms'][ARM];root=Path(arm['output']);p=root/'run/epoch50.pt'
    state=torch.load(p,map_location='cpu',weights_only=False)
    if (state['arm']!=ARM or state['epoch']!=50 or state['global_step']!=115200
            or state['manifest_sha256']!=sha or state['next_batch']!=0):
        raise RuntimeError('wrong fixed endpoint')
    model=load_model(arm,manifest,torch.device('cpu'))
    model.load_state_dict(state['model_state_dict'],strict=True)
    if frozen_sha256(model)!=arm['initialization']['frozen_parameter_sha256']:
        raise RuntimeError('endpoint frozen parameters changed')
    if not all(bool(torch.isfinite(v).all()) for v in model.state_dict().values()):
        raise RuntimeError('endpoint nonfinite')
    result=dict(status='PASS',time=now(),manifest_sha256=sha,source_commit=manifest['source']['commit'],
                epoch=50,global_step=115200,strict_load=True,frozen_weights_unchanged=True,
                parameters_finite=True,checkpoint_sha256=common.sha256_file(p),checkpoint=str(p))
    common._atomic_json(root/'audit/endpoint_acceptance.json',result)


def pipeline():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or os.environ.get('MCF_TARGET_GPU_UUID')!=UUID:
        raise RuntimeError('only user-approved GPU3 may run this experiment')
    with (RUN/'audit/pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,sha=verify();stop=threading.Event()
        def heartbeat():
            from .transfer import snapshot
            while not stop.is_set():
                try:snapshot()
                except Exception as error:common._atomic_json(RUN/'audit/transfer_error.json',dict(time=now(),error=repr(error)))
                stop.wait(60)
        worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
        try:
            for action in ['gate','train']:
                common._atomic_json(RUN/'audit/pipeline_status.json',dict(status='RUNNING',phase=action,
                    time=now(),pid=os.getpid(),uuid=UUID,manifest_sha256=sha))
                subprocess.run([sys.executable,'-m',__package__+'.runner',action],check=True)
            endpoint(manifest,sha)
            common._atomic_json(RUN/'audit/pipeline_status.json',dict(status='COMPLETED',time=now(),
                pid=os.getpid(),uuid=UUID,manifest_sha256=sha))
        except BaseException as error:
            common._atomic_json(RUN/'audit/pipeline_status.json',dict(status='FAILED',time=now(),
                error=repr(error),traceback=traceback.format_exc(),manifest_sha256=sha))
            raise
        finally:
            stop.set();worker.join(timeout=20)
            from .transfer import deliver_checkpoints,snapshot
            try:snapshot();deliver_checkpoints()
            except Exception as error:common._atomic_json(RUN/'audit/final_transfer_error.json',dict(time=now(),error=repr(error)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pipeline','verify','gate','train'])
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.action=='pipeline':return pipeline()
    manifest,sha=verify();configure_tree()
    if args.action=='verify':print(json.dumps(dict(status='PASS',manifest_sha256=sha)),flush=True);return
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or os.environ.get('MCF_TARGET_GPU_UUID')!=UUID:
        raise RuntimeError('only approved GPU3')
    from micoformer.domain_long_adaptation.workflow import preflight,train
    arm=manifest['arms'][ARM];device=preflight(Path(arm['output'])/'audit'/f'{args.action}_preflight.json')
    if args.action=='gate':
        from micoformer.domain_long_adaptation.gates import run
        return run(manifest,sha,device,ARM)
    gate=json.loads((Path(arm['output'])/'audit/cuda_gates.json').read_text())
    if gate['status']!='PASS' or gate['manifest_sha256']!=sha:
        raise RuntimeError('exact gut recipe CUDA gate required')
    if not train(arm,manifest,sha,device,resume=args.resume):
        raise RuntimeError('training interrupted; preserve checkpoint and allocations')


if __name__=='__main__':main()
