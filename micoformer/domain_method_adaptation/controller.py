"""Finite, completion-driven per-card queue. No cancellation or allocation mutation."""
from __future__ import annotations
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
WORK=Path(__file__).resolve().parents[2]
EXPERIMENT=ROOT/'tmp/20261024_respiratory_peft_mechanism_pilot'
STAGE=EXPERIMENT/'method_screen'
CONFIG={
    'adapter':dict(gpu='GPU-3c1bc64f-868c-2fb1-72d4-862fba149f6e',cpus='32-35',predecessor=EXPERIMENT/'a100_bucket',step='673251.17'),
    'loraplus':dict(gpu='GPU-b58967f1-b9e6-2344-feea-41677956a5f4',cpus='36-39',predecessor=EXPERIMENT/'a100_parallel/pairs/r128_seed43',step='673251.21'),
    'dora':dict(gpu='GPU-631f171f-167c-2f1c-5be1-6f421afd8fb3',cpus='48-51',predecessor=EXPERIMENT/'a100_parallel/pairs/r64_seed42',step='673251.22'),
}
LOCK=threading.Lock()


def now():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)


def predecessor_ready(base):
    final=base/'results/sequence_status.json'
    if not final.exists():return False
    seal=(base/'audit/manifest.sha256').read_text().strip()
    if sha(base/'audit/manifest.json')!=seal:raise RuntimeError('predecessor manifest changed')
    sequence=json.loads(final.read_text())
    if sequence.get('status')!='COMPLETED' or sequence['manifest_sha256']!=seal:
        raise RuntimeError('predecessor sequence rejected')
    for name in ['random','window4096']:
        run=base/'runs'/name
        status=json.loads((run/'results/training_status.json').read_text())
        epochs=[json.loads(s) for s in (run/'results/epochs.jsonl').read_text().splitlines()]
        if (status.get('status')!='COMPLETED' or status.get('global_step')!=8670 or status.get('epochs')!=10
                or status['manifest_sha256']!=seal or not (run/'run/epoch10.pt').is_file()
                or [e['epoch'] for e in epochs]!=list(range(1,11))
                or not all(e['exact_train_row_coverage'] and e['frozen_weights_unchanged']
                           and e['validation']['exact_row_coverage'] for e in epochs)):
            raise RuntimeError('predecessor endpoint failed acceptance')
    return True


def allocation():
    return subprocess.check_output(['squeue','-h','-j','673251','-o','%T'],text=True).strip()


def gpu_free(uuid):
    output=subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','gpu1',
        '/usr/local/nvidia/bin/nvidia-smi -i '+uuid+' --query-compute-apps=pid --format=csv,noheader,nounits'],text=True,timeout=30)
    return not any(line.strip() for line in output.splitlines())


def run_family(family,configuration):
    output=STAGE/'families'/family
    state=output/'audit/controller_status.json'
    started=time.monotonic();missing=0
    try:
        dump(state,dict(status='WAITING_PREDECESSOR',time=now(),predecessor=str(configuration['predecessor']),gpu=configuration['gpu']))
        while time.monotonic()-started<12*3600:
            if allocation()!='RUNNING':raise RuntimeError('placeholder is not RUNNING; no replacement allocation')
            if predecessor_ready(configuration['predecessor']) and gpu_free(configuration['gpu']):break
            active=subprocess.check_output(['squeue','-h','--steps','-j','673251','-o','%i'],text=True).split()
            if configuration['step'] not in active and not predecessor_ready(configuration['predecessor']):
                missing+=1
                if missing>=3:raise RuntimeError('predecessor step exited without accepted pair')
            else:missing=0
            time.sleep(45)
        else:raise RuntimeError('predecessor wait limit reached')
        manifest=output/'audit/manifest.json';seal=(output/'audit/manifest.sha256').read_text().strip()
        if sha(manifest)!=seal:raise RuntimeError('successor manifest changed')
        command=['srun','--jobid=673251','--overlap','--exact','--nodes=1','--ntasks=1','--cpus-per-task=15',
                 '--cpu-bind=none','--gres=gpu:3','--immediate=30','/bin/bash',str(WORK/'scripts/run_domain_methods_placeholder.sh'),
                 str(WORK),family,configuration['gpu'],configuration['cpus'],'--manifest',str(manifest),'--manifest-sha256',seal]
        with (output/'pipeline.log').open('xb') as log:
            process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        dump(output/'audit/driver_launch.json',dict(time=now(),pid=process.pid,command=command,manifest_sha256=seal,
            gpu=configuration['gpu'],actual_cpu_affinity=configuration['cpus'],allocation=673251,
            step_requests_all_owned_cpus_then_tasksets=True,preserve_placeholder=True))
        dump(state,dict(status='SUCCESSOR_PIPELINE_RUNNING',time=now(),driver_pid=process.pid))
        code=process.wait()
        final=output/'results/sequence_status.json'
        if code or not final.exists() or json.loads(final.read_text())['status']!='COMPLETED':
            raise RuntimeError(f'successor pipeline failed or incomplete, exit={code}')
        dump(state,dict(status='COMPLETED',time=now(),preserve_placeholder=True))
        return True
    except BaseException as error:
        dump(state,dict(status='FAILED',time=now(),error=repr(error),traceback=traceback.format_exc(),preserve_placeholder=True))
        return False


def main():
    with (STAGE/'audit/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        launch=STAGE/'audit/controller_launch.json'
        for _ in range(50):
            if launch.exists():break
            time.sleep(.1)
        if not launch.exists():raise RuntimeError('controller must be launched through the recorded launcher')
        dump(STAGE/'audit/controller_status.json',dict(status='WAITING_PREDECESSORS',time=now(),pid=os.getpid(),sid=os.getsid(0),
            families=list(CONFIG),finite_configurations=6,placeholder_lifecycle_untouched=True))
        from .reporting import collect
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs={family:pool.submit(run_family,family,config) for family,config in CONFIG.items()}
            cpu=pool.submit(collect)
            results={family:future.result() for family,future in jobs.items()}
            results['cpu_delivery']=cpu.result()
        dump(STAGE/'audit/controller_status.json',dict(status='COMPLETED' if all(results.values()) else 'PARTIAL_FAILURE',
            time=now(),families=results,placeholder_lifecycle_untouched=True))


if __name__=='__main__': main()
