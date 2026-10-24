"""Lightweight submit02 transfer/submission watcher; fits only on Slurm nodes."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path('/hpcdisk1/limk_group/caiqy/project/260905_MCF')
RID='20261025_001_respiratory_method_screen_cpu'
TASK=ROOT/'tasks'/RID
LOCAL='/home/cml_lab/caiqy/project/MCFProjet/tmp/20261024_respiratory_peft_mechanism_pilot/method_screen/evaluation'
STAGE=ROOT/'staging'/RID/'live'
OPTIONS=['ssh','-F','/home/caiqy/.ssh/config','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','ForwardAgent=no','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15']


def stamp():
    from datetime import datetime
    return datetime.now().astimezone().isoformat()


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)


def publish_status(value):
    path=TASK/'audit/transport_status.json'
    dump(path,value)
    result=subprocess.run(['rsync','-a','-e',' '.join(OPTIONS),str(path),
        'cml_lab:'+LOCAL+'/audit/big_transport_status.json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=60)
    if result.returncode:dump(TASK/'audit/status_push_error.json',dict(time=stamp(),error=result.stderr[-1500:]))


def main():
    STAGE.mkdir(parents=True,exist_ok=True)
    with (TASK/'audit/transport.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=json.loads((TASK/'audit/execution_manifest.json').read_text())
        submissions=TASK/'audit/submissions.json'
        if submissions.exists(): raise RuntimeError('existing submission ledger requires inspection; no duplicate launch')
        jobs={};returned=[];failures={};started=time.monotonic()
        excludes=['*.partial*','*.lock','/scripts','/delivery','/results','/logs','/big_return','/incoming',
                  '/static_bundle.tar','/static_bundle.json','/tasks/resp/results','/tasks/resp/logs',
                  '/tasks/resp/scripts','/tasks/resp/audit/source','/tasks/resp/cache/folds',
                  '/tasks/resp/cache/genus_ids.npy','/tasks/resp/cache/rclr.npy','/tasks/resp/cache/padding.npy',
                  '/tasks/resp/cache/weak_unifrac_e50.npy']
        transport=['rsync','-a','--copy-links','--partial','-e',' '.join(OPTIONS)]
        for item in excludes: transport += ['--exclude',item]
        transport += ['cml_lab:'+LOCAL+'/',str(STAGE)+'/']
        subprocess.run(transport[:2]+['--dry-run']+transport[2:],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        while time.monotonic()-started < 12*3600:
            static=TASK/'audit/static_acceptance.json'
            if not static.exists():
                submission=TASK/'audit/preflight_submission.json'
                if submission.exists():
                    job=json.loads(submission.read_text())['job']
                    state=subprocess.check_output(['sacct','-n','-X','-j',job,'-o','State,ExitCode','-P'],text=True).strip()
                    if state.startswith(('FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY')):
                        raise RuntimeError('static preflight failed: '+state)
                publish_status(dict(status='WAITING_STATIC_PREFLIGHT',time=stamp()))
                time.sleep(30);continue
            if json.loads(static.read_text())['status']!='PASS':raise RuntimeError('static preflight failed')
            transfer=subprocess.run(transport,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=600)
            if transfer.returncode:
                dump(TASK/'audit/transport_transfer_error.json',dict(time=stamp(),returncode=transfer.returncode,error=transfer.stderr[-3000:]))
                time.sleep(60);continue
            # squeue handles active-job accounting; never resubmit a vanished job.
            active=set(subprocess.check_output(['squeue','-h','-u','caiqy','-o','%i'],text=True).split())
            slots=6-sum(job in active for job in jobs.values())
            for arm in m['arms']:
                if arm in jobs or slots<=0:continue
                if not (STAGE/'audit/extracted'/f'{arm}.json').exists():continue
                if not (STAGE/'tasks/resp/cache'/f'{arm}.npy').exists():continue
                # Worker verifies all hashes before publishing and consuming the staged model.
                command=['sbatch','--parsable','--job-name=mcf_method_'+arm,'--output='+str(TASK/'logs'/f'{arm}_%j.log'),
                         '--export=ALL,MCF_ARM='+arm,str(TASK/'scripts/cpu_model.sbatch')]
                job=subprocess.check_output(command,text=True).strip().split(';')[0]
                if not job.isdigit():raise RuntimeError('unexpected sbatch receipt')
                jobs[arm]=job;slots-=1
                dump(submissions,dict(time=stamp(),jobs=jobs,commands_note='8CPU/32GiB/core56/6h; max6; no automatic retry'))
            for arm,job in jobs.items():
                ready=TASK/'delivery'/f'{arm}.ready.json'
                if arm in returned:continue
                if ready.exists():
                    command=['rsync','-a','--partial','-e',' '.join(OPTIONS),str(TASK/'delivery'/f'{arm}.tar'),str(ready),
                             'cml_lab:'+LOCAL+'/big_return/']
                    subprocess.run(command,check=True,timeout=600,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
                    returned.append(arm)
                    failures.pop(arm,None)
                elif job not in active:
                    state=subprocess.check_output(['sacct','-n','-X','-j',job,'-o','State,ExitCode','-P'],text=True).strip()
                    if state.startswith(('FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','DEADLINE')):
                        failures[arm]=dict(job=job,state=state)
            publish_status(dict(status='RUNNING',time=stamp(),jobs=jobs,returned=returned,failures=failures,
                pending_models=[a for a in m['arms'] if a not in jobs]))
            if len(returned)+len(failures)==len(m['arms']):
                publish_status(dict(status='COMPLETED' if not failures else 'PARTIAL_FAILURE',
                    time=stamp(),jobs=jobs,returned=returned,failures=failures));return
            time.sleep(45)
        dump(TASK/'audit/transport_status.json',dict(status='WAIT_LIMIT_REACHED',time=stamp(),jobs=jobs,returned=returned,failures=failures))


if __name__=='__main__':
    try:main()
    except BaseException as error:
        dump(TASK/'audit/transport_failure.json',dict(time=stamp(),error=repr(error),traceback=traceback.format_exc()))
        publish_status(dict(status='FAILED',time=stamp(),error=repr(error)))
        raise
