"""H100-to-CmlLab direct rsync, complete embedding before ready marker."""
import json
import fcntl
import time
import shlex
import subprocess
from pathlib import Path
from .prepare import RUN, EVAL


def options():
    return ['ssh','-F','/home/wyn/.ssh/config','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
            '-o','ForwardAgent=no','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15']


def send(paths,destination):
    # Exact immutable files, no deletes and no data through jump hosts.
    command=['rsync','-a','--partial','-e',shlex.join(options()),*[str(p) for p in paths],'cml_lab:'+destination]
    for attempt in range(3):
        result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
        if result.returncode==0:return
        if attempt==2:raise RuntimeError('direct rsync failed: '+result.stderr.decode(errors='replace')[-2000:])
        time.sleep(10*(attempt+1))



def publish_endpoint(key):
    config=json.loads((RUN/'settings.json').read_text());local=config['local_stage']+'/evaluation'
    send([EVAL/'tasks/resp/cache'/f'{key}.npy'],local+'/tasks/resp/cache/')
    send([EVAL/'tasks/resp/audit/embeddings'/f'{key}.json'],local+'/tasks/resp/audit/embeddings/')
    send([EVAL/'audit/extracted'/f'{key}.json'],local+'/audit/extracted/')


def snapshot():
    with (RUN/'audit/snapshot.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        _snapshot()


def _snapshot():
    config=json.loads((RUN/'settings.json').read_text());local=config['local_stage']
    for name in ['loraplus_r64','lora_r128','dora_r64','adapter_b256']:
        for folder in ['audit','logs','results']:
            src=RUN/'runs'/name/folder
            if src.exists():
                subprocess.run(['rsync','-a','--exclude','*.pt','--exclude','*.partial*','-e',shlex.join(options()),
                    str(src)+'/',f'cml_lab:{local}/runs/{name}/{folder}/'],check=True,
                    stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
    files=list((RUN/'audit').glob('queue*_status.json'))+list((RUN/'audit').glob('queue*_failure.json'))
    if files:send(files,local+'/audit/')
