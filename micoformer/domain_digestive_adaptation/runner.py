"""One protected GPU, four finite runs, frozen endpoints and direct delivery."""
import argparse,fcntl,json,os,subprocess,sys,threading,time,traceback
from pathlib import Path
from .prepare import RUN,NAMES,UUID,verify,now
from micoformer.mpc_unifrac import workflow as common

LOCAL='/home/cml_lab/caiqy/project/MCFProjet/tmp/20261025_digestive_l3_adaptation'
SSH=['ssh','-F','/home/wyn/.ssh/config','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ForwardAgent=no','-o','ClearAllForwardings=yes','-o','ConnectTimeout=15']

def sync(source,dest,exclude=(),timeout=300):
 import shlex
 cmd=['rsync','-a','--partial','--timeout=90','-e',shlex.join(SSH)]
 for pattern in exclude:cmd+=['--exclude',pattern]
 subprocess.run(cmd+[str(source),'cml_lab:'+dest],check=True,timeout=timeout,capture_output=True)

def snapshot():
 for name in NAMES:
  for folder in ['audit','logs','results']:
   sync(str(RUN/'runs'/name/folder)+'/',LOCAL+'/runs/'+name+'/'+folder+'/',exclude=['*.pt','*.partial*','*.lock'])
 sync(str(RUN/'audit')+'/',LOCAL+'/audit/h100/',exclude=['*.pt','*.partial*','*.lock'])

def deliver(name):
 records=[]
 for p in sorted((RUN/'runs'/name/'run').glob('*.pt')):
  if p.name not in ['epoch00.pt','epoch10.pt','epoch20.pt','epoch30.pt','epoch50.pt','last.pt','best.pt']:continue
  records.append(dict(file=p.name,bytes=p.stat().st_size,sha256=common.sha256_file(p)))
  sync(p,LOCAL+'/runs/'+name+'/run/',timeout=1800)
 p=RUN/'audit'/f'{name}_checkpoint_delivery.json';common._atomic_json(p,dict(status='TRANSFERRED',time=now(),arm=name,files=records));sync(p,LOCAL+'/audit/')

def pipeline():
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID or os.environ.get('MCF_TARGET_GPU_UUID')!=UUID:raise RuntimeError('expected one approved GPU3')
 with (RUN/'audit/pipeline.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);manifest,sha=verify();stop=threading.Event();complete=[];started=time.monotonic()
  def heartbeat():
   while not stop.is_set():
    try:snapshot()
    except Exception as error:common._atomic_json(RUN/'audit/transfer_error.json',dict(time=now(),error=repr(error)))
    stop.wait(60)
  worker=threading.Thread(target=heartbeat,daemon=True);worker.start()
  try:
   for name in NAMES:
    if time.monotonic()-started>36*3600:raise RuntimeError('finite queue deadline exceeded')
    for action in ['gate','train']:
     common._atomic_json(RUN/'audit/pipeline_status.json',dict(status='RUNNING',time=now(),current=name,phase=action,completed=complete,pid=os.getpid(),manifest_sha256=sha))
     subprocess.run([sys.executable,'-m',__package__+'.runner',action,'--arm',name],check=True)
    deliver(name)
    for epoch in manifest['execution']['evaluation_epochs'][name]:
     subprocess.run([sys.executable,str(RUN/'evaluation/scripts/extract.py'),'--arm',name,'--epoch',str(epoch)],check=True)
    complete.append(name);snapshot()
   common._atomic_json(RUN/'audit/pipeline_status.json',dict(status='COMPLETED',time=now(),completed=complete,manifest_sha256=sha))
  except BaseException as error:
   common._atomic_json(RUN/'audit/pipeline_status.json',dict(status='FAILED',time=now(),completed=complete,error=repr(error),traceback=traceback.format_exc(),manifest_sha256=sha));raise
  finally:
   stop.set();worker.join(timeout=20)
   try:snapshot()
   except Exception as error:print('snapshot error',repr(error),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['pipeline','verify','gate','train']);parser.add_argument('--arm',choices=NAMES);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
 if args.action=='pipeline':return pipeline()
 m,sha=verify()
 if args.action=='verify':print(json.dumps(dict(status='PASS',manifest_sha256=sha)));return
 if not args.arm:parser.error('--arm required')
 from micoformer.domain_long_adaptation.workflow import preflight,train
 a=m['arms'][args.arm];device=preflight(Path(a['output'])/'audit'/f'{args.action}_preflight.json')
 if args.action=='gate':
  from micoformer.domain_long_adaptation.gates import run
  return run(m,sha,device,args.arm)
 g=json.loads((Path(a['output'])/'audit/cuda_gates.json').read_text())
 if g['status']!='PASS' or g['manifest_sha256']!=sha:raise RuntimeError('exact CUDA numerical/restore gate required')
 if not train(a,m,sha,device,resume=args.resume):raise RuntimeError('interrupted; preserve work and allocation')

if __name__=='__main__':main()
