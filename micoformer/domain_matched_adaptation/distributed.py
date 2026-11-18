"""Three-rank execution reusing the sealed legacy model, LoRA+, losses and optimizer."""
from datetime import timedelta
from pathlib import Path
import gc,json,math,os,signal,time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from micoformer.domain_long_adaptation import workflow as w
from micoformer.domain_adaptation import distributed as old_ddp
from micoformer.domain_adaptation.workflow import capture_rng,restore_rng,state_equal
from micoformer.domain_method_adaptation.modules import frozen_sha256
from micoformer.domain_pretraining.input import InMemoryFullSupportRowDataset
from micoformer.domain_bucket_adaptation.sampling import BucketSampler,TensorCollator
from micoformer.mpc_unifrac.data import DualCorruptionCollator,batch_to_device
from micoformer.mpc_unifrac.teacher import weighted_unifrac,neighborhood_details
from micoformer.mpc_unifrac import workflow as common

def initialise(root,uuids):
 rank=int(os.environ['LOCAL_RANK']);cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cpus[4*rank:4*rank+4])
 assert os.environ.get('SLURM_JOB_ID')=='673251' and int(os.environ['WORLD_SIZE'])==3
 assert os.environ['CUDA_VISIBLE_DEVICES'].split(',')==uuids and torch.cuda.device_count()==3
 assert 'job_673251' in Path('/proc/self/cgroup').read_text()
 device=torch.device('cuda',rank);torch.cuda.set_device(device);torch.set_num_threads(1)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_bf16_supported() and 'A100' in torch.cuda.get_device_name(device)
 assert str(torch.cuda.get_device_properties(device).uuid).removeprefix('GPU-')==uuids[rank].removeprefix('GPU-')
 dist.init_process_group('nccl',timeout=timedelta(minutes=5),device_id=device)
 test=torch.tensor(rank+1.,device=device);dist.all_reduce(test);assert test.item()==6
 w.configure_tree()
 return rank,device

class Batches:
 def __init__(self,lengths,rank,train):self.lengths=lengths;self.rank=rank;self.train=train;self.set_epoch(0)
 def set_epoch(self,epoch,cursor=0):
  self.cursor=cursor
  if self.train:
   base=BucketSampler(self.lengths,mode='window4096',epoch=epoch,seed=42,batch_size=64,window=4096).batches
   groups=[]
   for start in range(0,len(base),3):
    group=base[start:start+3]
    if len(group)<3:group=[x.tolist() for x in np.array_split(np.concatenate(group),3)]
    assert all(2<=len(x)<=64 for x in group)
    groups.append(group)
   self.groups=groups;self.batches=[g[self.rank] for g in groups]
  else:
   ids=np.arange(len(self.lengths))[self.rank::3]
   self.batches=[x.tolist() for x in np.array_split(ids,math.ceil(len(ids)/64))]
  flat=np.concatenate(self.batches)
  assert len(np.unique(flat))==len(flat)
 def __len__(self):return len(self.batches)-self.cursor
 def __iter__(self):return iter(self.batches[self.cursor:])

def loader(arm,corpus,split,rank):
 rows=np.load(arm['data']['paths'][split]);ds=InMemoryFullSupportRowDataset(corpus,rows)
 ptr=ds._h5()['X']['indptr'];lengths=np.minimum(ptr[rows+1]-ptr[rows],512)
 sampler=Batches(lengths,rank,split=='train');epoch=old_ddp.ForkserverEpoch(0)
 collate=TensorCollator(DualCorruptionCollator(observed_genus_ids=np.load(arm['data']['observed_genus_ids']),
  seed=42,stream=1 if split=='train' else 2,shared_epoch=epoch))
 data=DataLoader(ds,batch_sampler=sampler,collate_fn=collate,num_workers=2,persistent_workers=True,
  multiprocessing_context='forkserver',pin_memory=True,prefetch_factor=2,generator=torch.Generator().manual_seed(7000+rank))
 return data,sampler,epoch,rows

def advance(ddp,opt,sched,batch,arm,device):
 stats,record=old_ddp.update(ddp,opt,sched,batch,arm,device)
 if record['gradient_norm_preclip']>1e4:raise RuntimeError('gradient norm exceeds contract')
 return stats,record

@torch.inference_mode()
def validate(model,arm,data,rows,device):
 model.eval();stats=torch.zeros(12,device=device,dtype=torch.float64);clean=torch.zeros(4,device=device,dtype=torch.float64)
 sum_h=torch.zeros(512,device=device,dtype=torch.float64);outer=torch.zeros((512,512),device=device,dtype=torch.float64);seen=[]
 for batch in data:
  seen.append(batch['row_ids'].numpy());args=batch_to_device(batch,device)
  with torch.autocast('cuda',dtype=torch.bfloat16):
   out=model(**args);_,mean=model.unmasked_representations(args['genus_ids'],args['rclr'],args['padding_mask'])
  ce,entropy,kl=neighborhood_details(mean,weighted_unifrac(args['edge_mass'],model.unifrac_lengths))
  stats+=common._loss_statistics_tensor(out,arm['weights'],device=device)
  n=len(mean);clean+=torch.stack([ce.double()*n,entropy.double()*n,kl.double()*n,torch.as_tensor(n,device=device)])
  h=mean.float().double();sum_h+=h.sum(0);outer+=h.T@h
 for x in [stats,clean,sum_h,outer]:dist.all_reduce(x)
 old_ddp.all_rows(seen,rows)
 return dict(loss=common._unpack_loss_statistics(stats,arm['weights']),row_count=len(rows),exact_row_coverage=True,
  clean_unifrac_diagnostics=dict(zip(['ce','teacher_entropy','kl'],(clean[:3]/clean[3]).cpu().tolist())),
  representation=common._spectral_summary(len(rows),sum_h.cpu(),outer.cpu()))

def save(path,model,opt,sched,arm,digest,device,rank,*,epoch,cursor,step,best,stats):
 rng=[None]*3;dist.all_gather_object(rng,capture_rng(device))
 if rank==0:
  state=w.checkpoint(model,opt,sched,arm,digest,device,epoch=epoch,cursor=cursor,global_step=step,best_score=best,statistics=stats)
  state.update(world_size=3,rng_by_rank=rng,local_batch=64,global_batch=192)
  common._atomic_torch(path,state)
 dist.barrier()

def restore(path,model,opt,sched,arm,digest,device,rank):
 state=w.restore(path,model,opt,sched,arm,digest,device)
 assert state['world_size']==3 and len(state['rng_by_rank'])==3
 restore_rng(state['rng_by_rank'][rank],device)
 return state

def clone(value):
 if torch.is_tensor(value):return value.detach().cpu().clone()
 if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
 if isinstance(value,list):return [clone(v) for v in value]
 if isinstance(value,tuple):return tuple(clone(v) for v in value)
 return value

def gate(arm,m,digest,root,rank,device):
 ds=InMemoryFullSupportRowDataset(m['assets']['corpus']['path'],np.load(arm['data']['paths']['train']))
 rows=np.load(arm['data']['paths']['train']);ptr=ds._h5()['X']['indptr'];lengths=np.minimum(ptr[rows+1]-ptr[rows],512);order=np.argsort(lengths,kind='stable')
 selected=np.r_[order[-1],order[rank*63:rank*63+63]]
 collate=TensorCollator(DualCorruptionCollator(observed_genus_ids=np.load(arm['data']['observed_genus_ids']),seed=42,stream=91,shared_epoch=old_ddp.ForkserverEpoch(0)))
 full=collate([ds[int(i)] for i in selected]);tail=collate([ds[int(i)] for i in selected[:20]])
 model=w.load_model(arm,m,device);frozen=frozen_sha256(model);ddp=old_ddp.wrap(model,device);opt,sched=w.make_optimizer(model,arm)
 common._seed_all(42+rank)
 stats,_=advance(ddp,opt,sched,full,arm,device)
 path=root/'audit/gates'/f'{arm["name"]}_resume.pt'
 save(path,model,opt,sched,arm,digest,device,rank,epoch=0,cursor=1,step=1,best=math.inf,stats=stats)
 _,expected_record=advance(ddp,opt,sched,tail,arm,device)
 expected=(clone(model.state_dict()),clone(opt.state_dict()),sched.state_dict());peak=torch.cuda.max_memory_allocated(device)/2**30
 del ddp,model,opt,sched;gc.collect();torch.cuda.empty_cache();dist.barrier()
 model=w.load_model(arm,m,device);opt,sched=w.make_optimizer(model,arm)
 state=restore(path,model,opt,sched,arm,digest,device,rank);assert state['global_step']==1;del state
 ddp=old_ddp.wrap(model,device);_,actual_record=advance(ddp,opt,sched,tail,arm,device)
 exact=state_equal(expected[0],model.state_dict()) and state_equal(expected[1],opt.state_dict()) and expected[2]==sched.state_dict() and expected_record==actual_record
 max_abs=max(float((expected[0][k]-v.detach().cpu()).abs().max()) for k,v in model.state_dict().items() if v.is_floating_point())
 summaries=[None]*3;dist.all_gather_object(summaries,dict(rank=rank,exact=exact,max_abs=max_abs,peak_gib=peak))
 if not all(x['exact'] for x in summaries):raise RuntimeError('DDP restore-next-update is not exact: '+str(summaries))
 assert frozen_sha256(model)==frozen
 if rank==0:common._atomic_json(root/'audit/gates'/f'{arm["name"]}.json',dict(status='PASS',time=w.now(),manifest_sha256=digest,world_size=3,local_batch=64,global_batch=192,
  longest_input=int(full['genus_ids'].shape[1]),trainable_parameters=12582912,frozen_origin_equal=True,strict_restore_next_update_exact=True,ranks=summaries,engineering_only=True))

def train(arm,m,digest,root,rank,device):
 assert json.loads((root/'audit/gates'/f'{arm["name"]}.json').read_text())['status']=='PASS'
 out=Path(arm['output']);corpus=m['assets']['corpus']['path']
 if (out/'run/last.pt').exists():raise RuntimeError('existing state requires separately authorized explicit recovery')
 model=w.load_model(arm,m,device);ddp=old_ddp.wrap(model,device);opt,sched=w.make_optimizer(model,arm);common._seed_all(42+rank)
 train_data,sampler,epoch_value,rows=loader(arm,corpus,'train',rank);val_data,_,_,val_rows=loader(arm,corpus,'val',rank)
 assert len(sampler)==arm['steps_per_epoch'];stop=[]
 for sig in [signal.SIGTERM,signal.SIGUSR1,signal.SIGINT]:signal.signal(sig,lambda n,f:stop.append(n))
 step=0;best=math.inf;stats=torch.zeros(12,device=device,dtype=torch.float64)
 if rank==0:common._atomic_json(out/'audit/formal_launch.json',dict(status='RUNNING',time=w.now(),world_size=3,local_batch=64,global_batch=192,total_steps=arm['total_steps'],manifest_sha256=digest,machine='CmlLab',node='gpu1',step=os.environ.get('SLURM_STEP_ID')))
 for epoch in range(50):
  sampler.set_epoch(epoch);epoch_value.set(epoch);seen=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
  for cursor,batch in enumerate(train_data,1):
   assert int(batch['corruption_epoch'])==epoch
   batch_stats,record=advance(ddp,opt,sched,batch,arm,device);stats+=batch_stats;step+=1;seen.append(batch['row_ids'].numpy())
   if rank==0 and (step==1 or step%20==0):
    common._append_jsonl(out/'logs/progress.jsonl',dict(time=w.now(),epoch=epoch+1,global_step=step,next_batch=cursor,**record))
    print(f'{w.now()} {arm["name"]} epoch={epoch+1}/50 step={step}/{arm["total_steps"]} loss={record["objective"]:.6f}',flush=True)
   flag=torch.tensor(bool(stop),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
   if step%500==0 or bool(flag):save(out/'run/last.pt',model,opt,sched,arm,digest,device,rank,epoch=epoch,cursor=cursor,step=step,best=best,stats=stats)
   if bool(flag):return
  old_ddp.all_rows(seen,rows);assert step==(epoch+1)*arm['steps_per_epoch']
  validation=validate(model,arm,val_data,val_rows,device);assert frozen_sha256(model)==arm['initialization']['frozen_parameter_sha256']
  score=validation['loss']['weighted_total'];improved=score<best;best=min(best,score)
  result=dict(time=w.now(),epoch=epoch+1,global_step=step,seconds=time.perf_counter()-start,train=common._unpack_loss_statistics(stats,arm['weights']),validation=validation,exact_train_row_coverage=True,frozen_origin_equal=True,world_size=3,local_batch=64,global_batch=192)
  if rank==0:common._append_jsonl(out/'results/epochs.jsonl',result)
  stats.zero_();save(out/'run/last.pt',model,opt,sched,arm,digest,device,rank,epoch=epoch+1,cursor=0,step=step,best=best,stats=stats)
  if improved:save(out/'run/best.pt',model,opt,sched,arm,digest,device,rank,epoch=epoch+1,cursor=0,step=step,best=best,stats=stats)
  if epoch+1 in [10,20,30,50]:save(out/f'run/epoch{epoch+1:02d}.pt',model,opt,sched,arm,digest,device,rank,epoch=epoch+1,cursor=0,step=step,best=best,stats=stats)
 if rank==0:
  checkpoint=out/'run/epoch50.pt'
  common._atomic_json(root/'audit/accepted'/f'{arm["name"]}.json',dict(status='PASS',time=w.now(),model=arm['name'],epoch=50,global_step=step,world_size=3,global_batch=192,frozen_origin_equal=True,checkpoint=str(checkpoint),checkpoint_sha256=common.sha256_file(checkpoint)))

def run(root,arm_name,action,uuids):
 root=Path(root);path=root/'audit/legacy_manifest.json';m=json.loads(path.read_text());digest=common.sha256_file(path);arm=m['arms'][arm_name]
 ready=json.loads((root/'audit/legacy_ready.json').read_text());assert ready['manifest_sha256']==digest
 for p,h in ready['source_files'].items():
  if common.sha256_file(p)!=h:raise RuntimeError('sealed legacy DDP source changed: '+p)
 rank,device=initialise(root,uuids)
 try:(gate if action=='gate' else train)(arm,m,digest,root,rank,device)
 finally:dist.destroy_process_group()
