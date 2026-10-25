"""Bind the expanded cohort and a two-by-two LR/presence experiment."""
from pathlib import Path
import copy,json,os,socket
import h5py,numpy as np,pandas as pd,torch
from micoformer.domain_long_adaptation.prepare import RUN,REPO,PROJECT,BASE,bind_manifest,source_record,now,configure_tree
from micoformer.domain_adaptation.prepare import verify_file,load_origin
from micoformer.domain_method_adaptation.modules import inject,frozen_sha256
from micoformer.mpc_unifrac import workflow as common

UUID='GPU-f412ca27-bee5-f140-4344-ebbe13b2438a'
NAMES=['digestive_loraplus','digestive_lrhalf','digestive_presencehalf','digestive_lrhalf_presencehalf']
REFERENCE_WEIGHTS=dict(mlm=1.,continuous=.534399128791011,presence=.24296011504612833,unifrac=5e-5)

def expected_specs():
 return {name:dict(lr_scale=.5 if 'lrhalf' in name else 1.,presence_scale=.5 if 'presencehalf' in name else 1.) for name in NAMES}

def verify():
 if RUN.name!='20261025_digestive_l3_adaptation':raise RuntimeError('explicit digestive MCF_LONG_ROOT required')
 path=RUN/'audit/manifest.json';sha=(RUN/'audit/manifest.sha256').read_text().strip()
 if common.sha256_file(path)!=sha:raise RuntimeError('digestive manifest changed')
 raw=json.loads(path.read_text())
 if raw['method']!='digestive_l3_loraplus' or list(raw['arms'])!=NAMES:raise RuntimeError('wrong experiment/arms')
 for name,h in raw['source']['files'].items():
  if common.sha256_file(REPO/name)!=h:raise RuntimeError('sealed source changed: '+name)
 m=bind_manifest(raw)
 for r in [*m['assets'].values(),m['contract']]:verify_file(r)
 for name,spec in expected_specs().items():
  a=m['arms'][name];weights=dict(REFERENCE_WEIGHTS);weights['presence']*=spec['presence_scale']
  expected=dict(epochs=50,microbatch=128,accumulate=1,seed=42,window=4096,steps_per_epoch=3762,total_steps=188100,
   learning_rate=5e-5*spec['lr_scale'],minimum_lr=1e-6*spec['lr_scale'],lr_ratio=16.,head_learning_rate=0.,weights=weights)
  if any(a[k]!=v for k,v in expected.items()):raise RuntimeError('frozen factorial changed: '+name)
  if a['adaptation']!=dict(method='loraplus',rank=64,alpha=128.,dropout=.05):raise RuntimeError('method scope changed')
 return m,sha

def prepare():
 if RUN.name!='20261025_digestive_l3_adaptation':raise RuntimeError('explicit digestive MCF_LONG_ROOT required')
 if (RUN/'audit/manifest.json').exists():raise FileExistsError('already sealed')
 torch.set_num_threads(2)
 oldroot=PROJECT/'tmp/20261025_gut_loraplus_h100_long';oldpath=oldroot/'audit/manifest.json'
 if common.sha256_file(oldpath)!='a2f9fe85c12f53d49d651b63689744732faa5d5a4c62ec6992a9d48ce6a5bc3c':raise RuntimeError('gut reference changed')
 old=json.loads(oldpath.read_text());cohort=json.loads((RUN/'audit/cohort_acceptance.json').read_text())
 if cohort['status']!='PASS':raise RuntimeError('cohort gate missing')
 for n,h in cohort['files'].items():
  if common.sha256_file(RUN/n)!=h:raise RuntimeError('cohort changed: '+n)
 summary=json.loads((RUN/'results/cohort_summary.json').read_text())
 if (summary['train']['rows'],summary['val']['rows'])!=(481487,2492):raise RuntimeError('cohort scope changed')
 assets={}
 for name in ['corpus','origin','prior','corpus_release','tree_genus_to_edge.npz','tree_branch_lengths.npy']:
  r=dict(old['assets'][name]);r['path']=str(oldroot/r['path']);verify_file(r)
  relative=old['assets'][name]['path'];dest=RUN/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() or dest.is_symlink():raise FileExistsError(dest)
  dest.symlink_to(Path(r['path']).resolve());assets[name]=dict(path=relative,sha256=r['sha256'],bytes=r['bytes'])
 train=np.load(RUN/'data/digestive_train.npy');val=np.load(RUN/'data/digestive_val.npy')
 table=pd.read_csv(RUN/'data/digestive_rows.tsv.gz',sep='\t')
 tr=set(table.loc[table.split=='train','Run'].dropna());va=set(table.loc[table.split=='val','Run'].dropna())
 if tr&va:raise RuntimeError('Run crosses train/val')
 with h5py.File(RUN/assets['corpus']['path'],'r') as f:
  ptr=f['X/indptr'][:];richness=np.diff(ptr);mask=np.zeros(len(richness),bool);mask[train]=True;seen=np.zeros(8114,bool)
  for first in range(0,len(richness),8192):
   stop=min(first+8192,len(richness))
   if not mask[first:stop].any():continue
   indices=f['X/indices'][int(ptr[first]):int(ptr[stop])];keep=np.repeat(mask[first:stop],richness[first:stop]);seen[indices[keep]]=True
  if richness[np.r_[train,val]].min()<5:raise RuntimeError('minimum support changed')
  smoke=train[np.lexsort((train,-richness[train]))[:128]]
 np.save(RUN/'data/digestive_observed_genus_ids.npy',np.flatnonzero(seen).astype(np.int64)+2,allow_pickle=False)
 np.save(RUN/'data/digestive_smoke_rows.npy',smoke,allow_pickle=False)
 for key,path in [('digestive_'+k,'data/digestive_'+k+'.npy') for k in ['all','train','val']]+[
  ('digestive_observed','data/digestive_observed_genus_ids.npy'),('digestive_smoke_rows','data/digestive_smoke_rows.npy'),('digestive_row_table','data/digestive_rows.tsv.gz')]:
  p=RUN/path;assets[key]=dict(path=path,bytes=p.stat().st_size,sha256=common.sha256_file(p))
 configure_tree();saved=old['arms']['gut_loraplus_r64'];common._seed_all(42)
 model=load_origin(str(RUN/assets['prior']['path']),RUN/assets['origin']['path']);origin=common._model_state_sha256(model)
 common._seed_all(42);inv=inject(model,saved['adaptation'])
 initial=dict(origin_model_state_sha256=origin,adapted_model_state_sha256=common._model_state_sha256(model),frozen_parameter_sha256=frozen_sha256(model))
 if initial!=saved['initialization'] or inv!=saved['trainable_inventory']:raise RuntimeError('initialization differs from accepted LoRA+')
 arms={}
 for name,spec in expected_specs().items():
  a=copy.deepcopy(saved);a.update(name=name,output='runs/'+name,steps_per_epoch=3762,total_steps=188100,
    learning_rate=5e-5*spec['lr_scale'],minimum_lr=1e-6*spec['lr_scale'],extract_on_milestone=False)
  a['weights']['presence']*=spec['presence_scale']
  a['data']=dict(rows=483979,train=481487,val=2492,paths={k:assets['digestive_'+k]['path'] for k in ['all','train','val']},
    observed_genus_ids=assets['digestive_observed']['path'],smoke_rows=assets['digestive_smoke_rows']['path'],observed_genus_count=int(seen.sum()),site_counts=summary['all']['gold_l4'])
  arms[name]=a
 contract=dict(path='PLAN.md',bytes=(RUN/'PLAN.md').stat().st_size,sha256=common.sha256_file(RUN/'PLAN.md'))
 raw=dict(schema_version=1,method='digestive_l3_loraplus',prepared_at=now(),status='SEALED',machine='CmlLab',node=socket.gethostname(),
  source=source_record(),assets=assets,arms=arms,contract=contract,reference_weights=REFERENCE_WEIGHTS,
  provenance=dict(legacy_manifest=str(oldpath),legacy_manifest_sha256=common.sha256_file(oldpath),cohort_sha256=common.sha256_file(RUN/'audit/cohort_acceptance.json')),
  execution=dict(machine='H100',host='NVH1003',gpu_uuid=UUID,world_size=1,queue=NAMES,cpu_threads=2,dataloader_workers=4,
   numerical_gates_before_each_arm=True,evaluation_epochs={n:[10,20,30,50] if n==NAMES[0] else [50] for n in NAMES},
   finite_queue_hours_limit=36,placeholder_673251_untouched=True))
 path=RUN/'audit/manifest.json';path.write_text(json.dumps(raw,ensure_ascii=False,indent=2)+'\n')
 sha=common.sha256_file(path);(RUN/'audit/manifest.sha256').write_text(sha+'\n')
 common._atomic_json(RUN/'audit/cpu_acceptance.json',dict(status='PASS',time=now(),manifest_sha256=sha,source_commit=raw['source']['commit'],
   same_initialization_as_old_lora=True,observed_genus_count=int(seen.sum()),train=481487,val=2492,Run_overlap=0,factorial=expected_specs()))
 print(json.dumps(dict(status='SEALED',manifest_sha256=sha,source=raw['source']['commit'])),flush=True)

if __name__=='__main__':prepare()
