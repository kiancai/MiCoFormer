"""Portable sealed 50-epoch respiratory recipe; paths bind to one task root."""
from __future__ import annotations
import copy
import gc
import json
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
from micoformer.domain_adaptation.prepare import load_origin, verify_file, file_record
from micoformer.domain_method_adaptation.modules import inject, frozen_sha256
from micoformer.mpc_unifrac import workflow as common

REPO=Path(__file__).resolve().parents[2]
PROJECT=REPO.parent
RUN=Path(os.environ.get('MCF_LONG_ROOT',str(PROJECT/'tmp/20261024_respiratory_peft_mechanism_pilot/h100_long')))
EVAL=RUN/'evaluation'
BASE='28de0ee4f2b363cb4130f86064a542c3e3666fea'
GIT='/home/cml_lab/anaconda3/bin/git'
NAMES=['loraplus_r64','lora_r128','dora_r64','adapter_b256']
MILESTONES=[10,20,30,50]


def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def configure_tree():
    from micoformer.mpc_unifrac import teacher, model, data
    teacher.ASSETS=model.ASSETS=data.ASSETS=RUN/'assets/tree'


def bind_manifest(raw,root=None):
    root=Path(root or RUN)
    m=copy.deepcopy(raw)
    for record in [*m['assets'].values(),m['contract']]:record['path']=str(root/record['path'])
    for arm in m['arms'].values():
        arm['output']=str(root/arm['output'])
        arm['data']['paths']={k:str(root/v) for k,v in arm['data']['paths'].items()}
        for k in ['observed_genus_ids','smoke_rows']:arm['data'][k]=str(root/arm['data'][k])
    return m


def source_record():
    def git(*args):return subprocess.check_output([GIT,'-C',str(REPO),*args],text=True).strip()
    if git('status','--porcelain'):raise RuntimeError('commit implementation before sealing')
    subprocess.run([GIT,'-C',str(REPO),'merge-base','--is-ancestor',BASE,'HEAD'],check=True)
    names=subprocess.check_output([GIT,'-C',str(REPO),'ls-files','-z']).decode().split('\0')
    return dict(base_commit=BASE,commit=git('rev-parse','HEAD'),worktree=str(REPO),
        inherited_method_commit='3b55cd57e7fc4d08673bc9047d1d74753b4cd702',
        files={n:common.sha256_file(REPO/n) for n in names if n})


def verify(path=None,expected_sha=None):
    path=Path(path or RUN/'audit/manifest.json')
    expected_sha=expected_sha or (path.parent/'manifest.sha256').read_text().strip()
    if common.sha256_file(path)!=expected_sha:raise RuntimeError('sealed manifest changed')
    raw=json.loads(path.read_text())
    if raw['method']!='respiratory_h100_long' or list(raw['arms'])!=NAMES:raise RuntimeError('unexpected recipe')
    for name,digest in raw['source']['files'].items():
        if common.sha256_file(REPO/name)!=digest:raise RuntimeError('sealed source changed: '+name)
    m=bind_manifest(raw)
    for record in [*m['assets'].values(),m['contract']]:verify_file(record)
    for arm in m['arms'].values():
        if (arm['epochs'],arm['total_steps'],arm['steps_per_epoch'],arm['microbatch'],arm['accumulate'],arm['seed'],arm['window'])!=(50,43350,867,128,1,42,4096):raise RuntimeError('approved budget changed')
    return m,expected_sha


def prepare(dora_key):
    target=RUN/'audit/manifest.json'
    if target.exists():raise FileExistsError(target)
    ref=PROJECT/'tmp/20261024_respiratory_peft_mechanism_pilot/a100_parallel/pairs/r64_seed42/audit/manifest.json'
    if common.sha256_file(ref)!='60e0e437ed7969acdd45f248d8a365a40865e63a456f33f317347f7fa5f61e21':raise RuntimeError('reference changed')
    reference=json.loads(ref.read_text());assets={}
    destinations={'corpus':'assets/corpus.h5ad','origin':'assets/checkpoint.pt','prior':'assets/prior.npz',
        'reference':'assets/reference.json','tree_genus_to_edge.npz':'assets/tree/genus_to_edge.npz',
        'tree_branch_lengths.npy':'assets/tree/branch_lengths.npy'}
    for name,record in reference['assets'].items():
        verify_file(record)
        relative=destinations.get(name,'assets/data/'+Path(record['path']).name)
        p=RUN/relative;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.symlink_to(record['path'])
        if common.sha256_file(p)!=record['sha256']:raise RuntimeError('asset link mismatch')
        assets[name]=dict(path=relative,bytes=record['bytes'],sha256=record['sha256'])
    train=np.load(RUN/assets['respiratory_train']['path']);val=np.load(RUN/assets['respiratory_val']['path'])
    if len(train)!=110868 or len(val)!=571 or len(np.unique(train))!=110868 or len(np.intersect1d(train,val)):raise RuntimeError('roster changed')
    contract=file_record(RUN/'PLAN.md');contract['path']='PLAN.md'
    configure_tree();arms={};selection={}
    old=PROJECT/'tmp/20261024_respiratory_peft_mechanism_pilot/method_screen'
    mapping={'loraplus_r64':('loraplus','loraplus_r64_ratio16'), 'dora_r64':('dora',dora_key), 'adapter_b256':('adapter','adapter_b256_lr1e4')}
    for name in NAMES:
        if name=='lora_r128':
            p=PROJECT/'tmp/20261024_respiratory_peft_mechanism_pilot/a100_bucket/audit/manifest.json'
            saved=json.loads(p.read_text())['arms']['window4096'];spec=dict(method='lora',**saved['lora']);lr=1e-4;ratio=1.
        else:
            family,key=mapping[name];p=old/'families'/family/'audit/manifest.json'
            saved=json.loads(p.read_text())['arms'][key];spec=saved['adaptation'];lr=saved['learning_rate'];ratio=saved['lr_ratio']
        common._seed_all(42)
        model=load_origin(str(RUN/assets['prior']['path']),RUN/assets['origin']['path'])
        origin=common._model_state_sha256(model);common._seed_all(42);inv=inject(model,spec)
        initial=dict(origin_model_state_sha256=origin,adapted_model_state_sha256=common._model_state_sha256(model),frozen_parameter_sha256=frozen_sha256(model))
        if initial!=saved['initialization']:raise RuntimeError('initial state differs from accepted method: '+name)
        arm=copy.deepcopy(reference['arms']['window4096']);arm.pop('lora')
        arm.update(name=name,output='runs/'+name,sampler='window4096',epochs=50,total_steps=43350,milestones=[0,10,20,30,50],
            workers=4,learning_rate=lr,lr_ratio=ratio,adaptation=spec,initialization=initial,trainable_inventory=inv)
        arm['data']=dict(paths=dict(train=assets['respiratory_train']['path'],val=assets['respiratory_val']['path']),
            observed_genus_ids=assets['respiratory_observed']['path'],smoke_rows=assets['respiratory_smoke_rows']['path'])
        arms[name]=arm;selection[name]=dict(path=str(p),sha256=common.sha256_file(p),old_arm=saved['name'])
        del model;gc.collect()
    manifest=dict(schema_version=1,method='respiratory_h100_long',prepared_at=now(),machine='CmlLab',node=socket.gethostname(),status='SEALED',
        source=source_record(),assets=assets,arms=arms,contract=contract,provenance=selection,
        execution=dict(machine='H100',host='NVH1003',world_size=1,batch_size=128,precision='BF16 autocast; FP32 parameters',
            activation_recomputation=False,cpu_threads=2,dataloader_workers=4,optimizer='fused AdamW',
            validation_seed=42,checkpoints=MILESTONES,placeholder_673251_untouched=True))
    # Preserve the explicit arm order, independent of atomic_json's sorted keys.
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    digest=common.sha256_file(target);(target.parent/'manifest.sha256').write_text(digest+'\n')
    common._atomic_json(RUN/'audit/SOURCE.json',manifest['source'])
    print(json.dumps(dict(status='SEALED',manifest_sha256=digest,source=manifest['source']['commit'],dora=dora_key)),flush=True)
    return manifest
