"""Seal four LoRA+ parameter contrasts and one fixed 100-epoch schedule."""
from __future__ import annotations
import copy, gc, json, os, socket, subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from micoformer.domain_adaptation.prepare import load_origin, verify_file, file_record
from micoformer.domain_method_adaptation.modules import inject, frozen_sha256
from micoformer.mpc_unifrac import workflow as common

REPO=Path(__file__).resolve().parents[2]
PROJECT=REPO.parent
RUN=Path(os.environ.get('MCF_LONG_ROOT',str(PROJECT/'tmp/20261025_respiratory_loraplus_search')))
EVAL=RUN/'evaluation'
BASE='28de0ee4f2b363cb4130f86064a542c3e3666fea'
GIT='/home/cml_lab/anaconda3/bin/git'
REFERENCE=PROJECT/'tmp/20261024_respiratory_peft_mechanism_pilot/h100_long'
REFERENCE_SHA='54ba71f86a75f72d627f03445e18f7421e25891f89a9ecc733cc6984e823a5d4'
NAMES=['loraplus_r64_lrhalf','loraplus_r64_lrdouble','loraplus_r64_ratio8','loraplus_r64_dropout0','loraplus_r64_long100']
MILESTONES=[50,100]

def configurations():
    return {
        NAMES[0]:dict(epochs=50,learning_rate=2.5e-5,lr_ratio=16.,minimum_lr=5e-7,dropout=.05),
        NAMES[1]:dict(epochs=50,learning_rate=1e-4,lr_ratio=16.,minimum_lr=2e-6,dropout=.05),
        NAMES[2]:dict(epochs=50,learning_rate=1e-4,lr_ratio=8.,minimum_lr=2e-6,dropout=.05),
        NAMES[3]:dict(epochs=50,learning_rate=5e-5,lr_ratio=16.,minimum_lr=1e-6,dropout=0.),
        NAMES[4]:dict(epochs=100,learning_rate=5e-5,lr_ratio=16.,minimum_lr=1e-6,dropout=.05)}

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()

def configure_tree():
    from micoformer.mpc_unifrac import teacher,model,data
    teacher.ASSETS=model.ASSETS=data.ASSETS=RUN/'assets/tree'

def specifications():
    return {name:dict(method='loraplus',rank=64,alpha=128.,dropout=c['dropout'])
            for name,c in configurations().items()}

def bind_manifest(raw,root=None):
    root=Path(root or RUN);m=copy.deepcopy(raw)
    for record in [*m['assets'].values(),m['contract']]:record['path']=str(root/record['path'])
    for arm in m['arms'].values():
        arm['output']=str(root/arm['output'])
        arm['data']['paths']={k:str(root/v) for k,v in arm['data']['paths'].items()}
        for k in ['observed_genus_ids','smoke_rows']:arm['data'][k]=str(root/arm['data'][k])
    return m

def source_record():
    def git(*args):return subprocess.check_output([GIT,'-C',str(REPO),*args],text=True).strip()
    if git('status','--porcelain'):raise RuntimeError('commit preparation code before sealing')
    subprocess.run([GIT,'-C',str(REPO),'merge-base','--is-ancestor',BASE,'HEAD'],check=True)
    names=subprocess.check_output([GIT,'-C',str(REPO),'ls-files','-z']).decode().split('\0')
    return dict(base_commit=BASE,commit=git('rev-parse','HEAD'),worktree=str(REPO),
        inherited_method_commit='9e552a7360bb63ba8d8f9f8cfd178a2e85842bbf',
        files={n:common.sha256_file(REPO/n) for n in names if n})

def apply_eva(model,arm,manifest):
    if arm['adaptation'].get('initialization')!='eva_fixedrank':return
    from .eva import apply_initialization
    record=manifest['assets']['eva_initialization'];verify_file(record)
    apply_initialization(model,Path(record['path']))

def verify(path=None,expected_sha=None):
    path=Path(path or RUN/'audit/manifest.json')
    expected_sha=expected_sha or (path.parent/'manifest.sha256').read_text().strip()
    if common.sha256_file(path)!=expected_sha:raise RuntimeError('sealed manifest changed')
    raw=json.loads(path.read_text())
    if raw['method']!='respiratory_loraplus_search' or list(raw['arms'])!=NAMES:raise RuntimeError('unexpected recipe')
    for name,digest in raw['source']['files'].items():
        if common.sha256_file(REPO/name)!=digest:raise RuntimeError('sealed source changed: '+name)
    m=bind_manifest(raw)
    for record in [*m['assets'].values(),m['contract']]:verify_file(record)
    for name,arm in m['arms'].items():
        cfg=configurations()[name];epochs=cfg['epochs']
        if (arm['epochs'],arm['total_steps'],arm['steps_per_epoch'],arm['microbatch'],arm['accumulate'],arm['seed'],arm['window'])!=(epochs,epochs*867,867,128,1,42,4096):raise RuntimeError('approved budget changed')
        for key in ['learning_rate','minimum_lr','lr_ratio']:
            if arm[key]!=cfg[key]:raise RuntimeError('approved LR trajectory changed')
        if arm['adaptation']!=specifications()[name] or arm['evaluation_epochs']!=([50,100] if epochs==100 else [50]):raise RuntimeError('approved adaptation/endpoints changed')
    return m,expected_sha

def stage_assets():
    path=REFERENCE/'audit/manifest.json'
    if common.sha256_file(path)!=REFERENCE_SHA:raise RuntimeError('accepted reference changed')
    reference=json.loads(path.read_text())
    for record in reference['assets'].values():
        src=REFERENCE/record['path'];verify_file(dict(record,path=str(src)))
        target=RUN/record['path'];target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.symlink_to(src.resolve())
    return reference

def prepare():
    target=RUN/'audit/manifest.json'
    if target.exists():raise FileExistsError(target)
    reference=stage_assets();assets=copy.deepcopy(reference['assets'])
    train=np.load(RUN/assets['respiratory_train']['path']);val=np.load(RUN/assets['respiratory_val']['path'])
    if len(train)!=110868 or len(val)!=571 or len(np.unique(train))!=len(train) or len(np.intersect1d(train,val)):raise RuntimeError('roster changed')
    configure_tree();arms={}
    bound=dict(assets={k:dict(v,path=str(RUN/v['path'])) for k,v in assets.items()})
    expected=[6291456]*len(NAMES)
    for name,count in zip(NAMES,expected,strict=True):
        common._seed_all(42)
        model=load_origin(bound['assets']['prior']['path'],Path(bound['assets']['origin']['path']))
        origin=common._model_state_sha256(model)
        if origin!=reference['arms']['loraplus_r64']['initialization']['origin_model_state_sha256']:raise RuntimeError('origin changed')
        common._seed_all(42);spec=specifications()[name];inv=inject(model,spec)
        arm=copy.deepcopy(reference['arms']['loraplus_r64'])
        cfg=configurations()[name]
        arm.update(name=name,output='runs/'+name,adaptation=spec,
            **{k:v for k,v in cfg.items() if k!='dropout'})
        arm['total_steps']=arm['epochs']*arm['steps_per_epoch']
        arm['evaluation_epochs']=[50,100] if arm['epochs']==100 else [50]
        arm['milestones']=[10,20,30,50]+([100] if arm['epochs']==100 else [])
        if inv['counts']['adaptation']!=count:raise RuntimeError('incorrect parameter budget: '+name)
        arm['trainable_inventory']=inv
        arm['initialization']=dict(origin_model_state_sha256=origin,adapted_model_state_sha256=common._model_state_sha256(model),frozen_parameter_sha256=frozen_sha256(model))
        if arm['initialization']!=reference['arms']['loraplus_r64']['initialization']:raise RuntimeError('LR-only arm changed initialization')
        arms[name]=arm;del model;gc.collect()
    manifest=dict(schema_version=1,method='respiratory_loraplus_search',prepared_at=now(),machine='CmlLab',node=socket.gethostname(),status='SEALED',
        source=source_record(),assets=assets,arms=arms,contract=dict(file_record(RUN/'PLAN.md'),path='PLAN.md'),
        provenance=dict(reference_manifest=str(REFERENCE/'audit/manifest.json'),reference_sha256=REFERENCE_SHA),
        execution=dict(machine='H100',host='NVH1003',world_size=1,batch_size=128,precision='BF16 autocast; FP32 parameters',
            activation_recomputation=False,cpu_threads=2,dataloader_workers=4,optimizer='fused AdamW',validation_seed=42,
            checkpoints=[0,10,20,30,50,100],evaluation_epochs=[50,100],placeholder_673251_untouched=True))
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    digest=common.sha256_file(target);(target.parent/'manifest.sha256').write_text(digest+'\n')
    common._atomic_json(RUN/'audit/SOURCE.json',manifest['source'])
    print(json.dumps(dict(status='SEALED',manifest_sha256=digest,source=manifest['source']['commit'],arms=NAMES)),flush=True)
    return manifest
