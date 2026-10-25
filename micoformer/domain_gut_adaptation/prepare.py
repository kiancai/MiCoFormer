"""Seal the existing GOLD gut roster and the accepted respiratory LoRA+ recipe."""
from __future__ import annotations

import copy
import json
import os
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from micoformer.domain_adaptation.prepare import load_origin, verify_file, file_record
from micoformer.domain_method_adaptation.modules import inject, frozen_sha256
from micoformer.domain_long_adaptation.prepare import bind_manifest, source_record, configure_tree, now
from micoformer.mpc_unifrac import workflow as common

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
RUN = Path(os.environ.get('MCF_LONG_ROOT', str(PROJECT/'tmp/20261025_gut_loraplus_h100_long')))
ARM = 'gut_loraplus_r64'
UUID = 'GPU-f412ca27-bee5-f140-4344-ebbe13b2438a'
WEIGHTS = dict(mlm=1., continuous=.534399128791011, presence=.24296011504612833, unifrac=5e-5)


def check_contract(raw):
    if raw['method'] != 'gut_h100_loraplus_long' or list(raw['arms']) != [ARM]:
        raise RuntimeError('wrong domain/arm')
    a = raw['arms'][ARM]
    expected = dict(domain='gut', epochs=50, microbatch=128, accumulate=1, seed=42,
                    window=4096, sampler='window4096', steps_per_epoch=2304, total_steps=115200,
                    learning_rate=5e-5, lr_ratio=16., head_learning_rate=0., minimum_lr=1e-6,
                    warmup_ratio=.02, weight_decay=.01, gradient_clip=1., workers=4,
                    extract_on_milestone=False, milestones=[0,10,20,30,50])
    differences={k:a.get(k) for k,v in expected.items() if a.get(k)!=v}
    if differences:
        raise RuntimeError('frozen gut recipe changed: '+repr(differences))
    from micoformer.mpc_unifrac.teacher import STUDENT_T, TEACHER_T
    if (STUDENT_T,TEACHER_T)!=(.05,.03):
        raise RuntimeError('actual neighborhood temperatures changed')
    if a['adaptation'] != dict(method='loraplus',rank=64,alpha=128.,dropout=.05) or a['weights'] != WEIGHTS:
        raise RuntimeError('method or objective changed')
    if a['trainable_inventory']['counts'] != dict(adaptation=6291456,heads=559747,frozen=37864576):
        raise RuntimeError('trainable scope changed')
    if raw['execution']['gpu_uuid'] != UUID or raw['execution']['world_size'] != 1:
        raise RuntimeError('wrong approved GPU')
    if a['data']['train'] != 294906 or a['data']['val'] != 1539 or a['data']['rows'] != 296445:
        raise RuntimeError('gut sample scope changed')
    if (raw['assets']['corpus']['sha256'] != '1bacb2ee3fc88006ad472c500df585997380a98bf88b37e52da04df4cedc0b41'
            or raw['assets']['origin']['sha256'] != '0eeef0cc83fae83217a5d6808629b840c919bc77de7824a4fe580479fced289c'):
        raise RuntimeError('corpus or starting checkpoint changed')


def verify():
    path = RUN/'audit/manifest.json'
    sha = (RUN/'audit/manifest.sha256').read_text().strip()
    if common.sha256_file(path) != sha:
        raise RuntimeError('sealed manifest changed')
    raw = json.loads(path.read_text())
    check_contract(raw)
    for name,digest in raw['source']['files'].items():
        if common.sha256_file(REPO/name) != digest:
            raise RuntimeError('sealed source changed: '+name)
    bound = bind_manifest(raw, RUN)
    for record in [*bound['assets'].values(),bound['contract']]:
        verify_file(record)
    return bound,sha


def prepare():
    if (RUN/'audit/manifest.json').exists():
        raise FileExistsError('do not reseal an existing experiment')
    torch.set_num_threads(2)
    respiratory = PROJECT/'tmp/20261024_respiratory_peft_mechanism_pilot/h100_long/audit/manifest.json'
    if common.sha256_file(respiratory) != '54ba71f86a75f72d627f03445e18f7421e25891f89a9ecc733cc6984e823a5d4':
        raise RuntimeError('accepted method reference changed')
    ref = json.loads(respiratory.read_text())
    old_path = PROJECT/'tmp/20261022_domain_adaptive_pretraining/frozen_heads/audit/lora_run_manifest_ddp3_exact.json'
    old = json.loads(old_path.read_text())
    saved = ref['arms']['loraplus_r64']
    assets = {}
    selected = ['corpus','origin','prior','corpus_release','gut_all','gut_train','gut_val',
                'gut_observed','gut_smoke_rows','gut_row_table','canonical_train','canonical_val']
    for name in selected + ['tree_genus_to_edge.npz','tree_branch_lengths.npy']:
        if name.startswith('tree_'):
            record = dict(ref['assets'][name]);record['path'] = str(respiratory.parents[1]/record['path'])
        else:
            record = old['assets'][name]
        verify_file(record)
        if name.startswith('tree_'):
            relative = 'assets/tree/'+Path(record['path']).name
        elif name in ['corpus','origin','prior']:
            relative = {'corpus':'assets/corpus.h5ad','origin':'assets/checkpoint.pt','prior':'assets/prior.npz'}[name]
        else:
            relative = 'assets/data/'+Path(record['path']).name
        dest = RUN/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.is_symlink():
            if dest.resolve()!=Path(record['path']).resolve():
                raise RuntimeError('existing asset link has another source')
        else:
            if dest.exists():raise FileExistsError(dest)
            dest.symlink_to(record['path'])
        assets[name] = dict(path=relative,bytes=record['bytes'],sha256=record['sha256'])
    arrays = {k:np.load(RUN/assets['gut_'+k]['path'],allow_pickle=False) for k in ['all','train','val']}
    for k,n in [('all',296445),('train',294906),('val',1539)]:
        if len(arrays[k]) != n or len(np.unique(arrays[k])) != n:
            raise RuntimeError('wrong or duplicated gut roster')
    if (len(np.intersect1d(arrays['train'],arrays['val']))
            or not np.array_equal(np.sort(np.r_[arrays['train'],arrays['val']]),np.sort(arrays['all']))):
        raise RuntimeError('gut roster partition mismatch')
    for k in ['train','val']:
        if not np.isin(arrays[k],np.load(RUN/assets['canonical_'+k]['path']),assume_unique=True).all():
            raise RuntimeError('canonical split changed')
    table = pd.read_csv(RUN/assets['gut_row_table']['path'],sep='\t')
    if set(table.gold_l4) != {'Large intestine','Small intestine','Lower gastrointestinal tract'}:
        raise RuntimeError('unexpected GOLD labels')
    for k in ['train','val']:
        if not np.array_equal(table.loc[table.split==k,'row_id'].to_numpy(),arrays[k]):
            raise RuntimeError('row-table identity differs from arrays')
    tr = set(table.loc[table.split=='train','Run'].dropna())
    va = set(table.loc[table.split=='val','Run'].dropna())
    if tr & va:
        raise RuntimeError('Run overlaps validation')
    configure_tree()
    model = load_origin(str(RUN/assets['prior']['path']),RUN/assets['origin']['path'])
    origin = common._model_state_sha256(model);common._seed_all(42)
    inv = inject(model,saved['adaptation'])
    initial = dict(origin_model_state_sha256=origin,adapted_model_state_sha256=common._model_state_sha256(model),
                   frozen_parameter_sha256=frozen_sha256(model))
    if initial != saved['initialization'] or inv != saved['trainable_inventory']:
        raise RuntimeError('initialization/scope differs from accepted LoRA+')
    a = copy.deepcopy(saved)
    data = copy.deepcopy(old['arms']['gut_lora']['data'])
    data['paths'] = {k:assets['gut_'+k]['path'] for k in ['all','train','val']}
    data['observed_genus_ids'] = assets['gut_observed']['path'];data['smoke_rows'] = assets['gut_smoke_rows']['path']
    a.update(name=ARM,domain='gut',output='runs/'+ARM,data=data,steps_per_epoch=2304,total_steps=115200,
             extract_on_milestone=False)
    contract = file_record(RUN/'PLAN.md');contract['path']='PLAN.md'
    raw = dict(schema_version=1,method='gut_h100_loraplus_long',prepared_at=now(),machine='CmlLab',node=socket.gethostname(),
               status='SEALED',source=source_record(),assets=assets,arms={ARM:a},contract=contract,
               provenance=dict(method_manifest=str(respiratory),method_manifest_sha256=common.sha256_file(respiratory),
                               gut_manifest=str(old_path),gut_manifest_sha256=common.sha256_file(old_path),
                               domain_policy=old['domain_policy'],source_run_overlap=0),
               execution=dict(machine='H100',host='NVH1003',gpu_uuid=UUID,world_size=1,batch_size=128,
                              precision='BF16 autocast; FP32 parameters',activation_recomputation=False,
                              cpu_threads=2,dataloader_workers=4,optimizer='fused AdamW',validation_seed=42,
                              checkpoints=[10,20,30,50],extract_on_milestone=False,placeholder_673251_untouched=True))
    check_contract(raw)
    p=RUN/'audit/manifest.json';p.write_text(json.dumps(raw,indent=2,ensure_ascii=False)+'\n')
    sha=common.sha256_file(p);(RUN/'audit/manifest.sha256').write_text(sha+'\n')
    common._atomic_json(RUN/'audit/cpu_acceptance.json',dict(status='PASS',time=now(),manifest_sha256=sha,
        initialization_matches_accepted_loraplus=True,train=294906,val=1539,Run_overlap=0,
        strict_gold=True,canonical_partition=True,source_commit=raw['source']['commit']))
    print(json.dumps(dict(status='SEALED',manifest_sha256=sha,source=raw['source']['commit'])),flush=True)


if __name__ == '__main__':
    prepare()
