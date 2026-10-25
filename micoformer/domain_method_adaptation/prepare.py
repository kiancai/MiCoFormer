"""Seal the approved finite method screen independently of running LoRA pairs."""
from __future__ import annotations
import copy
import json
import os
import socket
from pathlib import Path
import numpy as np

from micoformer.domain_adaptation.prepare import load_origin, verify_file, file_record, WEIGHTS
from micoformer.domain_bucket_adaptation.prepare import source_record, now
from micoformer.mpc_unifrac.teacher import ASSETS
from micoformer.mpc_unifrac import workflow as common
from .modules import inject, frozen_sha256

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
EXPERIMENT = PROJECT/'tmp/20261024_respiratory_peft_mechanism_pilot'
FAMILY = os.environ.get('MCF_METHOD_FAMILY', 'adapter')
FAMILIES = ('adapter', 'loraplus', 'dora')
if FAMILY not in FAMILIES:
    raise ValueError(FAMILY)
RUN = EXPERIMENT/'method_screen/families'/FAMILY
REFERENCE = EXPERIMENT/'a100_parallel/pairs/r64_seed42/audit/manifest.json'
REFERENCE_SHA = '60e0e437ed7969acdd45f248d8a365a40865e63a456f33f317347f7fa5f61e21'
SPECS = {
    'adapter': dict(method='adapter', bottleneck=256, dropout=.05,
                    position='after_existing_attention_dropout_and_ffn_dropout_before_residual_add', activation='gelu'),
    'loraplus': dict(method='loraplus', rank=64, alpha=128., dropout=.05),
    'dora': dict(method='dora', rank=64, alpha=128., dropout=.05,
                 norm_axis=1, norm_precision='FP32', norm_gradient='detached',
                 dropout_correction='peft_v0.10.0', bias='unscaled'),
}
CONFIGS = {
    'adapter': [('adapter_b256_lr1e4', 1e-4, 1.), ('adapter_b256_lr3e4', 3e-4, 1.)],
    'loraplus': [('loraplus_r64_ratio4', 5e-5, 4.), ('loraplus_r64_ratio16', 5e-5, 16.)],
    'dora': [('dora_r64_lr5e5', 5e-5, 1.), ('dora_r64_lr1e4', 1e-4, 1.)],
}


def prepare():
    path = RUN/'audit/manifest.json'
    if path.exists():
        raise FileExistsError(path)
    if common.sha256_file(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError('accepted r64 recipe changed')
    reference = json.loads(REFERENCE.read_text())
    for record in reference['assets'].values():
        verify_file(record)
    assets = copy.deepcopy(reference['assets'])
    assets['reference_bucket_manifest'] = file_record(REFERENCE)
    original = reference['arms']['window4096']
    common._seed_all(42)
    model = load_origin(assets['prior']['path'], Path(assets['origin']['path']))
    origin_hash = common._model_state_sha256(model)
    if origin_hash != original['initialization']['origin_model_state_sha256']:
        raise RuntimeError('pretrained origin mismatch')
    common._seed_all(42)
    spec = SPECS[FAMILY]
    inventory = inject(model, spec)
    expected = {'adapter': 6309888, 'loraplus': 6291456, 'dora': 6346752}[FAMILY]
    if inventory['counts']['adaptation'] != expected:
        raise RuntimeError('unexpected adaptation parameter budget')
    initialization = dict(origin_model_state_sha256=origin_hash,
        adapted_model_state_sha256=common._model_state_sha256(model), frozen_parameter_sha256=frozen_sha256(model))
    arms = {}
    for name, lr, ratio in CONFIGS[FAMILY]:
        arm = copy.deepcopy(original)
        arm.pop('lora')
        arm.update(name=name, sampler='window4096', output=str(RUN/'runs'/name), workers=2,
                   learning_rate=lr, lr_ratio=ratio, adaptation=spec,
                   initialization=initialization, trainable_inventory=inventory)
        arms[name] = arm
    contract = EXPERIMENT/'design/METHOD_SCREEN_OVERNIGHT.md'
    manifest = dict(schema_version=1, method='respiratory_method_screen', family=FAMILY,
        prepared_at=now(), machine='CmlLab', node=socket.gethostname(), status='SEALED',
        source=source_record(), assets=assets, arms=arms, contract=file_record(contract),
        execution=dict(activation_recomputation=False, precision='BF16 autocast; FP32 parameters',
            optimizer='fused AdamW; per-group shared schedule multiplier', validation_seed=42,
            allocation=673251, cpu_threads=1, dataloader_workers=2, world_size=1))
    common._atomic_json(path, manifest)
    digest = common.sha256_file(path)
    (path.parent/'manifest.sha256').write_text(digest+'\n')
    print(json.dumps(dict(family=FAMILY, manifest=str(path), sha256=digest, parameters=expected)), flush=True)
    return manifest


def verify(path, expected_sha):
    path = Path(path)
    if common.sha256_file(path) != expected_sha:
        raise RuntimeError('method manifest changed')
    manifest = json.loads(path.read_text())
    if manifest['method'] != 'respiratory_method_screen' or manifest['family'] != FAMILY:
        raise RuntimeError('method family mismatch')
    current = source_record()
    if manifest['source']['commit'] != current['commit'] or manifest['source']['files'] != current['files']:
        raise RuntimeError('sealed source changed')
    for record in list(manifest['assets'].values()) + [manifest['contract']]:
        verify_file(record)
    return manifest
