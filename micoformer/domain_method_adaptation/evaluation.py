"""Verified epoch10 inference for both the preceding LoRA batch and new methods."""
from __future__ import annotations
import gc
import json
import os
import socket
from pathlib import Path
import time
import numpy as np
import torch
from micoformer.mpc_unifrac import workflow as common
from micoformer.domain_adaptation.model_io import load_adapted_checkpoint as load_lora
from .model_io import load_adapted_checkpoint as load_method
from .prepare import EXPERIMENT, now

EVAL = EXPERIMENT/'method_screen/evaluation'


def sha(path):
    return common.sha256_file(path)


def endpoint(recipe):
    manifest_path = Path(recipe['training_manifest'])
    if sha(manifest_path) != recipe['training_manifest_sha256']:
        raise RuntimeError('training manifest changed')
    manifest = json.loads(manifest_path.read_text())
    arm = manifest['arms'][recipe['training_arm']]
    run = Path(arm['output'])
    status = json.loads((run/'results/training_status.json').read_text())
    epochs = [json.loads(line) for line in (run/'results/epochs.jsonl').read_text().splitlines()]
    if (status['status'] != 'COMPLETED' or status['global_step'] != 8670 or status['epochs'] != 10
            or status['manifest_sha256'] != recipe['training_manifest_sha256']
            or [e['epoch'] for e in epochs] != list(range(1, 11))
            or not all(e['exact_train_row_coverage'] and e['frozen_weights_unchanged']
                       and e['validation']['exact_row_coverage'] and e['samples'] == 110868 for e in epochs)):
        raise RuntimeError('incomplete or invalid training endpoint')
    path = run/'run/epoch10.pt'
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['epoch'] != 10 or state['global_step'] != 8670 or state['next_batch'] != 0
            or state['arm'] != recipe['training_arm'] or state['manifest_sha256'] != recipe['training_manifest_sha256']
            or state['frozen_parameter_sha256'] != arm['initialization']['frozen_parameter_sha256']):
        raise RuntimeError('endpoint checkpoint identity mismatch')
    if not all(bool(torch.isfinite(v).all()) for v in state['model_state_dict'].values()):
        raise RuntimeError('nonfinite endpoint')
    record = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    del state
    return path, record


def forward(model, arrays, indices, *, bf16=False):
    length = int((~arrays['padding'][indices]).sum(1).max())
    device = next(model.parameters()).device
    args = [torch.from_numpy(np.array(arrays[k][indices, :length], copy=True)).to(device)
            for k in ['genus_ids', 'rclr', 'padding']]
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=bf16):
        value = model.unmasked_representations(*args)[1]
    return value.float().cpu().numpy()


def extract(key, device):
    start = time.monotonic()
    configuration = json.loads((EVAL/'audit/recipes.json').read_text())
    recipe = configuration['models'][key]
    path, checkpoint_record = endpoint(recipe)
    output = EVAL/'tasks/resp/cache'/f'{key}.npy'
    receipt = EVAL/'tasks/resp/audit/embeddings'/f'{key}.json'
    if output.exists() or receipt.exists():
        if not (output.exists() and receipt.exists()):
            raise RuntimeError('partial embedding requires inspection')
        saved = json.loads(receipt.read_text())
        if saved['checkpoint_sha256'] != checkpoint_record['sha256'] or sha(output) != saved['sha256']:
            raise RuntimeError('existing embedding identity mismatch')
        return
    loader = load_method if recipe['kind'] == 'method' else load_lora
    model = loader(path, manifest_sha256=recipe['training_manifest_sha256'], device=device)
    model.eval()
    cache = EVAL/'tasks/resp/cache'
    for name, expected in configuration['inference_inputs'].items():
        if sha(cache/name) != expected:
            raise RuntimeError('frozen inference input changed: '+name)
    arrays = {k: np.load(cache/f'{k}.npy', mmap_mode='r') for k in ['genus_ids', 'rclr', 'padding', 'corpus_rows']}
    rows = arrays['corpus_rows']
    if len(rows) != 90764 or sha(cache/'corpus_rows.npy') != configuration['rows_sha256']:
        raise RuntimeError('respiratory panel changed')
    lengths = (~arrays['padding']).sum(1)
    ordered = np.argsort(lengths, kind='stable')
    probes = np.unique(np.r_[np.arange(4), ordered[[0, len(rows)//2, -1]]])
    fp = forward(model, arrays, probes)
    bf = forward(model, arrays, probes, bf16=True)
    cosine = float(np.min(np.sum(fp*bf, 1)/(np.linalg.norm(fp, axis=1)*np.linalg.norm(bf, axis=1))))
    if cosine <= .9999 or not np.isfinite(fp).all() or not np.isfinite(bf).all():
        raise RuntimeError('same-device BF16/FP32 probe failed')
    temporary = output.with_suffix('.partial.npy')
    if temporary.exists():
        raise RuntimeError('partial embedding requires inspection')
    target = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(rows), 512))
    for first in range(0, len(rows), 64):
        indices = np.arange(first, min(first+64, len(rows)))
        h = forward(model, arrays, indices, bf16=True)
        if not np.isfinite(h).all() or not (np.linalg.norm(h, axis=1)>0).all():
            raise RuntimeError('invalid embedding rows')
        target[first:first+len(h)] = h
        if first % 6400 == 0:
            target.flush()
            common._atomic_json(EVAL/'audit/extract_progress'/f'{key}.json', dict(time=now(), model=key, rows=first+len(h), total=len(rows)))
    target.flush(); del target
    os.replace(temporary, output)
    common._atomic_json(receipt, dict(status='PASS', time=now(), machine='CmlLab', node=socket.gethostname(),
        arm=key, sha256=sha(output), rows=len(rows), rows_sha256=configuration['rows_sha256'],
        checkpoint=checkpoint_record, checkpoint_sha256=checkpoint_record['sha256'],
        training_manifest_sha256=recipe['training_manifest_sha256'],
        manifest_sha256=sha(EVAL/'tasks/resp/audit/manifest.json'), precision='BF16 autocast; FP32 storage',
        fp32_probe_device='same CUDA device', bf16_fp32_min_cosine=cosine, batch_size=64, seconds=time.monotonic()-start))
    common._atomic_json(EVAL/'audit/extracted'/f'{key}.json', dict(status='PASS', time=now(), arm=key,
        checkpoint_sha256=checkpoint_record['sha256'], manifest_sha256=sha(EVAL/'audit/execution_manifest.json')))
    print(f'{now()} EMBEDDING_PASS {key} {time.monotonic()-start:.1f}s', flush=True)
    del model
    gc.collect(); torch.cuda.empty_cache()
