"""Fixed-rank EVA-style initialization from training-only valid activations.

Implements centered PCA without rank redistribution or a PEFT dependency.
Statistics are sampled with a fixed finite budget and saved as a sealed asset;
all later model constructions read exactly that asset, including strict resume.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from micoformer.domain_adaptation.lora import LoRALinear
from micoformer.domain_adaptation.prepare import load_origin, verify_file
from micoformer.domain_method_adaptation.modules import inject
from micoformer.mpc_unifrac import workflow as common
from .prepare import RUN, REPO, REFERENCE_SHA, bind_manifest, configure_tree, now


class Moments:
    def __init__(self, width, device):
        self.count = 0
        self.sum = torch.zeros(width, dtype=torch.float64, device=device)
        self.cross = torch.zeros(width, width, dtype=torch.float64, device=device)

    @torch.no_grad()
    def add(self, values):
        values = values.to(dtype=torch.float64)
        if values.ndim != 2 or not bool(torch.isfinite(values).all()):
            raise RuntimeError('invalid PCA activations')
        self.count += len(values)
        self.sum += values.sum(0)
        self.cross.addmm_(values.T, values)

    @torch.no_grad()
    def principal(self, rank):
        if self.count <= rank:
            raise RuntimeError('insufficient valid calibration tokens')
        mean = self.sum / self.count
        covariance = (self.cross - self.count * torch.outer(mean, mean)) / (self.count - 1)
        covariance = (covariance + covariance.T) * .5
        spectrum, directions = torch.linalg.eigh(covariance)
        if not bool(torch.isfinite(spectrum).all()) or float(spectrum[-1]) <= 0:
            raise RuntimeError('degenerate calibration covariance')
        a = directions[:, -rank:].flip(1).T.contiguous()
        # Resolve arbitrary eigenvector signs once, then save the exact result.
        signs = a.gather(1, a.abs().argmax(1)[:, None]).sign()
        a = (a * signs).float()
        orthogonality = float((a @ a.T - torch.eye(rank, device=a.device)).abs().max())
        if orthogonality > 1e-4 or float(spectrum[-rank]) <= 0:
            raise RuntimeError('invalid principal subspace')
        return a.cpu(), dict(tokens=self.count, orthogonality_max_abs=orthogonality,
            explained_variance_ratio=float(spectrum[-rank:].sum()/spectrum.clamp_min(0).sum()),
            eigenvalues=spectrum[-rank:].flip(0).cpu().tolist())


def apply_initialization(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload['kind'] != 'respiratory_eva_fixedrank_v1' or payload['rank'] != 64:
        raise RuntimeError('unexpected EVA asset')
    targets = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if set(targets) != set(payload['a_weights']) or len(targets) != 48:
        raise RuntimeError('EVA targets differ')
    with torch.no_grad():
        for name, module in targets.items():
            a = payload['a_weights'][name]
            if a.shape != module.lora_A.shape or a.dtype != torch.float32 or not bool(torch.isfinite(a).all()):
                raise RuntimeError('invalid EVA A: '+name)
            if bool(module.lora_B.any()):
                raise RuntimeError('EVA initialization requires zero B')
            module.lora_A.copy_(a)


def validate_provenance(path, train, val, reference):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    selected = np.asarray(payload['train_rows'], dtype=np.int64)
    expected = np.random.default_rng(20261025).choice(train, 8192, replace=False)
    if not np.array_equal(selected, expected) or len(np.intersect1d(selected, val)):
        raise RuntimeError('EVA selected rows changed or overlap validation')
    if payload['reference_sha256'] != REFERENCE_SHA or payload['origin_sha256'] != reference['assets']['origin']['sha256']:
        raise RuntimeError('EVA origin identity changed')
    if payload['prior_sha256'] != reference['assets']['prior']['sha256']:
        raise RuntimeError('EVA prior identity changed')
    if payload['train_rows_sha256'] != hashlib.sha256(selected.tobytes()).hexdigest() or payload['updates'] != 0:
        raise RuntimeError('EVA calibration provenance invalid')


def calibrate(reference_root):
    from .workflow import preflight, make_loader, device_batch
    start = perf_counter()
    output = RUN/'assets/eva_initialization.pt'
    if output.exists():
        raise FileExistsError(output)
    deployment = json.loads((RUN/'audit/calibration_source.json').read_text())
    for name, digest in deployment['files'].items():
        if common.sha256_file(REPO/name) != digest:
            raise RuntimeError('calibration source changed: '+name)
    path = Path(reference_root)/'audit/manifest.json'
    if common.sha256_file(path) != REFERENCE_SHA:
        raise RuntimeError('calibration reference changed')
    raw = json.loads(path.read_text())
    manifest = bind_manifest(raw, reference_root)
    for record in manifest['assets'].values():
        verify_file(record)
    device = preflight(RUN/'audit/eva_cuda_preflight.json')
    configure_tree()
    arm = copy.deepcopy(manifest['arms']['loraplus_r64'])
    train = np.load(arm['data']['paths']['train'])
    val = np.load(arm['data']['paths']['val'])
    selected = np.random.default_rng(20261025).choice(train, 8192, replace=False)
    selected_path = RUN/'audit/eva_train_rows.npy'
    np.save(selected_path, selected)
    arm['data']['paths']['train'] = str(selected_path)
    common._seed_all(42)
    model = load_origin(manifest['assets']['prior']['path'], Path(manifest['assets']['origin']['path']))
    common._seed_all(42)
    inject(model, arm['adaptation'])
    model = model.to(device).eval().requires_grad_(False)
    initial = common._model_state_sha256(model)
    targets = {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}
    if len(targets) != 48:
        raise RuntimeError('unexpected EVA targets')
    moments = {n: Moments(m.base.in_features, device) for n, m in targets.items()}
    generator = torch.Generator().manual_seed(20261025)
    state = {}

    def select_valid(module, args, kwargs):
        mask = kwargs['key_padding_mask']
        valid = (~mask).reshape(-1).nonzero().flatten().cpu()
        if not len(valid):
            raise RuntimeError('no valid EVA tokens')
        chosen = torch.randperm(len(valid), generator=generator)[:128]
        state['indices'] = valid[chosen].to(device)
        state['shape'] = mask.shape

    def hook(name):
        def collect(module, args):
            values = args[0]
            if values.shape[:2] != state['shape']:
                raise RuntimeError('EVA padding alignment changed: '+name)
            moments[name].add(values.reshape(-1, values.shape[-1])[state['indices']].float())
        return collect

    handles = [model.backbone.encoder.layers[0].register_forward_pre_hook(select_valid, with_kwargs=True)]
    handles += [m.register_forward_pre_hook(hook(n)) for n, m in targets.items()]
    loader, sampler, _ = make_loader(arm, manifest['assets']['corpus']['path'], 0, workers=4)
    batches = 0
    try:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for batch in loader:
                args = device_batch(batch, device)
                model._mlm_loss(args['genus_ids'], args['rclr'], args['padding_mask'], args['mlm_mask'])
                model._query_output(args['genus_ids'], args['rclr'], args['padding_mask'], args['query_mask'], args['negative_genus_ids'])
                batches += 1
                if batches % 8 == 0:
                    print(f'{now()} EVA calibration batches={batches}/64', flush=True)
    finally:
        for handle in handles:
            handle.remove()
    if batches != 64 or common._model_state_sha256(model) != initial:
        raise RuntimeError('EVA coverage changed or model updated')
    weights, diagnostics = {}, {}
    for name, values in moments.items():
        weights[name], diagnostics[name] = values.principal(64)
    payload = dict(kind='respiratory_eva_fixedrank_v1', time=now(), rank=64, updates=0,
        source_commit=deployment['commit'], reference_sha256=REFERENCE_SHA,
        origin_sha256=manifest['assets']['origin']['sha256'], prior_sha256=manifest['assets']['prior']['sha256'],
        train_rows=selected.tolist(), train_rows_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        batches=batches, centered=True, token_cap_per_branch_batch=128, rank_redistribution=False,
        a_weights=weights, diagnostics=diagnostics, seconds=perf_counter()-start)
    common._atomic_torch(output, payload)
    validate_provenance(output, train, val, raw)
    common._atomic_json(RUN/'audit/eva_initialization.json', {k: v for k, v in payload.items() if k not in ('a_weights', 'train_rows')})
    print(json.dumps(dict(status='PASS', sha256=common.sha256_file(output), seconds=payload['seconds'], targets=len(weights))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-root', required=True)
    calibrate(parser.parse_args().reference_root)
