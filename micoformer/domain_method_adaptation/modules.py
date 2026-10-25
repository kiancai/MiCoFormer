"""Frozen-backbone bottleneck Adapter, LoRA+, and DoRA for unlabeled DAPT.

DoRA follows the detached weight-norm update and dropout correction in
https://github.com/huggingface/peft/blob/v0.10.0/src/peft/tuners/lora/layer.py
The normalization is computed in FP32; base biases are never rescaled.
"""
from __future__ import annotations

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from micoformer.domain_adaptation.lora import LoRALinear, LoRAConfig, TARGET_SUFFIXES, HEAD_PREFIXES, frozen_sha256


class LowRankBranch(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.lora_A = nn.Parameter(a.detach().clone())
        self.lora_B = nn.Parameter(b.detach().clone())

    def forward(self, value):
        return F.linear(F.linear(value, self.lora_A), self.lora_B)


class IndependentQKVLoRA(nn.Module):
    """Split only the adapter; retain packed base/bias and a shared dropout mask.

    Cloning the already initialized shared A preserves the other projections'
    initialization and the entire RNG state. Q/K/V then learn independently.
    """
    def __init__(self, packed):
        super().__init__()
        if packed.base.out_features != 3 * packed.base.in_features:
            raise ValueError('expected packed equal-width QKV')
        self.base = packed.base
        self.dropout = packed.dropout
        self.scaling = packed.scaling
        self.branches = nn.ModuleDict({key: LowRankBranch(packed.lora_A, b)
            for key, b in zip(('q', 'k', 'v'), packed.lora_B.chunk(3, dim=0), strict=True)})

    def forward(self, value):
        dropped = self.dropout(value)
        delta = torch.cat([branch(dropped) for branch in self.branches.values()], dim=-1)
        return self.base(value) + self.scaling * delta


def internal_norm(name):
    return name.startswith('backbone.encoder.layers.') and name.rsplit('.', 2)[-2] in ('norm1', 'norm2')


class BottleneckAdapter(nn.Module):
    def __init__(self, width=512, bottleneck=256, dropout=.05):
        super().__init__()
        self.down = nn.Linear(width, bottleneck)
        self.up = nn.Linear(bottleneck, width)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value):
        return value + self.up(self.dropout(F.gelu(self.down(value))))


class DoRALinear(LoRALinear):
    def __init__(self, base, config):
        super().__init__(base, config)
        self.magnitude = nn.Parameter(torch.linalg.vector_norm(base.weight.detach().float(), dim=1))

    def forward(self, value):
        base = self.base(value)
        dropped = self.dropout(value)
        delta = F.linear(F.linear(dropped, self.lora_A), self.lora_B) * self.scaling
        # DoRA section 4.3: denominator tracks the current direction but is detached.
        with torch.no_grad(), torch.autocast(value.device.type, enabled=False):
            direction = self.base.weight.float() + self.scaling * (self.lora_B.float() @ self.lora_A.float())
            norm = torch.linalg.vector_norm(direction, dim=1)
        scale = self.magnitude / norm.clamp_min(torch.finfo(norm.dtype).tiny)
        correction = F.linear(dropped, self.base.weight, bias=None)
        # Keep scale-1 in FP32 so small magnitude changes survive BF16 rounding.
        return (base + (scale - 1) * correction + scale * delta).to(base.dtype)


def inventory(model, specification):
    counts = dict(adaptation=0, heads=0, frozen=0)
    parameters = {key: {} for key in counts}
    for name, parameter in model.named_parameters():
        in_encoder = name.startswith('backbone.encoder.layers.')
        selected = in_encoder and (
            name.endswith(('.lora_A', '.lora_B', '.magnitude'))
            or '.adapter.down.' in name or '.adapter.up.' in name)
        selected = selected or (specification.get('train_norm', False) and internal_norm(name))
        if parameter.requires_grad != selected:
            raise RuntimeError(f'incorrect method trainable scope: {name}')
        group = 'adaptation' if selected else 'heads' if name.startswith(HEAD_PREFIXES) else 'frozen'
        counts[group] += parameter.numel()
        parameters[group][name] = parameter.numel()
    return dict(specification=specification, counts=counts, parameters=parameters)


def inject(model, specification):
    method = specification['method']
    if method not in ('adapter', 'loraplus', 'dora', 'lora'):
        raise ValueError(method)
    if any(isinstance(m, (LoRALinear, BottleneckAdapter)) for m in model.modules()):
        raise RuntimeError('adaptation already injected')
    model.requires_grad_(False)
    if method == 'adapter':
        for layer in model.backbone.encoder.layers:
            # Both adapters follow the complete existing branch, including dropout.
            layer.attn_dropout = nn.Sequential(OrderedDict([
                ('dropout', layer.attn_dropout),
                ('adapter', BottleneckAdapter(512, specification['bottleneck'], specification['dropout']))]))
            layer.ff.add_module('adapter', BottleneckAdapter(512, specification['bottleneck'], specification['dropout']))
    else:
        config = LoRAConfig(specification['rank'], specification['alpha'], specification['dropout'])
        wrapper = DoRALinear if method == 'dora' else LoRALinear
        targets = [f'backbone.encoder.layers.{i}.{suffix}'
                   for i in range(model.config.num_layers) for suffix in TARGET_SUFFIXES]
        if any(not isinstance(model.get_submodule(name), nn.Linear) for name in targets):
            raise TypeError('expected ordinary Linear projections')
        for name in targets:
            parent, child = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), child, wrapper(model.get_submodule(name), config))
        if specification.get('qkv_independent', False):
            for layer in model.backbone.encoder.layers:
                layer.self_attn.in_proj = IndependentQKVLoRA(layer.self_attn.in_proj)
        if specification.get('train_norm', False):
            for layer in model.backbone.encoder.layers:
                layer.norm1.requires_grad_(True)
                layer.norm2.requires_grad_(True)
    return inventory(model, specification)


def parameter_groups(model, arm):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if arm['adaptation'].get('train_norm', False) and internal_norm(name):
            key, lr = 'norm', arm['norm_learning_rate']
        elif arm['adaptation']['method'] == 'loraplus':
            key = 'B' if name.endswith('lora_B') else 'A'
            lr = arm['learning_rate'] * (arm['lr_ratio'] if key == 'B' else 1.)
        else:
            key, lr = 'adaptation', arm['learning_rate']
        groups.setdefault(key, dict(params=[], lr=lr, group_name=key))['params'].append(parameter)
        if key == 'norm':
            groups[key]['weight_decay'] = 0.
    if not groups:
        raise RuntimeError('no trainable adaptation parameters')
    return list(groups.values())
