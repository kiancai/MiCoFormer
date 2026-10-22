"""Standard LoRA: W x + (alpha / rank) B A dropout(x).

No quantization, new objective, or change to the original attention forward.
The packed QKV projection is one target matrix, with a shared rank-r A.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05

    def __post_init__(self):
        if self.rank <= 0 or self.alpha <= 0 or not 0 <= self.dropout < 1:
            raise ValueError("invalid LoRA configuration")

    def to_dict(self):
        return asdict(self)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, config: LoRAConfig):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(config.rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, config.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        update = nn.functional.linear(nn.functional.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scaling


HEAD_PREFIXES = ("backbone.mlm_head.", "query_head.")
TARGET_SUFFIXES = ("self_attn.in_proj", "self_attn.out_proj", "ff.0", "ff.3")


def inject_lora(model: nn.Module, config: LoRAConfig) -> dict:
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise RuntimeError("LoRA already injected")
    targets = [f"backbone.encoder.layers.{i}.{suffix}"
               for i in range(model.config.num_layers) for suffix in TARGET_SUFFIXES]
    # Validate every target before any mutation.
    if any(not isinstance(model.get_submodule(name), nn.Linear) for name in targets):
        raise TypeError("expected standard Linear projections in all encoder layers")
    model.requires_grad_(False)
    for name in targets:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, LoRALinear(model.get_submodule(name), config))
    for name, parameter in model.named_parameters():
        if name.startswith(HEAD_PREFIXES):
            parameter.requires_grad_(True)
    return trainable_inventory(model, config)


def trainable_inventory(model, config: LoRAConfig) -> dict:
    groups = {"lora": {}, "heads": {}, "frozen": {}}
    for name, parameter in model.named_parameters():
        is_lora = name.startswith("backbone.encoder.layers.") and name.endswith((".lora_A", ".lora_B"))
        is_head = name.startswith(HEAD_PREFIXES)
        if parameter.requires_grad != (is_lora or is_head):
            raise RuntimeError(f"incorrect trainable scope: {name}")
        group = "lora" if is_lora else "heads" if is_head else "frozen"
        groups[group][name] = parameter.numel()
    return dict(config=config.to_dict(), parameters=groups,
                counts={key: sum(values.values()) for key, values in groups.items()})


def frozen_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            value = parameter.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parameter_groups(model, *, lora_lr: float, head_lr: float):
    lora, heads = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(HEAD_PREFIXES):
            heads.append(parameter)
        elif name.endswith((".lora_A", ".lora_B")):
            lora.append(parameter)
        else:
            raise RuntimeError(f"unexpected optimizer parameter: {name}")
    if not lora or not heads:
        raise RuntimeError("both adapter and existing self-supervised heads must train")
    return [dict(params=lora, lr=lora_lr, group_name="lora"),
            dict(params=heads, lr=head_lr, group_name="heads")]
