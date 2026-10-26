"""Low-rank updates for the existing Pre-LN Transformer projections.
为现有 Pre-LN Transformer 的投影增加低秩更新。
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from micoformer.config import ModelConfig


class LoRAProjection(nn.Module):
    """Return only the update (alpha/r) B A dropout(x).
    仅计算低秩增量，原投影继续使用已有权重。
    """

    def __init__(self, weight: Tensor, config: ModelConfig) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(weight.new_empty(config.lora_rank, weight.shape[1]))
        self.lora_B = nn.Parameter(weight.new_zeros(weight.shape[0], config.lora_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = config.lora_alpha / config.lora_rank
        self.dropout = nn.Dropout(config.lora_dropout)

    def forward(self, value: Tensor) -> Tensor:
        return F.linear(F.linear(self.dropout(value), self.lora_A), self.lora_B) * self.scaling


def _additive_mask(mask: Tensor, dtype: torch.dtype) -> Tensor:
    # PyTorch encoder masks use True for excluded positions; SDPA accepts additive masks.
    # Encoder 的 True 表示屏蔽；转成 SDPA 可用的加性 mask。
    if mask.dtype == torch.bool:
        return torch.zeros_like(mask, dtype=dtype).masked_fill_(mask, float("-inf"))
    return mask


class LoRAEncoderLayer(nn.Module):
    """Keep the standard layer's weights and add four low-rank branches.
    保留标准编码层的权重，为 QKV、attention 输出、两层 FFN 增加低秩分支。
    """

    def __init__(self, layer: nn.TransformerEncoderLayer, config: ModelConfig) -> None:
        super().__init__()
        # Reuse the original modules and state-dict names, including norms and biases.
        # 复用原模块和参数名称，包含 Norm 与 bias。
        for name, module in layer.named_children():
            self.add_module(name, module)
        self.activation = layer.activation
        self.lora = nn.ModuleDict({
            "qkv": LoRAProjection(self.self_attn.in_proj_weight, config),
            "out": LoRAProjection(self.self_attn.out_proj.weight, config),
            "ff1": LoRAProjection(self.linear1.weight, config),
            "ff2": LoRAProjection(self.linear2.weight, config),
        })

    def _attention(
        self, value: Tensor, mask: Tensor | None,
        padding_mask: Tensor | None, is_causal: bool,
    ) -> Tensor:
        attention = self.self_attn
        batch, length, width = value.shape
        heads = attention.num_heads
        # Packed Q/K/V share A; B has three output blocks.
        # Q/K/V 共用 A，B 按三个输出区块排列。
        projected = F.linear(value, attention.in_proj_weight, attention.in_proj_bias)
        projected = projected + self.lora["qkv"](value)
        q, k, v = (
            part.reshape(batch, length, heads, width // heads).transpose(1, 2)
            for part in projected.chunk(3, dim=-1)
        )
        if mask is not None:
            mask = _additive_mask(mask, value.dtype)
            if mask.ndim == 3:
                mask = mask.reshape(batch, heads, length, length)
        if padding_mask is not None:
            padding = _additive_mask(padding_mask, value.dtype)[:, None, None, :]
            mask = padding if mask is None else mask + padding
        if is_causal and mask is not None:
            causal = torch.ones(length, length, dtype=torch.bool, device=value.device).triu(1)
            mask = mask + _additive_mask(causal, value.dtype)
            is_causal = False
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask,
            dropout_p=attention.dropout if self.training else 0.0,
            is_causal=is_causal,
        ).transpose(1, 2).reshape(batch, length, width)
        return attention.out_proj(output) + self.lora["out"](output)

    def forward(
        self, src: Tensor, src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None, is_causal: bool = False,
    ) -> Tensor:
        # An explicit forward also keeps eval/inference from bypassing LoRA via fused layers.
        # 显式执行也避免 eval/inference 的融合编码层绕过 LoRA 分支。
        value = src + self.dropout1(
            self._attention(self.norm1(src), src_mask, src_key_padding_mask, is_causal)
        )
        normalized = self.norm2(value)
        hidden = self.linear1(normalized) + self.lora["ff1"](normalized)
        hidden = self.dropout(self.activation(hidden))
        return value + self.dropout2(self.linear2(hidden) + self.lora["ff2"](hidden))


def add_lora(model: nn.Module, config: ModelConfig) -> None:
    """Freeze the base model and attach independently initialized adapters per layer.
    冻结基座，各层分别初始化适配参数；预测头仍参与前向和反向传播。
    """
    model.requires_grad_(False)
    model.config = config
    model.backbone.config = model.config
    model.backbone.encoder.layers = nn.ModuleList(
        LoRAEncoderLayer(layer, config) for layer in model.backbone.encoder.layers
    )
