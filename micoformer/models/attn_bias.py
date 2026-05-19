"""V4 Attention Bias 模块

提供两种基于全局距离矩阵的 per-head additive attention bias，统一接口：

    forward(var_indices: [B, L] int64, dist_matrix: [V, V]) -> [B, nhead, L, L]

两种实现：
  - TaxoDistBias：离散 7 bucket（基于 varp['taxo_dist']，取值 {0..6}）
        0=same genus(self)
        1=same family
        2=same order
        3=same class
        4=same phylum
        5=same domain
        6=cross domain
    bias_table 形状 [num_buckets=7, nhead]，零初始化。

  - PhyloDistBias：连续 MLP（基于 varp['phylo_dist']，patristic 距离 float32）
    输入 log1p(distance) 压到合理范围，过 2 层 MLP（Linear→GELU→Linear），
    末层权重/偏置零初始化（训练初期对 attention 无干扰，与离散版语义一致）。

两种都通过 var_indices 双张量索引查全局 dist_matrix（[V, V]）。
encoder 内部将输出 pad 一圈零变 [B, nhead, L+1, L+1] 后传给标准 SDPA。

同时提供 BiasedMultiheadAttention / BiasedTransformerEncoderLayer / BiasedTransformerEncoder
作为承载 attn_bias 注入的通用 attention 层（与具体 bias 来源解耦）。
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 距离驱动的 bias 模块
# ---------------------------------------------------------------------------

NUM_TAXO_BUCKETS = 7  # 与 varp['taxo_dist'] 取值 {0..6} 一致


class TaxoDistBias(nn.Module):
    """离散 R2：查 varp['taxo_dist']（int8, 0~6），过 [num_buckets, nhead] embedding 表。

    bias_table 零初始化 → 训练初期 attention 等价于无 bias，由梯度决定是否利用先验。
    """

    def __init__(self, nhead: int, num_buckets: int = NUM_TAXO_BUCKETS) -> None:
        super().__init__()
        self.bias_table = nn.Parameter(torch.zeros(num_buckets, nhead))

    def forward(self, var_indices: Tensor, dist_matrix: Tensor) -> Tensor:
        """
        Args:
            var_indices: [B, L] int64，每个 token 在 var 中的行号（0~V-1）
            dist_matrix: [V, V]，varp['taxo_dist']（int8 或可转 long）
        Returns:
            attn_bias: [B, nhead, L, L] float
        """
        # 双张量索引获取每对 (q, k) 的 hop bucket，结果 [B, L, L]
        bucket = dist_matrix[var_indices.unsqueeze(2), var_indices.unsqueeze(1)].long()
        # bias_table[bucket]: [B, L, L, nhead] → permute → [B, nhead, L, L]
        bias = self.bias_table[bucket]
        return bias.permute(0, 3, 1, 2).contiguous()


class PhyloDistBias(nn.Module):
    """连续 R2:查 varp['phylo_dist'](float32 patristic 距离),过 3 层 MLP → per-head bias。

    V5 设计(design 文档 §4.3):
      - 输入 log1p(distance) 压到 [0, log1p(660)~6.5] 范围
      - 3 层 MLP(Linear → GELU → Linear → GELU → Linear),hidden_dim 默认 64
      - 末层 zero-init,训练初期 attention 等价无 bias
      - 不接 LayerNorm(直接加到 attention score 上,量级 ~1,LN 会破坏自然幅度)
    """

    def __init__(self, nhead: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, nhead),
        )
        # 末层零初始化（训练初期 attention 等价无 bias）
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, var_indices: Tensor, dist_matrix: Tensor) -> Tensor:
        """
        Args:
            var_indices: [B, L] int64
            dist_matrix: [V, V] float32，varp['phylo_dist']
        Returns:
            attn_bias: [B, nhead, L, L] float
        """
        # 查表得到每对 (q, k) 的连续距离 [B, L, L]
        dists = dist_matrix[var_indices.unsqueeze(2), var_indices.unsqueeze(1)]
        # log1p 压缩 + 增加 feature 维 → [B, L, L, 1]
        dists = torch.log1p(dists.float()).unsqueeze(-1)
        # 过 MLP → [B, L, L, nhead] → [B, nhead, L, L]
        bias = self.mlp(dists)
        return bias.permute(0, 3, 1, 2).contiguous()


def make_dist_bias(bias_type: str, nhead: int, phylo_mlp_hidden: int = 64) -> Optional[nn.Module]:
    """工厂函数：根据 bias_type 创建对应的 bias 模块（none 时返回 None）。

    V5 默认 phylo_mlp_hidden=64(对应 3 层 MLP 设计)。
    """
    if bias_type == "none":
        return None
    if bias_type == "taxo":
        return TaxoDistBias(nhead=nhead)
    if bias_type == "phylo":
        return PhyloDistBias(nhead=nhead, hidden_dim=phylo_mlp_hidden)
    raise ValueError(f"Unknown bias_type: {bias_type!r}. Expected 'none' | 'taxo' | 'phylo'.")


# ---------------------------------------------------------------------------
# 通用 BiasedAttention：支持外部 attn_bias 注入的标准 SDPA attention
# ---------------------------------------------------------------------------

class BiasedMultiheadAttention(nn.Module):
    """支持 per-head additive attention bias 注入的多头自注意力。

    用 F.scaled_dot_product_attention 计算。当 attn_bias 不为 None 时，
    与 key_padding_mask 合并成统一的 float attn_mask 传入 SDPA。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model
        self.dropout = dropout

        # 合并 Q/K/V 投影为单一矩阵（与 PyTorch MHA 参数布局一致）
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: Tensor,                                    # [B, L, d_model]
        key_padding_mask: Optional[Tensor] = None,    # [B, L], True=PAD（需忽略）
        attn_bias: Optional[Tensor] = None,           # [B, nhead, L, L] float
    ) -> Tensor:
        B, L, _ = x.shape

        # QKV 投影并拆分 → [B, nhead, L, d_head]
        qkv = self.in_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)

        # 合并 key_padding_mask 与 attn_bias 成统一的 float attn_mask
        combined_bias: Optional[Tensor] = None
        if key_padding_mask is not None:
            float_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            float_mask = float_mask.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
            combined_bias = float_mask
        if attn_bias is not None:
            combined_bias = (combined_bias + attn_bias) if combined_bias is not None else attn_bias

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, nhead, L, d_head]

        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out_proj(out)


class BiasedTransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer Encoder 层，forward 接受 attn_bias 参数。

    与 nn.TransformerEncoderLayer(norm_first=True, activation='gelu') 完全对应，
    唯一区别是 self_attn 替换为 BiasedMultiheadAttention，支持 attn_bias 注入。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = BiasedMultiheadAttention(d_model, nhead, dropout=dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        # Pre-LN
        x = x + self.attn_dropout(
            self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        )
        x = x + self.ff(self.norm2(x))
        return x


class BiasedTransformerEncoder(nn.Module):
    """BiasedTransformerEncoderLayer 的多层堆叠，逐层传递 attn_bias。"""

    def __init__(self, layer: BiasedTransformerEncoderLayer, num_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        return x
