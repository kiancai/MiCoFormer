"""R2：Taxonomy 距离注意力偏置（Graphormer-style）

本模块包含 R2 功能的完整实现：
  1. LCA bucket 计算：将两两 taxon 的系统发育距离离散为 5 个桶
  2. TaxonomyBiasParams：可学习的偏置参数表 [nhead, 5]，零初始化
  3. 自定义 Attention 层：支持 per-head additive attention bias 注入

encoder.py 中的 MiCoFormerEncoder 在 use_taxonomy_bias=True 时使用本模块的组件。
"""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# LCA Bucket 定义与计算
# ---------------------------------------------------------------------------

# LCA bucket 含义（5 个层级，从最近到最远）
# bucket 0: same Family（最近）
# bucket 1: same Order
# bucket 2: same Class
# bucket 3: same Phylum（最远的有效匹配）
# bucket 4: far（无共同 Phylum，或 PAD/UNK 无法比较）
NUM_TAXONOMY_BUCKETS = 5

# taxon_path_ids 各列的 rank 索引（与 RANK_COLUMNS 顺序一致）
_PHYLUM_IDX = 0
_CLASS_IDX = 1
_ORDER_IDX = 2
_FAMILY_IDX = 3


def compute_taxonomy_bucket_matrix(path_ids: Tensor) -> Tensor:
    """根据 taxon path IDs 计算两两之间的 LCA bucket 矩阵。

    Args:
        path_ids: [B, L, 5]，列顺序为 [Phylum, Class, Order, Family, Genus]
                  ID 约定：0=PAD, 1=UNK, 2~=真实值（只有 >= 2 才参与匹配）

    Returns:
        bucket_matrix: [B, L, L] uint8
            0=same_family, 1=same_order, 2=same_class, 3=same_phylum, 4=far
    """
    B, L, _ = path_ids.shape

    # 初始化：全部设为 bucket 4（far）
    buckets = torch.full((B, L, L), 4, dtype=torch.uint8, device=path_ids.device)

    # 从粗到细逐层覆写（越细越优先，后写的覆盖先写的）
    # 匹配条件：两侧 rank ID 相等，且值 >= 2（排除 PAD=0 和 UNK=1）
    for rank_idx, bucket_val in [
        (_PHYLUM_IDX, 3),   # 同 Phylum → bucket 3
        (_CLASS_IDX,  2),   # 同 Class  → bucket 2
        (_ORDER_IDX,  1),   # 同 Order  → bucket 1
        (_FAMILY_IDX, 0),   # 同 Family → bucket 0（最近）
    ]:
        ids = path_ids[:, :, rank_idx]              # [B, L]
        ids_i = ids.unsqueeze(2)                    # [B, L, 1]
        ids_j = ids.unsqueeze(1)                    # [B, 1, L]
        match = (ids_i == ids_j) & (ids_i >= 2)    # [B, L, L] bool
        buckets[match] = bucket_val

    return buckets


def compute_taxonomy_attn_bias(bucket_matrix: Tensor, bias_table: Tensor) -> Tensor:
    """将 bucket 矩阵映射为 attention bias 张量。

    Args:
        bucket_matrix: [B, L, L] uint8，值域 {0..4}
        bias_table:    [nhead, num_buckets] float，可学习偏置表

    Returns:
        attn_bias: [B, nhead, L, L] float
    """
    # bias_table[:, bucket_matrix]: [nhead, B, L, L] → permute → [B, nhead, L, L]
    return bias_table[:, bucket_matrix.long()].permute(1, 0, 2, 3)


class TaxonomyBiasParams(nn.Module):
    """R2 的可学习偏置参数：bias_table [nhead, num_buckets]，初始化为全零。

    全零初始化 = R2 在训练开始时无任何影响，梯度驱动模型自主决定是否利用进化先验。
    若对某个 head 进化先验没有帮助，bias_table 对应行会保持接近 0，等价于自动关闭。
    """

    def __init__(self, nhead: int, num_buckets: int = NUM_TAXONOMY_BUCKETS) -> None:
        super().__init__()
        # 初始化为全零：训练初期不干扰 attention，让数据驱动梯度决定偏置方向和幅度
        self.bias_table = nn.Parameter(torch.zeros(nhead, num_buckets))

    def forward(self, bucket_matrix: Tensor) -> Tensor:
        """
        Args:
            bucket_matrix: [B, L, L] uint8
        Returns:
            attn_bias: [B, nhead, L, L] float
        """
        return compute_taxonomy_attn_bias(bucket_matrix, self.bias_table)


# ---------------------------------------------------------------------------
# 支持 attn_bias 注入的自定义 Transformer 层
# ---------------------------------------------------------------------------

class BiasedMultiheadAttention(nn.Module):
    """支持 per-head additive attention bias 注入的多头自注意力。

    与 nn.MultiheadAttention 不同，此实现将 attn_bias [B, nhead, L, L]
    直接加到每个 head 的注意力 logits 上，再做 softmax。
    使用 F.scaled_dot_product_attention 计算，支持 flash attention 加速。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model
        self.dropout = dropout

        # 合并 Q/K/V 投影为单一矩阵（与 PyTorch MHA 的参数布局一致）
        self.in_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(
        self,
        x: Tensor,                                    # [B, L, d_model]
        key_padding_mask: Optional[Tensor] = None,   # [B, L], True=PAD（需忽略）
        attn_bias: Optional[Tensor] = None,          # [B, nhead, L, L]
    ) -> Tensor:
        B, L, _ = x.shape

        # QKV 投影并拆分
        qkv = self.in_proj(x)                  # [B, L, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)         # 各 [B, L, d_model]

        # 变形为多头格式：[B, nhead, L, d_head]
        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)

        # 构造传给 SDPA 的 additive float mask（PAD 位置 = -inf，有效位置 = 0）
        # F.scaled_dot_product_attention 接受 float attn_mask：直接加到 logits 上，再 softmax
        sdpa_mask: Optional[Tensor] = None
        if key_padding_mask is not None:
            # key_padding_mask: [B, L]，True=需忽略 → 转为 [B, 1, 1, L] float mask
            float_mask = torch.zeros(B, 1, 1, L, dtype=q.dtype, device=q.device)
            float_mask = float_mask.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )  # [B, 1, 1, L]，可广播到 [B, nhead, L, L]
            sdpa_mask = float_mask

        if attn_bias is not None:
            # attn_bias: [B, nhead, L, L]，与 padding mask 叠加
            if sdpa_mask is not None:
                sdpa_mask = sdpa_mask + attn_bias   # broadcast [B, nhead, L, L]
            else:
                sdpa_mask = attn_bias

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # [B, nhead, L, d_head]

        # 合并多头并投影
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)  # [B, L, d_model]
        return self.out_proj(out)


class BiasedTransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer Encoder 层，forward 接受 attn_bias 参数。

    设计与 nn.TransformerEncoderLayer(norm_first=True) 完全对应，
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
        self.attn_dropout = nn.Dropout(dropout)  # 等价于 PyTorch 标准层的 dropout1
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # FFN：Linear → GELU → Dropout → Linear → Dropout（与标准层一致）
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,                                    # [B, L, d_model]
        key_padding_mask: Optional[Tensor] = None,   # [B, L], True=PAD
        attn_bias: Optional[Tensor] = None,          # [B, nhead, L, L]
    ) -> Tensor:
        # Pre-LN：先归一化，再做自注意力，attn_dropout 后加残差（等价于 PyTorch 标准层 dropout1）
        x = x + self.attn_dropout(
            self.self_attn(self.norm1(x), key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        )
        # Pre-LN：先归一化，再过 FFN，最后加残差
        x = x + self.ff(self.norm2(x))
        return x


class BiasedTransformerEncoder(nn.Module):
    """BiasedTransformerEncoderLayer 的多层堆叠，逐层传递 taxonomy attention bias。"""

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
