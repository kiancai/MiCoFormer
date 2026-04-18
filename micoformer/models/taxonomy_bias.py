"""R2：Taxonomy 距离注意力偏置（Graphormer-style）

本模块包含 R2 功能的完整实现：
  1. LCA bucket 计算：将两两 taxon 的系统发育距离离散为 5 个桶
  2. TaxonomyBiasParams：可学习的偏置参数表 [nhead, 5]，零初始化
  3. 自定义 Attention 层：支持 per-head additive attention bias 注入

encoder.py 中的 MiCoFormerEncoder 在 use_taxonomy_bias=True 时使用本模块的组件。

性能说明：
  BiasedMultiheadAttention 优先使用 torch.nn.attention.flex_attention（PyTorch 2.5+ 内置）。
  该路径通过 score_mod 将 bias 查表融入 Triton kernel，不预先物化 [B, nhead, L, L] 的
  float bias 矩阵，backward 也无需额外 scatter_add，接近 Flash Attention 的速度。
  若 FlexAttention 不可用（PyTorch < 2.5），回退到 F.scaled_dot_product_attention
  （Flash 会被禁用，训练速度明显下降）。
"""
from __future__ import annotations

import copy
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# FlexAttention（PyTorch 2.5+ 内置）优先启用；若当前节点/当前 batch 的编译失败，
# 会在运行时自动回退到 F.scaled_dot_product_attention。
#
# 如需显式关闭，可设置：
#   export MICOFORMER_ENABLE_FLEX_ATTENTION=0
_FLEX_ATTENTION_AVAILABLE = False
_FLEX_ATTENTION_RUNTIME_DISABLED = False
_flex_attention = None
if os.environ.get("MICOFORMER_ENABLE_FLEX_ATTENTION", "1").strip().lower() not in {"0", "false", "no", "off"}:
    try:
        from torch.nn.attention.flex_attention import flex_attention as _flex_attention_raw
        # dynamic=True：序列长度在 batch 间动态变化，不重新编译 kernel
        _flex_attention = torch.compile(_flex_attention_raw, dynamic=True, fullgraph=False)
        _FLEX_ATTENTION_AVAILABLE = True
    except Exception:
        _FLEX_ATTENTION_AVAILABLE = False


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
        attn_bias: Optional[Tensor] = None,          # [B, nhead, L, L]，回退路径用
        score_mod=None,                              # FlexAttention score_mod 闭包
    ) -> Tensor:
        global _FLEX_ATTENTION_RUNTIME_DISABLED
        B, L, _ = x.shape

        # QKV 投影并拆分，变形为 [B, nhead, L, d_head]
        qkv = self.in_proj(x)                  # [B, L, 3*d_model]
        q, k, v = qkv.chunk(3, dim=-1)         # 各 [B, L, d_model]
        q = q.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        k = k.view(B, L, self.nhead, self.d_head).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.d_head).transpose(1, 2)

        use_flex_attention = (
            _FLEX_ATTENTION_AVAILABLE
            and not _FLEX_ATTENTION_RUNTIME_DISABLED
            and score_mod is not None
            and _flex_attention is not None
        )
        if use_flex_attention:
            # FlexAttention 路径：bias 查表融入 Triton kernel，不物化 [B, nhead, L, L] float 矩阵
            #
            # PAD masking：在 score_mod 外层包一个加法，对 PAD key 位置加极大负数
            # 使用有限大负数（而非 -inf）避免 softmax 产生 NaN
            actual_score_mod = score_mod
            if key_padding_mask is not None:
                _large_neg = torch.finfo(q.dtype).min / 2
                pad_addend = torch.zeros(B, L, dtype=q.dtype, device=q.device)
                pad_addend.masked_fill_(key_padding_mask, _large_neg)
                _inner = score_mod

                def actual_score_mod(score, b, h, q_idx, kv_idx):
                    return _inner(score, b, h, q_idx, kv_idx) + pad_addend[b, kv_idx]

            # 注意：FlexAttention 不支持 attention weight dropout；
            # BiasedTransformerEncoderLayer 的 attn_dropout 会对输出做 dropout，效果等价
            try:
                out = _flex_attention(q, k, v, score_mod=actual_score_mod)
            except Exception:
                _FLEX_ATTENTION_RUNTIME_DISABLED = True
                use_flex_attention = False

        if not use_flex_attention:
            # 回退路径：F.scaled_dot_product_attention（传入密集 attn_mask 时 Flash 会被禁用）
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
        attn_bias: Optional[Tensor] = None,          # [B, nhead, L, L]，回退路径用
        score_mod=None,                              # FlexAttention score_mod 闭包
    ) -> Tensor:
        # Pre-LN：先归一化，再做自注意力，attn_dropout 后加残差
        x = x + self.attn_dropout(
            self.self_attn(
                self.norm1(x),
                key_padding_mask=key_padding_mask,
                attn_bias=attn_bias,
                score_mod=score_mod,
            )
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
        attn_bias: Optional[Tensor] = None,          # 回退路径用
        score_mod=None,                              # FlexAttention score_mod 闭包
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask,
                      attn_bias=attn_bias, score_mod=score_mod)
        return x


class _TaxonomyScoreModCallable:
    """FlexAttention score_mod 的稳定 callable 包装器。

    torch.compile 对 callable 参数做 identity guard：每次 forward 新建的闭包都会触发
    重新编译 Triton kernel，导致 GPU 在等 CPU 编译时利用率跌到 4%。

    本类在 MiCoFormerEncoder.__init__ 中创建一次，forward 中只调用 update() 更新
    当前 batch 的张量引用，callable 对象本身不变，torch.compile 不会重新编译。
    """

    def __init__(self) -> None:
        self.phylum_ids: Optional[Tensor] = None
        self.class_ids: Optional[Tensor] = None
        self.order_ids: Optional[Tensor] = None
        self.family_ids: Optional[Tensor] = None
        self._bt: Optional[Tensor] = None

    def update(
        self,
        phylum_ids: Tensor,
        class_ids: Tensor,
        order_ids: Tensor,
        family_ids: Tensor,
        bt: Tensor,
    ) -> None:
        """每次 forward 调用，更新当前 batch 的张量引用。"""
        self.phylum_ids = phylum_ids
        self.class_ids = class_ids
        self.order_ids = order_ids
        self.family_ids = family_ids
        self._bt = bt

    def __call__(self, score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        p_q = self.phylum_ids[b, q_idx].long()
        p_k = self.phylum_ids[b, kv_idx].long()
        c_q = self.class_ids[b, q_idx].long()
        c_k = self.class_ids[b, kv_idx].long()
        o_q = self.order_ids[b, q_idx].long()
        o_k = self.order_ids[b, kv_idx].long()
        f_q = self.family_ids[b, q_idx].long()
        f_k = self.family_ids[b, kv_idx].long()
        match_p = ((p_q == p_k) & (p_q >= 2)).long()
        match_c = ((c_q == c_k) & (c_q >= 2)).long()
        match_o = ((o_q == o_k) & (o_q >= 2)).long()
        match_f = ((f_q == f_k) & (f_q >= 2)).long()
        bucket = 4 - match_p - match_c - match_o - match_f
        return score + self._bt[h, bucket]
