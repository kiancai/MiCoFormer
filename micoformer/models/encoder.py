"""Transformer backbone and mean pooling over valid tokens.
Transformer 编码主干与有效 token 的均值汇总。
"""
from __future__ import annotations

from torch import Tensor, nn

from micoformer.config import ModelConfig


def masked_token_mean(h: Tensor, padding_mask: Tensor) -> Tensor:
    """Mean over valid tokens: [B, L, D] -> [B, D]; no L2 normalization.
    对有效 token 取均值：[B, L, D] -> [B, D]；不做 L2 归一化。
    """
    # Exclude PAD and Query-removed positions; weights have shape [B, L, 1].
    # 排除 PAD 和 Query 移除位置，生成 [B, L, 1] 的有效位置权重。
    valid = (~padding_mask).to(dtype=h.dtype).unsqueeze(-1)
    denominator = valid.sum(dim=1)
    return (h * valid).sum(dim=1) / denominator


class EncoderBackbone(nn.Module):
    """Encode input embeddings with a Transformer and final token normalization.
    通过 Transformer 和最终 token 归一化处理输入 embedding。
    """

    def __init__(self, config: ModelConfig, input_stem: nn.Module) -> None:
        super().__init__()

        self.config = config
        self.input_stem = input_stem

        # Pre-LN Transformer layer with GELU and [B, L, D] input layout.
        # 使用 GELU、先归一化结构和 [B, L, D] 输入布局的标准编码层。
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.encoder_heads,
            dim_feedforward=config.encoder_ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Stack independent layers while retaining dense padded batches.
        # 堆叠参数独立的编码层，保持补齐后的稠密 batch 布局。
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )

        # Normalize each output token independently; shape remains [B, L, D].
        # 分别归一化每个输出 token，形状保持 [B, L, D]。
        self.final_token_norm = nn.LayerNorm(config.d_model)

    def _encode(
        self,
        genus_ids: Tensor,
        abundance: Tensor,
        padding_mask: Tensor,
        abundance_mask: Tensor | None,
        *,
        prior_table: Tensor,
    ) -> Tensor:
        """Encode genus IDs and abundance [B, L] into contextual vectors [B, L, D].
        将菌属编号与丰度 [B, L] 编码为上下文向量 [B, L, D]。
        """
        # abundance_mask hides abundance only; None preserves all supplied abundance.
        # abundance_mask 仅遮盖丰度、保留身份；None 表示不额外遮盖丰度。
        # Only fixed input identities use the prior table supplied by MiCoFormer.
        # 仅 fixed 输入身份使用 MiCoFormer 传入的先验表，不在此保存副本。
        if self.config.identity_type == "fixed":
            h = self.input_stem(genus_ids, abundance, padding_mask, abundance_mask, prior_table=prior_table)
        else:
            h = self.input_stem(genus_ids, abundance, padding_mask, abundance_mask)
        # True entries exclude PAD or Query-removed positions from attention keys/values.
        # True 标记 PAD 或 Query 移除位置，使其不作为注意力的键和值。
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        h = self.final_token_norm(h)
        # Attention masking does not zero output positions; clear all D features explicitly.
        # 注意力屏蔽不保证对应输出为零，因此明确将忽略位置的全部 D 个特征归零。
        return h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
