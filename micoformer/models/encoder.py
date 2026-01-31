from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class MiCoFormerEncoder(nn.Module):

    def __init__(
        self,
        *,
        vocab_size: int,
        num_abundance_bins: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_taxon_id: int = 0,
        pad_bin_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_taxon_id = pad_taxon_id
        
        self.taxon_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_taxon_id)
        self.abund_embed = nn.Embedding(num_abundance_bins, d_model, padding_idx=pad_bin_id)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,   # 输入格式为 [Batch, Length, Dim]
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,  # [Batch, Length]
        abund_bins: torch.Tensor,  # [Batch, Length]
        attention_mask: Optional[torch.Tensor] = None,  # [Batch, Length], True=Valid, False=Pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # 嵌入层叠加后归一化
        x = self.taxon_embed(input_ids) + self.abund_embed(abund_bins)
        
        # 构造 Padding Mask
        # PyTorch 要求 mask 是 True 表忽略 (Padding)，HuggingFace 的习惯则相反
        # 我们的 attention_mask 是 True 表示有效，False 表示 Padding，所以需要取反
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask  # True -> False (Valid), False -> True (Ignore)

        # 通过 Transformer
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # 最终归一化（对于 Pre-Norm Transformer 是必须的）
        h = self.layer_norm(h)

        # 提取 [SAMPLE] Token 的特征
        sample_repr = h[:, 0, :]

        # return 的 h 是 [Batch, Length, Dim]，是一个batch每个样本的每个菌的每个特征向量
        # sample_repr 是 [Batch, Dim]，是每个样本的 [SAMPLE] Token 特征向量
        return h, sample_repr
