from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from micoformer.data.datasets import RANK_COLUMNS


class MiCoFormerEncoder(nn.Module):

    def __init__(
        self,
        *,
        genus_vocab_size: Optional[int] = None,   # taxon 模式必须提供；taxon_path 模式不需要
        total_abundance_bins: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_taxon_id: int = 0,
        pad_bin_id: int = 0,
        token_embedding_mode: str = "taxon_path",
        rank_vocab_sizes: Optional[Dict[str, int]] = None,  # taxon_path 模式必须提供
    ) -> None:
        super().__init__()
        self.pad_taxon_id = pad_taxon_id
        if token_embedding_mode not in {"taxon", "taxon_path"}:
            raise ValueError(
                f"Unknown token_embedding_mode: {token_embedding_mode}. "
                "Expected 'taxon' or 'taxon_path'."
            )
        self.token_embedding_mode = token_embedding_mode

        # [SAMPLE] 使用独立可学习向量
        self.sample_embed = nn.Embedding(1, d_model)

        # taxon 模式：每个 genus 一个独立 embedding；taxon_path 模式：不需要此表
        self.taxon_embed: Optional[nn.Embedding] = None
        if self.token_embedding_mode == "taxon":
            if genus_vocab_size is None:
                raise ValueError("genus_vocab_size is required when token_embedding_mode='taxon'")
            self.taxon_embed = nn.Embedding(genus_vocab_size, d_model, padding_idx=pad_taxon_id)

        self.abund_embed = nn.Embedding(total_abundance_bins, d_model, padding_idx=pad_bin_id)
        self.rank_embeds = nn.ModuleDict()

        # taxon_path 模式：5 个 rank 各自独立的 embedding 表，相加得到 taxon embedding
        if self.token_embedding_mode == "taxon_path":
            if rank_vocab_sizes is None:
                raise ValueError("rank_vocab_sizes is required when token_embedding_mode='taxon_path'")
            for rank_name in RANK_COLUMNS:
                if rank_name not in rank_vocab_sizes:
                    raise ValueError(
                        f"Missing rank vocab size for '{rank_name}'. "
                        f"Expected ranks: {RANK_COLUMNS}, got: {list(rank_vocab_sizes.keys())}."
                    )
                self.rank_embeds[rank_name] = nn.Embedding(
                    int(rank_vocab_sizes[rank_name]), d_model, padding_idx=0
                )

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

    def _build_token_embedding(
        self,
        token_ids: torch.Tensor,
        taxon_path_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # 统一生成 token embedding：
        # - Baseline: 使用 taxon_id embedding
        # - R1: 使用 taxon-path 各层级 embedding 相加
        if self.token_embedding_mode == "taxon_path":
            if taxon_path_ids is None:
                raise ValueError("taxon_path_ids is required when token_embedding_mode='taxon_path'")

            token_x = self.rank_embeds[RANK_COLUMNS[0]](taxon_path_ids[:, :, 0])
            for rank_idx, rank_name in enumerate(RANK_COLUMNS[1:], start=1):
                rank_ids = taxon_path_ids[:, :, rank_idx]
                token_x = token_x + self.rank_embeds[rank_name](rank_ids)
        else:
            if self.taxon_embed is None:
                raise RuntimeError("taxon_embed is not initialized.")
            token_x = self.taxon_embed(token_ids)

        return token_x

    def forward(
        self,
        token_ids: torch.Tensor,  # [Batch, Length]
        abund_bins: torch.Tensor,  # [Batch, Length]
        taxon_path_ids: Optional[torch.Tensor] = None,  # [Batch, Length, 5]
        attention_mask: Optional[torch.Tensor] = None,  # [Batch, Length], True=Valid, False=Pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        token_x = self._build_token_embedding(
            token_ids=token_ids,
            taxon_path_ids=taxon_path_ids,
        )

        # 与 abundance 嵌入逐元素相加
        x = token_x + self.abund_embed(abund_bins)
        
        # 构造 Padding Mask
        # PyTorch 要求 mask 是 True 表忽略 (Padding)，HuggingFace 的习惯则相反
        # 我们的 attention_mask 是 True 表示有效，False 表示 Padding，所以需要取反
        key_padding_mask = None
        if attention_mask is not None:
            # attention_mask 是 [Batch, Length]
            # 我们要拼一个 [SAMPLE] 在最前面，它是始终 Valid 的 (True)
            # 所以新的 mask 应该是 [Batch, Length + 1]
            B = token_ids.size(0)
            sample_mask = torch.ones((B, 1), dtype=torch.bool, device=token_ids.device)
            # 拼接: [SAMPLE_MASK, TAXON_MASK]
            extended_mask = torch.cat([sample_mask, attention_mask], dim=1)
            
            # 转为 key_padding_mask: True -> False (Valid), False -> True (Ignore)
            key_padding_mask = ~extended_mask

        # 拼接 [SAMPLE] Token Embedding
        # sample_vec: [1, d_model] -> [Batch, 1, d_model]
        B = token_ids.size(0)
        sample_vec = self.sample_embed.weight.view(1, 1, -1).expand(B, -1, -1)
        
        # 最终输入序列: [SAMPLE, Taxon_1, Taxon_2, ...]
        # 注意：SAMPLE Token 不加丰度 Embedding！
        x = torch.cat([sample_vec, x], dim=1)

        # 通过 Transformer
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        
        # 对 Encoder 输出做一次最终归一化
        h = self.layer_norm(h)

        # 提取 [SAMPLE] Token 的特征 (第 0 位)
        sample_repr = h[:, 0, :]
        
        # 返回的 h 应该只包含物种的部分 (去除第 0 位的 SAMPLE)，以便后续任务使用
        # 或者保留也可以，取决于下游任务。通常保留比较通用。
        # 这里我们返回完整序列，让下游自己切
        return h, sample_repr
