from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from micoformer.data.datasets import RANK_COLUMNS
from micoformer.models.taxonomy_bias import (
    BiasedTransformerEncoder,
    BiasedTransformerEncoderLayer,
    TaxonomyBiasParams,
    compute_taxonomy_bucket_matrix,
)


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
        use_taxonomy_bias: bool = False,  # R2：启用 taxonomy 距离注意力偏置
    ) -> None:
        super().__init__()
        self.pad_taxon_id = pad_taxon_id
        self.nhead = nhead
        self.use_taxonomy_bias = use_taxonomy_bias

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

        # R1 taxon_path 模式：5 个 rank 各自独立的 embedding 表，相加得到 taxon embedding
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

        # R2=on：使用自定义 biased 层（来自 taxonomy_bias.py）
        # R2=off：使用标准 PyTorch 层
        if use_taxonomy_bias:
            biased_layer = BiasedTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.encoder = BiasedTransformerEncoder(biased_layer, num_layers=num_layers)
            # 可学习的 taxonomy 偏置参数表 [nhead, 5]，初始化为全零
            self.taxonomy_bias_params = TaxonomyBiasParams(nhead=nhead)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
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
                token_x = token_x + self.rank_embeds[rank_name](taxon_path_ids[:, :, rank_idx])
        else:
            if self.taxon_embed is None:
                raise RuntimeError("taxon_embed is not initialized.")
            token_x = self.taxon_embed(token_ids)
        return token_x

    def forward(
        self,
        token_ids: torch.Tensor,        # [Batch, Length]
        abund_bins: torch.Tensor,       # [Batch, Length]
        taxon_path_ids: Optional[torch.Tensor] = None,   # [Batch, Length, 5]
        attention_mask: Optional[torch.Tensor] = None,  # [Batch, Length], True=Valid, False=Pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = token_ids.size(0)

        # token embedding（R1 或 baseline）+ abundance embedding
        x = self._build_token_embedding(token_ids, taxon_path_ids) + self.abund_embed(abund_bins)

        # 构造 key_padding_mask：PyTorch 约定 True=忽略，与我们的 attention_mask 语义相反
        # 扩展一位给 [SAMPLE]（始终有效）
        key_padding_mask = None
        if attention_mask is not None:
            sample_mask = torch.ones((B, 1), dtype=torch.bool, device=token_ids.device)
            key_padding_mask = ~torch.cat([sample_mask, attention_mask], dim=1)

        # 拼接 [SAMPLE] token（不加丰度 embedding，保持语义纯粹性）
        sample_vec = self.sample_embed.weight.view(1, 1, -1).expand(B, -1, -1)
        x = torch.cat([sample_vec, x], dim=1)  # [B, L+1, d_model]

        # R2：计算 taxonomy attention bias，零初始化的 bias_table 保证训练初期无影响
        # taxon-taxon 部分 [B, nhead, L, L] → 扩展为 [B, nhead, L+1, L+1]
        # [SAMPLE] 行/列（index 0）保持 0（中性偏置，不偏向任何进化谱系）
        attn_bias = None
        if self.use_taxonomy_bias and taxon_path_ids is not None:
            bucket_matrix = compute_taxonomy_bucket_matrix(taxon_path_ids)  # [B, L, L]
            taxon_bias = self.taxonomy_bias_params(bucket_matrix)            # [B, nhead, L, L]
            L_seq = taxon_bias.shape[2]
            full_bias = torch.zeros(B, self.nhead, L_seq + 1, L_seq + 1,
                                    dtype=taxon_bias.dtype, device=taxon_bias.device)
            full_bias[:, :, 1:, 1:] = taxon_bias
            attn_bias = full_bias

        # Transformer 前向（两种路径接口不同）
        if self.use_taxonomy_bias:
            h = self.encoder(x, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        else:
            h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        h = self.layer_norm(h)
        sample_repr = h[:, 0, :]
        return h, sample_repr
