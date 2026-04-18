from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from micoformer.data.datasets import RANK_COLUMNS
from micoformer.models.taxonomy_bias import (
    BiasedTransformerEncoder,
    BiasedTransformerEncoderLayer,
    TaxonomyBiasParams,
    _FLEX_ATTENTION_AVAILABLE,
    _TaxonomyScoreModCallable,
    compute_taxonomy_bucket_matrix,
)


class MiCoFormerEncoder(nn.Module):

    def __init__(
        self,
        *,
        genus_vocab_size: int,
        total_abundance_bins: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_taxon_id: int = 0,
        pad_bin_id: int = 0,
        token_embedding_mode: str = "taxon_path",
        rank_vocab_sizes: Dict[str, int],
        use_taxonomy_bias: bool = False,  # R2：启用 taxonomy 距离注意力偏置
        bias_grad_every_k: int = 1,       # R2：每 k 步才对 bias_table 计算梯度（1=每步都算，默认行为）
    ) -> None:
        super().__init__()
        self.pad_taxon_id = pad_taxon_id
        self.nhead = nhead
        self.use_taxonomy_bias = use_taxonomy_bias
        self.bias_grad_every_k = bias_grad_every_k
        # 不是 nn.Parameter，不进入 state_dict，仅用于训练时的步数计数
        self._bias_grad_counter: int = 0

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
            self.taxon_embed = nn.Embedding(genus_vocab_size, d_model, padding_idx=pad_taxon_id)

        self.abund_embed = nn.Embedding(total_abundance_bins, d_model, padding_idx=pad_bin_id)
        self.rank_embeds = nn.ModuleDict()

        # taxon_path 模式：5 个 rank 各自独立的 embedding 表，相加得到 taxon embedding
        if self.token_embedding_mode == "taxon_path":
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
            # FlexAttention score_mod 稳定包装器：__init__ 时创建一次，forward 里只更新引用
            # 避免每次 forward 新建闭包导致 torch.compile identity guard miss（重新编译）
            if _FLEX_ATTENTION_AVAILABLE:
                self._score_mod_obj = _TaxonomyScoreModCallable()
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
        taxon_path_ids: torch.Tensor,
    ) -> torch.Tensor:
        # 统一生成 token embedding：
        # - Baseline: 使用 taxon_id embedding
        # - R1: 使用 taxon-path 各层级 embedding 相加
        if self.token_embedding_mode == "taxon_path":
            token_x = self.rank_embeds[RANK_COLUMNS[0]](taxon_path_ids[:, :, 0])
            for rank_idx, rank_name in enumerate(RANK_COLUMNS[1:], start=1):
                token_x = token_x + self.rank_embeds[rank_name](taxon_path_ids[:, :, rank_idx])
        else:
            token_x = self.taxon_embed(token_ids)
        return token_x

    def forward(
        self,
        token_ids: torch.Tensor,        # [Batch, Length]
        abund_bins: torch.Tensor,       # [Batch, Length]
        taxon_path_ids: torch.Tensor,   # [Batch, Length, 5]
        attention_mask: torch.Tensor,   # [Batch, Length], True=Valid, False=Pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = token_ids.size(0)

        # token embedding（R1 或 baseline）+ abundance embedding
        x = self._build_token_embedding(token_ids, taxon_path_ids) + self.abund_embed(abund_bins)

        # 构造 key_padding_mask：PyTorch 约定 True=忽略，与我们的 attention_mask 语义相反
        # 扩展一位给 [SAMPLE]（始终有效）
        sample_mask = torch.ones((B, 1), dtype=torch.bool, device=token_ids.device)
        key_padding_mask = ~torch.cat([sample_mask, attention_mask], dim=1)

        # 拼接 [SAMPLE] token（不加丰度 embedding，保持语义纯粹性）
        sample_vec = self.sample_embed.weight.view(1, 1, -1).expand(B, -1, -1)
        x = torch.cat([sample_vec, x], dim=1)  # [B, L+1, d_model]

        # R2：构造 taxonomy attention bias
        # FlexAttention 路径（推荐）：bucket_matrix 留在 GPU 上，bias 查表融入 Triton kernel，
        #   不物化 [B, nhead, L+1, L+1] float 矩阵，backward 无需 scatter_add。
        # 回退路径（PyTorch < 2.5）：预先展开为 float bias，传给标准 SDPA（Flash 会被禁用）。
        attn_bias = None
        score_mod = None
        if self.use_taxonomy_bias:
            L_tok = taxon_path_ids.shape[1]

            # ── bias_grad_every_k 判断（两条路径共用）────────────────────────────
            # 训练模式下：每 bias_grad_every_k 步才对 bias_table 做梯度反传；
            # eval 模式下：始终 detach（推理不需要梯度）。
            # k=1（默认）：行为与旧版完全一致，每步都计算梯度。
            if self.training:
                self._bias_grad_counter += 1
                _need_bias_grad = (self.bias_grad_every_k == 1 or
                                   self._bias_grad_counter % self.bias_grad_every_k == 0)
            else:
                _need_bias_grad = False

            if _FLEX_ATTENTION_AVAILABLE:
                # FlexAttention 路径：在 score_mod 内直接从 path_ids 计算 LCA bucket。
                #
                # 内存优化：用 4 个独立的 [B, L+1] int16 张量（共 ~420 KB）替代
                # full_bucket [B, L+1, L+1] uint8（~5 MB），cache 利用率更高。
                #
                # 重要：score_mod 内不能混用 tensor 索引和 Python int 常量（如 ids[b, q, 0]），
                # 因为 FlexAttention 内部用 vmap，混合索引会触发隐式 .item() 报错。
                # 解决方案：把 4 个 rank 拆成独立 2D 张量，统一使用 ids[b, q] 双张量索引。
                zeros_1d = torch.zeros(B, 1, dtype=torch.int16, device=x.device)
                pids = taxon_path_ids[:, :, :4].to(torch.int16)  # [B, L, 4]
                phylum_ids = torch.cat([zeros_1d, pids[:, :, 0]], dim=1)  # [B, L+1]
                class_ids  = torch.cat([zeros_1d, pids[:, :, 1]], dim=1)
                order_ids  = torch.cat([zeros_1d, pids[:, :, 2]], dim=1)
                family_ids = torch.cat([zeros_1d, pids[:, :, 3]], dim=1)

                _bt = (self.taxonomy_bias_params.bias_table
                       if _need_bias_grad
                       else self.taxonomy_bias_params.bias_table.detach())

                # 用稳定的 callable 对象替代每次新建的闭包：
                # 同一对象 → torch.compile identity guard 命中 → 不重新编译 Triton kernel
                self._score_mod_obj.update(phylum_ids, class_ids, order_ids, family_ids, _bt)
                score_mod = self._score_mod_obj
            else:
                # 回退路径：物化 float bias（SDPA，Flash 会被禁用）
                bucket_matrix = compute_taxonomy_bucket_matrix(taxon_path_ids)  # [B, L, L]
                if _need_bias_grad:
                    taxon_bias = self.taxonomy_bias_params(bucket_matrix)       # [B, nhead, L, L]
                else:
                    with torch.no_grad():
                        taxon_bias = self.taxonomy_bias_params(bucket_matrix)
                full_bias = torch.zeros(
                    B, self.nhead, L_tok + 1, L_tok + 1,
                    dtype=taxon_bias.dtype, device=taxon_bias.device,
                )
                full_bias[:, :, 1:, 1:] = taxon_bias
                attn_bias = full_bias

        # Transformer 前向（两种路径接口不同）
        if self.use_taxonomy_bias:
            h = self.encoder(x, key_padding_mask=key_padding_mask,
                             attn_bias=attn_bias, score_mod=score_mod)
        else:
            h = self.encoder(x, src_key_padding_mask=key_padding_mask)

        h = self.layer_norm(h)
        sample_repr = h[:, 0, :]
        return h, sample_repr
