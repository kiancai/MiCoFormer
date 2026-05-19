from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from micoformer.data.datasets import RANK_COLUMNS
from micoformer.models.attn_bias import (
    BiasedTransformerEncoder,
    BiasedTransformerEncoderLayer,
    make_dist_bias,
)
from micoformer.models.phylo_pe import PhyloPE


_VALID_BIAS_TYPES = {"none", "taxo", "phylo"}
_VALID_ABUNDANCE_ENCODING = {"mlp", "bin"}


class MiCoFormerEncoder(nn.Module):
    """V5 三段相加 encoder。

    input_token[i] = genus_embed[i] + abundance_embed[i] + phylo_pe[i]
                     └── 身份 ──┘     └── 数值 ──┘     └── 位置/几何 ──┘

    保留旧 flag:
      - use_sample_token=True 时拼接 [SAMPLE] token,使用 self.sample_embed
      - abundance_encoding='bin' 时用 self.abund_embed(nn.Embedding) 代替 self.abund_mlp
      - use_hierarchical_embed=True 时把 5/6-rank embedding 相加(旧 R1)代替单 genus_embed
      - use_phylo_pe=False 时不加 phylo_pe
    """

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
        token_embedding_mode: Optional[str] = None,  # 旧 alias:'taxon' / 'taxon_path';显式传时覆盖 use_hierarchical_embed
        rank_vocab_sizes: Dict[str, int],
        # V4 R2:距离驱动的 attention bias
        # - "none"  :baseline,不注入任何距离 bias
        # - "taxo"  :离散 7-bucket bias_table,查 varp['taxo_dist']
        # - "phylo" :3 层 MLP bias,查 varp['phylo_dist'](V5 默认)
        bias_type: str = "none",
        # phylo MLP 隐藏层维度(仅 bias_type="phylo" 时生效);V5 默认 64(3 层 MLP)
        phylo_mlp_hidden: int = 64,
        # 词表大小(用于在不持有 dist_matrix 时占位创建 buffer,避免 ckpt 加载时无 buffer)
        n_vars: Optional[int] = None,
        # ---------------- V5 新增 ----------------
        abundance_encoding: str = "mlp",      # "mlp"(默认) | "bin"
        use_phylo_pe: bool = True,             # 启用 PhyloPE(V5 默认 True)
        phylo_pe_hidden: int = 128,
        pe_dim: Optional[int] = None,          # PE 坐标维度;use_phylo_pe=True 时必须
        use_sample_token: bool = False,        # V5 默认 False(删 [SAMPLE]);True 时启用旧路径
        use_hierarchical_embed: bool = False,   # V5 默认 False(单 genus embedding)
    ) -> None:
        super().__init__()
        self.pad_taxon_id = pad_taxon_id
        self.nhead = nhead

        if bias_type not in _VALID_BIAS_TYPES:
            raise ValueError(
                f"Unknown bias_type: {bias_type!r}. Expected one of {sorted(_VALID_BIAS_TYPES)}."
            )
        self.bias_type = bias_type

        # 兼容旧 alias:显式传 token_embedding_mode 时覆盖 use_hierarchical_embed
        if token_embedding_mode is not None:
            if token_embedding_mode not in {"taxon", "taxon_path"}:
                raise ValueError(
                    f"Unknown token_embedding_mode: {token_embedding_mode}. "
                    "Expected 'taxon' or 'taxon_path'."
                )
            use_hierarchical_embed = (token_embedding_mode == "taxon_path")
        self.use_hierarchical_embed = bool(use_hierarchical_embed)
        # 保留 token_embedding_mode 供 ckpt 兼容/调试用
        self.token_embedding_mode = "taxon_path" if self.use_hierarchical_embed else "taxon"

        if abundance_encoding not in _VALID_ABUNDANCE_ENCODING:
            raise ValueError(
                f"Unknown abundance_encoding: {abundance_encoding!r}. "
                f"Expected one of {sorted(_VALID_ABUNDANCE_ENCODING)}."
            )
        self.abundance_encoding = abundance_encoding
        self.use_phylo_pe = use_phylo_pe
        self.use_sample_token = use_sample_token

        # ============ Token identity embedding ============
        # genus_embed:V5 默认路径(单 genus embedding)。保留 self.taxon_embed 作为同一对象的别名,旧 ckpt 兼容
        # 当 use_hierarchical_embed=False 时用它;True 时不用(走 rank_embeds 相加)
        self.taxon_embed: Optional[nn.Embedding] = None
        if not self.use_hierarchical_embed:
            self.taxon_embed = nn.Embedding(genus_vocab_size, d_model, padding_idx=pad_taxon_id)
        # 别名(V5 语义),与 taxon_embed 共享对象;若旧路径未构造则为 None
        self.genus_embed = self.taxon_embed

        # rank_embeds:旧 R1 路径(6 级 embedding 相加),use_hierarchical_embed=True 时使用
        self.rank_embeds = nn.ModuleDict()
        if self.use_hierarchical_embed:
            for rank_name in RANK_COLUMNS:
                if rank_name not in rank_vocab_sizes:
                    raise ValueError(
                        f"Missing rank vocab size for '{rank_name}'. "
                        f"Expected ranks: {RANK_COLUMNS}, got: {list(rank_vocab_sizes.keys())}."
                    )
                self.rank_embeds[rank_name] = nn.Embedding(
                    int(rank_vocab_sizes[rank_name]), d_model, padding_idx=0
                )

        # ============ Abundance embedding ============
        # bin 路径(旧):nn.Embedding(num_bins+2, d_model)
        # mlp 路径(V5):MLP(1 → d/4 → d) + LayerNorm + abund_mask_token
        # 两套权重并存,根据 self.abundance_encoding 选用(便于 ablation)
        self.abund_embed = nn.Embedding(total_abundance_bins, d_model, padding_idx=pad_bin_id)
        if self.abundance_encoding == "mlp":
            self.abund_mlp = nn.Sequential(
                nn.Linear(1, max(1, d_model // 4)),
                nn.GELU(),
                nn.Linear(max(1, d_model // 4), d_model),
                nn.LayerNorm(d_model),
            )
            self.abund_mask_token = nn.Parameter(torch.zeros(d_model))
        else:
            # 占位属性,bin 路径下不被消费
            self.abund_mlp = None
            self.abund_mask_token = None

        # ============ [SAMPLE] token(旧路径) ============
        # V5 默认删除([SAMPLE] 不在输入序列中);use_sample_token=True 时启用
        self.sample_embed: Optional[nn.Embedding] = None
        if self.use_sample_token:
            self.sample_embed = nn.Embedding(1, d_model)

        # ============ Phylo PE(V5) ============
        # vocab_size 含 PAD/UNK,而 anndata.n_vars(=V_real)不含 → PhyloPE 内部 = genus_vocab_size
        # genus_vocab_size 已经是 V_real + 2(0=PAD, 1=UNK, 2~=真实 genus)
        # set_coords([V_real, pe_dim]) 会前置 2 行 0
        self.phylo_pe: Optional[PhyloPE] = None
        if self.use_phylo_pe:
            if pe_dim is None:
                raise ValueError(
                    "use_phylo_pe=True requires pe_dim (PE 坐标维度,通常从 datamodule.pe_dim 透传)."
                )
            self.phylo_pe = PhyloPE(
                d_model=d_model,
                pe_dim=pe_dim,
                vocab_size=genus_vocab_size,
                hidden=phylo_pe_hidden,
            )

        # ============ Transformer ============
        biased_layer = BiasedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = BiasedTransformerEncoder(biased_layer, num_layers=num_layers)

        # V4 R2:距离 bias 模块(None / TaxoDistBias / PhyloDistBias)
        self.dist_bias = make_dist_bias(
            bias_type=bias_type,
            nhead=nhead,
            phylo_mlp_hidden=phylo_mlp_hidden,
        )

        # 距离矩阵 buffer:persistent=False(不进 ckpt,需 workflow 重新注入)
        if bias_type != "none":
            if n_vars is None:
                raise ValueError(
                    f"bias_type={bias_type!r} requires n_vars (var 表行数) to allocate "
                    f"the dist_matrix buffer. Pass n_vars=adata.n_vars when building encoder."
                )
            placeholder_dtype = torch.float32 if bias_type == "phylo" else torch.int8
            self.register_buffer(
                "dist_matrix",
                torch.zeros((n_vars, n_vars), dtype=placeholder_dtype),
                persistent=False,
            )
            self._dist_matrix_loaded = False
        else:
            self.dist_matrix = None
            self._dist_matrix_loaded = True

        self.layer_norm = nn.LayerNorm(d_model)

    def set_dist_matrix(self, matrix: torch.Tensor) -> None:
        """注入真实的距离矩阵。"""
        if self.bias_type == "none":
            raise RuntimeError("bias_type='none' does not need a dist_matrix.")
        if matrix.shape != self.dist_matrix.shape:
            raise ValueError(
                f"dist_matrix shape mismatch: expected {tuple(self.dist_matrix.shape)}, "
                f"got {tuple(matrix.shape)}."
            )
        expected_dtype = torch.float32 if self.bias_type == "phylo" else torch.int8
        if matrix.dtype != expected_dtype:
            matrix = matrix.to(expected_dtype)
        self.register_buffer("dist_matrix", matrix.to(self.dist_matrix.device), persistent=False)
        self._dist_matrix_loaded = True

    def _build_token_embedding(
        self,
        token_ids: torch.Tensor,
        taxon_path_ids: torch.Tensor,
    ) -> torch.Tensor:
        # baseline / V5: 用 genus_embed
        # 旧 R1: 6 级 embedding 相加(use_hierarchical_embed=True)
        if self.use_hierarchical_embed:
            token_x = self.rank_embeds[RANK_COLUMNS[0]](taxon_path_ids[:, :, 0])
            for rank_idx, rank_name in enumerate(RANK_COLUMNS[1:], start=1):
                token_x = token_x + self.rank_embeds[rank_name](taxon_path_ids[:, :, rank_idx])
        else:
            token_x = self.taxon_embed(token_ids)
        return token_x

    def _build_abundance_embedding(
        self,
        abund_bins: Optional[torch.Tensor],
        abund_values: Optional[torch.Tensor],
        mask_positions: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.abundance_encoding == "mlp":
            if abund_values is None:
                raise RuntimeError(
                    "abundance_encoding='mlp' requires abund_values to be passed to forward()."
                )
            # [B, L] → [B, L, 1] → MLP → [B, L, d_model]
            abund_x = self.abund_mlp(abund_values.unsqueeze(-1).float())
            # mask 位置替换为 abund_mask_token(广播)
            if mask_positions is not None and mask_positions.any():
                mask_expanded = mask_positions.unsqueeze(-1)  # [B, L, 1]
                abund_x = torch.where(mask_expanded, self.abund_mask_token, abund_x)
            return abund_x
        else:
            # bin 路径
            if abund_bins is None:
                raise RuntimeError(
                    "abundance_encoding='bin' requires abund_bins to be passed to forward()."
                )
            return self.abund_embed(abund_bins)

    def forward(
        self,
        token_ids: torch.Tensor,                            # [B, L]
        attention_mask: torch.Tensor,                       # [B, L]  True=Valid, False=PAD
        *,
        abund_bins: Optional[torch.Tensor] = None,          # [B, L]  bin 路径用
        abund_values: Optional[torch.Tensor] = None,        # [B, L]  mlp 路径用
        mask_positions: Optional[torch.Tensor] = None,      # [B, L]  bool,MLM 被 mask 的位置(mlp 路径用)
        taxon_path_ids: Optional[torch.Tensor] = None,      # [B, L, 6]  use_hierarchical_embed=True 时必须
        var_indices: Optional[torch.Tensor] = None,         # [B, L]  int64  bias_type!='none' 时必需
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            h:           [B, L, d_model] (V5)  或  [B, L+1, d_model] (use_sample_token=True)
            sample_repr: use_sample_token=True 时是 [B, d_model]([SAMPLE] 位);否则 None
                        (V5 路径:PMA pooling 在 module 层做,encoder 不返回 sample_repr)
        """
        B = token_ids.size(0)

        # ============ Token embedding 三段相加 ============
        token_x = self._build_token_embedding(token_ids, taxon_path_ids)
        abund_x = self._build_abundance_embedding(abund_bins, abund_values, mask_positions)
        x = token_x + abund_x
        if self.use_phylo_pe and self.phylo_pe is not None:
            x = x + self.phylo_pe(token_ids)

        # ============ [SAMPLE] 拼接(旧路径) + attention mask ============
        if self.use_sample_token:
            sample_mask = torch.ones((B, 1), dtype=torch.bool, device=token_ids.device)
            key_padding_mask = ~torch.cat([sample_mask, attention_mask], dim=1)
            sample_vec = self.sample_embed.weight.view(1, 1, -1).expand(B, -1, -1)
            x = torch.cat([sample_vec, x], dim=1)  # [B, L+1, d_model]
        else:
            key_padding_mask = ~attention_mask  # [B, L]

        # ============ Attention bias(R2) ============
        attn_bias: Optional[torch.Tensor] = None
        if self.bias_type != "none":
            if var_indices is None:
                raise RuntimeError(
                    f"bias_type={self.bias_type!r} requires var_indices to be passed to forward()."
                )
            if not self._dist_matrix_loaded:
                raise RuntimeError(
                    "dist_matrix has not been loaded. Call encoder.set_dist_matrix(...) before forward()."
                )
            taxon_bias = self.dist_bias(var_indices, self.dist_matrix)  # [B, nhead, L, L]
            if self.use_sample_token:
                # 旧路径:pad 一圈零让 attn_bias 兼容 [B, nhead, L+1, L+1]
                attn_bias = F.pad(taxon_bias, (1, 0, 1, 0), mode="constant", value=0.0)
            else:
                attn_bias = taxon_bias

        h = self.encoder(x, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        h = self.layer_norm(h)

        # V5 路径:sample_repr 由 PMA 在上层做,encoder 不返回(None)
        # 旧路径:返回 [SAMPLE] 位置的输出作 sample_repr
        if self.use_sample_token:
            sample_repr = h[:, 0, :]
        else:
            sample_repr = None
        return h, sample_repr
