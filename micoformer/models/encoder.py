from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

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

    可选 flag:
      - abundance_encoding='bin' 时用 self.abund_embed(nn.Embedding) 代替 self.abund_mlp
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
        # hierarchical 删除后此参数不再使用,保留以免签名/ckpt-hparam 改动(调用方仍传它)
        rank_vocab_sizes: Dict[str, int],
        # V4 R2:距离驱动的 attention bias
        # - "none"  :baseline,不注入任何距离 bias
        # - "taxo"  :离散 7-bucket bias_table,查 varp['taxo_dist']
        # - "phylo" :3 层 MLP bias,查 varp['phylo_dist'](V5 默认)
        bias_type: str = "none",
        # phylo MLP 隐藏层维度(仅 bias_type="phylo" 时生效);V5 默认 64(3 层 MLP)
        phylo_mlp_hidden: int = 64,
        # phylo MLP 末层是否保留 bias 项(仅 bias_type="phylo" 时生效)。False=关掉末层 bias,
        # 见 attn_bias.PhyloDistBias 注释。默认 True 保持现有 ckpt 兼容。
        phylo_bias_last_layer_bias: bool = True,
        # 词表大小(用于在不持有 dist_matrix 时占位创建 buffer,避免 ckpt 加载时无 buffer)
        n_vars: Optional[int] = None,
        # ---------------- V5 新增 ----------------
        abundance_encoding: str = "mlp",      # "mlp"(默认) | "bin"
        use_phylo_pe: bool = True,             # 启用 PhyloPE(V5 默认 True)
        phylo_pe_hidden: int = 128,
        pe_dim: Optional[int] = None,          # PE 坐标维度;use_phylo_pe=True 时必须
        # X2 多任务(2026-05-28 夜):蛋白功能 prior,镜像 phylo_pe 同构
        use_protein_pe: bool = False,          # 启用 ProteinPE(等 bacformer_prior 出 varm['protein_pe'])
        protein_pe_hidden: int = 128,
        protein_pe_dim: Optional[int] = None,  # protein PE 坐标维度;use_protein_pe=True 时必须
        grad_checkpointing: bool = False,       # 激活重算开关(以时间换显存),默认关
    ) -> None:
        super().__init__()
        self.pad_taxon_id = pad_taxon_id
        self.nhead = nhead

        if bias_type not in _VALID_BIAS_TYPES:
            raise ValueError(
                f"Unknown bias_type: {bias_type!r}. Expected one of {sorted(_VALID_BIAS_TYPES)}."
            )
        self.bias_type = bias_type

        if abundance_encoding not in _VALID_ABUNDANCE_ENCODING:
            raise ValueError(
                f"Unknown abundance_encoding: {abundance_encoding!r}. "
                f"Expected one of {sorted(_VALID_ABUNDANCE_ENCODING)}."
            )
        self.abundance_encoding = abundance_encoding
        self.use_phylo_pe = use_phylo_pe
        self.use_protein_pe = use_protein_pe

        # ============ Token identity embedding ============
        # genus_embed:V5 单 genus embedding。保留 self.taxon_embed 作为同一对象的别名,ckpt 兼容
        self.taxon_embed = nn.Embedding(genus_vocab_size, d_model, padding_idx=pad_taxon_id)
        # 别名(V5 语义),与 taxon_embed 共享对象
        self.genus_embed = self.taxon_embed

        # ============ Abundance embedding ============
        # bin 路径:nn.Embedding(num_bins+2, d_model)
        # mlp 路径(V5 默认):MLP(1 → d/4 → d) + LayerNorm + abund_mask_token
        # 仅创建当前 abundance_encoding 对应的权重(避免无关参数不参与 forward → DDP find_unused)
        if self.abundance_encoding == "mlp":
            self.abund_embed = None
            self.abund_mlp = nn.Sequential(
                nn.Linear(1, max(1, d_model // 4)),
                nn.GELU(),
                nn.Linear(max(1, d_model // 4), d_model),
                nn.LayerNorm(d_model),
            )
            self.abund_mask_token = nn.Parameter(torch.zeros(d_model))
        else:
            self.abund_embed = nn.Embedding(total_abundance_bins, d_model, padding_idx=pad_bin_id)
            self.abund_mlp = None
            self.abund_mask_token = None

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

        # ============ Protein PE(X2 多任务,2026-05-28 夜) ============
        # 蛋白功能 prior,镜像 phylo_pe:复用同一 PhyloPE 类(语义对偶)
        # 等 bacformer_prior 出 varm['protein_pe'] 才启用;phase 1 默认 None
        self.protein_pe: Optional[PhyloPE] = None
        if self.use_protein_pe:
            if protein_pe_dim is None:
                raise ValueError(
                    "use_protein_pe=True requires protein_pe_dim (蛋白 PE 坐标维度)."
                )
            self.protein_pe = PhyloPE(
                d_model=d_model,
                pe_dim=protein_pe_dim,
                vocab_size=genus_vocab_size,
                hidden=protein_pe_hidden,
            )

        # ============ Genus mask token(X2 范式,2026-05-28 夜) ============
        # 镜像 abund_mask_token:mask 位置的 token embed 被替换为此可学习 token
        # 防止模型从 token_id 直接 lookup phylo coord 答案(X2 任务的硬约束)
        # 默认零初始化,跟 abund_mask_token 一致
        self.genus_mask_token = nn.Parameter(torch.zeros(d_model))

        # ============ Transformer ============
        biased_layer = BiasedTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.encoder = BiasedTransformerEncoder(
            biased_layer, num_layers=num_layers, grad_checkpointing=grad_checkpointing
        )

        # V4 R2:距离 bias 模块(None / TaxoDistBias / PhyloDistBias)
        self.dist_bias = make_dist_bias(
            bias_type=bias_type,
            nhead=nhead,
            phylo_mlp_hidden=phylo_mlp_hidden,
            phylo_last_layer_bias=phylo_bias_last_layer_bias,
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

    def _build_token_embedding(self, token_ids: torch.Tensor) -> torch.Tensor:
        # V5: 单 genus embedding
        return self.taxon_embed(token_ids)

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
        var_indices: Optional[torch.Tensor] = None,         # [B, L]  int64  bias_type!='none' 时必需
        mask_token_id_replace: bool = False,                # X2 范式:mask 位置 token embed 替换为 genus_mask_token + PE 输出乘 0 防作弊
    ) -> torch.Tensor:
        """
        Returns:
            h: [B, L, d_model]  token-level 输出(sample-level PMA pooling 在 module 层做)
        """
        # ============ Token embedding 三/四段相加 ============
        token_x = self._build_token_embedding(token_ids)
        # X2 范式:mask 位置 token embed 替换为可学习 mask token(防 token_id 泄露答案)
        if mask_token_id_replace and mask_positions is not None and mask_positions.any():
            mask_expanded = mask_positions.unsqueeze(-1)  # [B, L, 1]
            token_x = torch.where(mask_expanded, self.genus_mask_token, token_x)
        abund_x = self._build_abundance_embedding(abund_bins, abund_values, mask_positions)
        x = token_x + abund_x
        # phylo PE:mask 位置在 X2 下要乘 0(防 phylo_pe.coords[token_id] 泄露答案)
        if self.use_phylo_pe and self.phylo_pe is not None:
            pe_x = self.phylo_pe(token_ids)
            if mask_token_id_replace and mask_positions is not None and mask_positions.any():
                pe_x = pe_x * (~mask_positions).unsqueeze(-1).to(pe_x.dtype)
            x = x + pe_x
        # protein PE:同 phylo PE 屏蔽逻辑
        if self.use_protein_pe and self.protein_pe is not None:
            ppe_x = self.protein_pe(token_ids)
            if mask_token_id_replace and mask_positions is not None and mask_positions.any():
                ppe_x = ppe_x * (~mask_positions).unsqueeze(-1).to(ppe_x.dtype)
            x = x + ppe_x

        # ============ attention mask ============
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
            attn_bias = self.dist_bias(var_indices, self.dist_matrix)  # [B, nhead, L, L]

        h = self.encoder(x, key_padding_mask=key_padding_mask, attn_bias=attn_bias)
        h = self.layer_norm(h)
        return h
