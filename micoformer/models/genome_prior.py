"""V5 GenomePrior —— 把 Bacformer 基因组 embedding 经可学 attention pool 注入 token embedding。

设计要点(对齐 bacformer_prior architecture.md §3.2-§3.3):
  - species_vecs 来自 bacformer_prior pipeline,形状 [V_total, K_max, d_in=480]
    其中 V_total = V_real + 2(前 2 行 PAD/UNK 全零),V_real 是 anndata genus 数(8114)
  - mask 形状 [V_total, K_max]: True = 有效 species, False = padding / 盲区 fallback
  - 可学 attention pool(共享 query) 把每 token 的 K_max 个 species 向量压成 1 个 genus 向量
  - 沿用 phylo_pe.py 风格:proj 末层 zero-init + learnable scale → 训练 step 0 输出 0
  - species_vecs / mask 作为 buffer(persistent=False),由 workflow 注入

forward 路径(同 phylo_pe.py 一样直接加到 token embedding):
  token_ids → species[V,K,D] + mask → attn_pool → [B,L,D] → proj → LN → * scale → [B,L,d_model]

镜像 phylo_pe.py 的 set_coords → 这里叫 set_species_vecs(vecs, mask)。

当前状态(2026-05-28):骨架,不接入 encoder。等 4.pack_tensor 产物落地 + bacformer 包装好后
由 task #5(协议 protocols/genome_prior_ablation/)正式接入 + 跑 sanity + 跑消融。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GenomeAttnPool(nn.Module):
    """Attention pool over K_max species → 单 genus 向量(共享 query)。

    设计选择(decisions.md §9):
      - 共享 query: 所有 genus 用同一 query,避免 V×D 参数爆炸
      - 单 head: 起步简化,后续若证明不够再扩 multi-head
      - 全 mask=False 行健壮处理: softmax 在全 -1e9 上会 NaN/uniform → 直接置 0
    """

    def __init__(self, d_in: int = 480) -> None:
        super().__init__()
        self.d_in = d_in
        # 共享 query,初始 N(0, 0.02) 避免起步过强
        self.query = nn.Parameter(torch.randn(d_in) * 0.02)
        self.k_proj = nn.Linear(d_in, d_in)
        self.v_proj = nn.Linear(d_in, d_in)

    def forward(self, species: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            species: [..., K_max, d_in] 任意 leading shape(如 [B,L,K,D] 或 [V,K,D])
            mask:    [..., K_max] bool, True = 有效
        Returns:
            pooled:  [..., d_in]
        """
        # 投影
        K = self.k_proj(species)
        V_ = self.v_proj(species)
        # 计 attn logits: einsum 内积 K·query
        attn = torch.einsum("...kd,d->...k", K, self.query) / math.sqrt(self.d_in)
        # mask 无效位
        attn = attn.masked_fill(~mask, -1e9)
        weights = attn.softmax(dim=-1)  # [..., K_max]
        # 全无效行(盲区 fallback 完全 mask=0):softmax 退化为 NaN/uniform → 直接置 0
        all_invalid = ~mask.any(dim=-1, keepdim=True)
        weights = torch.where(all_invalid, torch.zeros_like(weights), weights)
        # 加权和
        return (weights.unsqueeze(-1) * V_).sum(dim=-2)


class GenomePrior(nn.Module):
    """V5 Genome Prior(可学 attention pool + zero-init scale 注入)。

    Args:
        d_model:    encoder 输入维度
        d_in:       Bacformer embedding 维度(small=480 / large=960)
        vocab_size: 模型词表总大小(含 PAD/UNK),即 V_real+2
        K_max:      每属保留 species 数(由 4.pack_tensor 决定,候选 16/32/64)
        hidden:     proj MLP 中间维度
    """

    def __init__(
        self,
        d_model: int,
        d_in: int = 480,
        vocab_size: int = 8116,
        K_max: int = 32,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_in = d_in
        self.vocab_size = vocab_size
        self.K_max = K_max
        self.hidden = hidden

        # species_vecs / mask 占位 buffer:persistent=False(不进 ckpt)
        # 等 workflow 调 set_species_vecs() 注入真实数据
        self.register_buffer(
            "species_vecs",
            torch.zeros(vocab_size, K_max, d_in, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "mask",
            torch.zeros(vocab_size, K_max, dtype=torch.bool),
            persistent=False,
        )
        self._loaded = False

        # Attention pool over species
        self.attn_pool = GenomeAttnPool(d_in)

        # 投影 + LN + 可学 scale(镜像 phylo_pe)
        self.proj = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.ones(1))

        # 末层 zero-init:训练 step 0 输出为 0,后续渐进
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def set_species_vecs(self, raw_vecs: torch.Tensor, raw_mask: torch.Tensor) -> None:
        """注入真实 species_vecs + mask(由 workflow 调,镜像 phylo_pe.set_coords)。

        Args:
            raw_vecs: [V_real, K_max, d_in] float, V_real = anndata.n_vars(8114)
            raw_mask: [V_real, K_max] bool, True = 有效 species 槽位
        """
        if raw_vecs.dim() != 3 or raw_vecs.size(1) != self.K_max or raw_vecs.size(2) != self.d_in:
            raise ValueError(
                f"GenomePrior.set_species_vecs expects [V_real, {self.K_max}, {self.d_in}], "
                f"got {tuple(raw_vecs.shape)}"
            )
        v_real = raw_vecs.size(0)
        if v_real + 2 != self.vocab_size:
            raise ValueError(
                f"GenomePrior vocab_size mismatch: model expects {self.vocab_size}, "
                f"raw_vecs gives V_real={v_real} → V_total={v_real + 2}. "
                f"(model vocab_size 必须 = anndata.n_vars + 2)"
            )
        if tuple(raw_mask.shape) != (v_real, self.K_max):
            raise ValueError(
                f"GenomePrior.set_species_vecs mask shape mismatch: expected "
                f"[{v_real}, {self.K_max}], got {tuple(raw_mask.shape)}"
            )
        # 前置 2 行 0(PAD/UNK)+ mask=False
        vecs = torch.cat(
            [torch.zeros(2, self.K_max, self.d_in, dtype=torch.float32), raw_vecs.float()],
            dim=0,
        )
        mask = torch.cat(
            [torch.zeros(2, self.K_max, dtype=torch.bool), raw_mask.bool()],
            dim=0,
        )
        # 保持 buffer 在原 device 上
        device = self.species_vecs.device
        self.register_buffer("species_vecs", vecs.to(device), persistent=False)
        self.register_buffer("mask", mask.to(device), persistent=False)
        self._loaded = True

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] long, 值 ∈ [0, vocab_size)
        Returns:
            pe: [B, L, d_model]
        """
        # 同 phylo_pe:防 silent failure。如果 workflow 漏调 set_species_vecs,
        # 占位 species_vecs 全 0 + mask 全 False → attn_pool 走 all_invalid 分支返 0,
        # 训练初期看不出问题但 proj 学到非零后 PE 退化为 norm(bias)*scale 常量 → silent 失效
        if not self._loaded:
            raise RuntimeError(
                "GenomePrior.species_vecs has not been loaded. Call "
                "genome_prior.set_species_vecs(raw_vecs, raw_mask) before forward(). "
                "After loading from a checkpoint, species_vecs/mask are reset to "
                "placeholders (persistent=False) and must be re-injected by the workflow."
            )
        species = self.species_vecs[token_ids]  # [B, L, K_max, d_in]
        mask = self.mask[token_ids]              # [B, L, K_max]
        pooled = self.attn_pool(species, mask)   # [B, L, d_in]
        out = self.proj(pooled)                   # [B, L, d_model]
        out = self.norm(out)
        return out * self.scale
