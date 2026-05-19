"""V5 PhyloPE — 把预计算的欧氏 PE coords 投影到 d_model,加到 token embedding。

设计要点(design 文档 §2.3):
  - coords 来自 adata.varm['position_encoding'](V2 中是 [8114, 32] float32)
  - 模型内部 coords buffer 形状为 [vocab_size_total, pe_dim] = [V_real+2, pe_dim]
    前 2 行对应 PAD(id=0) / UNK(id=1),填零
  - 投影:2 层 MLP(Linear → GELU → Linear)+ LayerNorm + learnable scale
  - 末层 zero-init:训练 step 0 PE 贡献为 0,信号渐进出现
  - coords 作为 buffer 冻结(persistent=False 不进 ckpt;workflow 必须重新注入)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhyloPE(nn.Module):
    """Phylogenetic Position Encoding.

    Args:
        d_model:    encoder 输入维度
        pe_dim:     PE 坐标维度(V2 是 32)
        vocab_size: 模型词表总大小(含 PAD/UNK),即 V_real+2
        hidden:     MLP 中间隐藏维度(默认 128)
    """

    def __init__(self, d_model: int, pe_dim: int, vocab_size: int, hidden: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.pe_dim = pe_dim
        self.vocab_size = vocab_size
        self.hidden = hidden

        # coords buffer:[vocab_size, pe_dim],persistent=False(不进 ckpt,由 workflow 注入)
        # 默认全零占位,等待 set_coords() 注入真实坐标
        self.register_buffer(
            "coords",
            torch.zeros(vocab_size, pe_dim, dtype=torch.float32),
            persistent=False,
        )
        self._coords_loaded = False

        # 2 层 MLP + LayerNorm + learnable scale
        self.proj = nn.Sequential(
            nn.Linear(pe_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.scale = nn.Parameter(torch.ones(1))

        # Zero-init 最后一层:训练 step 0 PE 输出为 0
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def set_coords(self, raw_coords: torch.Tensor) -> None:
        """注入真实 PE coords。

        Args:
            raw_coords: [V_real, pe_dim] float32, V_real 是 anndata 实际 genus 数(如 8114)
                       本方法会在前面补 2 行 0(对应模型 vocab 中的 PAD=0 / UNK=1)
        """
        if raw_coords.dim() != 2 or raw_coords.size(1) != self.pe_dim:
            raise ValueError(
                f"PhyloPE.set_coords expects [V_real, {self.pe_dim}], got {tuple(raw_coords.shape)}"
            )
        v_real = raw_coords.size(0)
        expected_v_total = v_real + 2  # +2 for PAD/UNK
        if expected_v_total != self.vocab_size:
            raise ValueError(
                f"PhyloPE vocab_size mismatch: model expects {self.vocab_size}, "
                f"raw_coords gives V_real={v_real} → V_total={expected_v_total}. "
                f"(model vocab_size 必须 = anndata.n_vars + 2)"
            )
        # 前置 2 行 0(PAD/UNK),与真实 coords 拼接
        expanded = torch.cat(
            [torch.zeros(2, self.pe_dim, dtype=torch.float32), raw_coords.float()],
            dim=0,
        )
        # 直接替换 buffer 内容(保持 persistent=False 语义;.to(device) 时仍会一并迁移)
        self.register_buffer("coords", expanded.to(self.coords.device), persistent=False)
        self._coords_loaded = True

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, L] long,值 ∈ [0, vocab_size)
        Returns:
            pe: [B, L, d_model]
        """
        # 防 silent failure:如果 workflow 漏调 set_coords,占位 coords 全 0,
        # 训练初期因 proj zero-init 输出 0 看不出问题,后期 proj weight 学到非零后
        # PE 会退化为 norm(bias)*scale 的常量 → silent 失效
        if not self._coords_loaded:
            raise RuntimeError(
                "PhyloPE.coords has not been loaded. Call phylo_pe.set_coords(raw_coords) "
                "before forward(). After loading from a checkpoint, coords are reset to a "
                "placeholder (persistent=False) and must be re-injected by the workflow."
            )
        coords = self.coords[token_ids]   # [B, L, pe_dim]
        pe = self.proj(coords)             # [B, L, d_model]
        pe = self.norm(pe)
        return pe * self.scale
