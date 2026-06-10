"""V5 PMA(Pooling by Multihead Attention) — Set Transformer 风格的 sample 聚合层。

来自 Lee et al. 2019 "Set Transformer"。
设计要点(design 文档 §3):
  - 一个 learnable query 向量(k=1 起步,本期不实现 k>1)
  - 通过 multihead attention 让 query 去"问"每个 token 重要性
  - 输出是所有 token 的加权和(权重由 attention 给出)
  - key_padding_mask 只屏蔽 PAD,不屏蔽 MLM mask 位置(§3.4)
  - query 用小标准差初始化(std=0.02),避免训练初期 PMA 输出剧烈波动
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PMA(nn.Module):
    """Pooling by Multihead Attention.

    Args:
        d_model:    encoder 输出维度
        nhead_pma:  PMA 内 attention 头数(独立于 encoder 的 nhead,默认 4)
        k:          query 数量(本期固定 1,k>1 留待后续 ablation)
    """

    def __init__(self, d_model: int, nhead_pma: int = 4, k: int = 1) -> None:
        super().__init__()
        if k != 1:
            raise ValueError(f"PMA currently only supports k=1, got k={k}. (k>1 planned for future ablation)")
        self.d_model = d_model
        self.nhead_pma = nhead_pma
        self.k = k

        # 小标准差初始化(0.02),避免训练初期 PMA 输出剧烈波动
        self.query = nn.Parameter(torch.randn(k, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(d_model, nhead_pma, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        h: torch.Tensor,                                # [B, L, d_model]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, L] bool, True=PAD(被屏蔽)
    ) -> torch.Tensor:
        """
        Returns:
            sample_repr: [B, d_model] (k=1 时已 squeeze)
        """
        B = h.size(0)
        # query: [k, d_model] → [B, k, d_model]
        q = self.query.unsqueeze(0).expand(B, -1, -1)
        out, _ = self.mha(q, h, h, key_padding_mask=key_padding_mask)
        # k=1 时 squeeze 中间维 → [B, d_model]
        return self.norm(out).squeeze(1)
