"""JEPA per-token 潜空间 predictor(2026-06-04 起)。

设计依据(见 .claude/docs/.../tmp/20260604_jepa/PLAN.md 检索固化):
  - I-JEPA(Assran 2023):predictor 是**窄 bottleneck** transformer(宽度 << encoder),
    bottleneck 本身是防塌主力之一。
  - 红线(用户最关心):被遮 genus 的 phylo/protein **坐标只当"问哪个 genus"的地址
    (query 输入)**,真正要预测的 target 是 EMA target encoder 输出的"含义向量"
    (在 module 里算)。predictor 绝不直接拟合坐标——否则退化成已被证伪的 X2_phylo。

forward 思路(I-JEPA 式):
  组装一个长度 L 的序列:
    - context 位置 → 放 context encoder 输出 h_ctx(这些 token 模型看得见)
    - target  位置 → 放 "mask query" = 可学习 mask_token + 被遮 genus 坐标地址投影
  跑 depth 层窄 transformer(full attention:target 能 attend context 聚合信息),
  取 target 位置输出投回 d_model = 预测的"含义向量"。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class JEPAPredictor(nn.Module):
    """窄 bottleneck transformer predictor。

    Args:
        d_model:        encoder 输出维度(context 表征 / target 含义向量 / 本模块输出 都是它)
        pred_dim:       predictor 内部宽度(窄 bottleneck,典型 256 = d_model 的 0.5x)
        depth:          predictor transformer 层数(典型 2)
        nhead:          predictor 注意力头数(pred_dim 须能整除,典型 4)
        phylo_pe_dim:   phylo 坐标维度(当地址 query;None=不用 phylo 地址)
        protein_pe_dim: protein 坐标维度(当地址 query;None=不用 protein 地址)
        dropout:        predictor dropout(I-JEPA predictor 通常 0)
    """

    def __init__(
        self,
        *,
        d_model: int,
        pred_dim: int = 256,
        depth: int = 2,
        nhead: int = 4,
        phylo_pe_dim: Optional[int] = None,
        protein_pe_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # 被遮位置的可学习 mask query seed(类 I-JEPA mask token);小随机初始化
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # 坐标地址投影:把 frozen 几何坐标投到 d_model,加到 mask query 上当"问哪个 genus"
        # (红线:这是地址/索引,不是要拟合的答案)
        self.coord_proj_phylo: Optional[nn.Linear] = (
            nn.Linear(phylo_pe_dim, d_model) if phylo_pe_dim else None
        )
        self.coord_proj_protein: Optional[nn.Linear] = (
            nn.Linear(protein_pe_dim, d_model) if protein_pe_dim else None
        )

        # 窄 bottleneck:d_model → pred_dim → (transformer) → d_model
        self.in_proj = nn.Linear(d_model, pred_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=pred_dim,
            nhead=nhead,
            dim_feedforward=pred_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,            # pre-LN,from-scratch 更稳
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(pred_dim, d_model)

    def forward(
        self,
        h_ctx: torch.Tensor,            # [B, L, d_model]  context encoder 输出
        target_mask: torch.Tensor,      # [B, L]  bool,被遮(要预测)的位置
        full_mask: torch.Tensor,        # [B, L]  bool,所有有效位置(True=valid,含 ctx+target)
        phylo_coords: Optional[torch.Tensor] = None,    # [B, L, phylo_pe_dim]  frozen 地址
        protein_coords: Optional[torch.Tensor] = None,  # [B, L, protein_pe_dim]
        genus_query: Optional[torch.Tensor] = None,     # [B, L, d_model]  被遮菌身份地址(Cell-JEPA 式)
    ) -> torch.Tensor:
        """Returns: [B, L, d_model];调用方取 target_mask 位置当预测。"""
        B, L, d = h_ctx.shape

        # 1) 构造 target 位置的 mask query = mask_token + 地址(genus 身份 或 坐标)
        query = self.mask_token.view(1, 1, d).expand(B, L, d)
        if genus_query is not None:                       # Cell-JEPA 式:用"哪个 genus"(身份)定位被遮位置
            query = query + genus_query
        if self.coord_proj_phylo is not None and phylo_coords is not None:
            query = query + self.coord_proj_phylo(phylo_coords)
        if self.coord_proj_protein is not None and protein_coords is not None:
            query = query + self.coord_proj_protein(protein_coords)

        # 2) 组合输入:context 位置用 h_ctx,target 位置用 query(覆盖 h_ctx 在 target 的值)
        tm = target_mask.unsqueeze(-1)            # [B, L, 1]
        inp = torch.where(tm, query, h_ctx)

        # 3) 窄 transformer:full attention(target 能 attend context),PAD 屏蔽
        x = self.in_proj(inp)
        key_padding_mask = ~full_mask             # True=屏蔽(PAD)
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        out = self.out_proj(x)
        return out
