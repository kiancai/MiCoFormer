"""MPC model whose token identity is the fixed joint PHY+PROT descriptor.

This module deliberately leaves the first full-data MPC implementation untouched.
Only the backbone input stem is replaced; objectives, Query candidate geometry,
encoder, heads and deliverable valid-token mean are inherited verbatim.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from micoformer.mpc_pretraining.model import MPCModelConfig, MPCPretrainingModel


@dataclass(frozen=True)
class MPCFixedModelConfig(MPCModelConfig):
    """Hash-visible declaration of the pure fixed-context endpoint."""

    context_identity: str = "fixed_phy32_prot480_direct"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.context_identity != "fixed_phy32_prot480_direct":
            raise ValueError("the fixed-context identity contract drifted")


class FixedPriorTokenStem(nn.Module):
    """Fixed descriptor plus the matched trainable abundance stem.

    There is intentionally no trainable per-genus lookup, residual or shared
    descriptor projection.  The 512-dimensional fixed table is injected again
    when a checkpoint is restored and therefore is not duplicated in state_dict.
    """

    def __init__(
        self,
        *,
        fixed_table: Tensor,
        rclr_mlp: nn.Module,
        abundance_mask_embedding: nn.Parameter,
    ) -> None:
        super().__init__()
        if fixed_table.ndim != 2 or fixed_table.shape[1] != abundance_mask_embedding.numel():
            raise ValueError("fixed table must be [vocab,d_model]")
        if not bool(torch.isfinite(fixed_table).all()):
            raise ValueError("fixed table contains non-finite values")
        self.register_buffer(
            "fixed_table", fixed_table.detach().float().clone(), persistent=False
        )
        self.rclr_mlp = rclr_mlp
        self.abundance_mask_embedding = abundance_mask_embedding

    def identity(self, genus_ids: Tensor) -> Tensor:
        return F.embedding(genus_ids, self.fixed_table, padding_idx=0)

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
        abundance_mask: Tensor | None = None,
    ) -> Tensor:
        if abundance_mask is None:
            abundance_mask = torch.zeros_like(padding_mask)
        if abundance_mask.shape != padding_mask.shape or abundance_mask.dtype != torch.bool:
            raise TypeError("abundance_mask must be aligned bool [B,L]")
        if bool((abundance_mask & padding_mask).any()):
            raise ValueError("abundance mask selected padding")
        abundance = self.rclr_mlp(
            rclr.masked_fill(abundance_mask, 0.0).unsqueeze(-1)
        )
        abundance = abundance + (
            abundance_mask.unsqueeze(-1).to(abundance.dtype)
            * self.abundance_mask_embedding
        )
        output = self.identity(genus_ids) + abundance
        return output.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class MPCFixedPretrainingModel(MPCPretrainingModel):
    """Pure fixed-PP input context with the unchanged MPC objectives."""

    def __init__(
        self,
        candidate_table: Tensor,
        config: MPCFixedModelConfig | None = None,
    ) -> None:
        fixed_config = config or MPCFixedModelConfig()
        # Construct the original model first.  Besides maximizing reuse, this
        # preserves RNG consumption so all non-identity modules have the same
        # seed-42 initialization order as the learned-ID full run.
        super().__init__(candidate_table, config=fixed_config)
        old_stem = self.backbone.input_stem
        self.backbone.input_stem = FixedPriorTokenStem(
            fixed_table=candidate_table,
            rclr_mlp=old_stem.rclr_mlp,
            abundance_mask_embedding=old_stem.abundance_mask_embedding,
        )
        if any(isinstance(module, nn.Embedding) for module in self.backbone.input_stem.modules()):
            raise RuntimeError("fixed context unexpectedly contains a trainable lookup")
