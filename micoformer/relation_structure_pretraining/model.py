"""Matched sample-representation variants over the frozen relation backbone.

The existing ``relation_pretraining`` package remains byte-for-byte unchanged so
historical schema-v1 checkpoints retain their fail-closed source binding.  This
module reuses the exact C0 token stem/encoder/decoder objects and only changes
which sample representation receives relation loss and downstream readout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from micoformer.relation_pretraining.model import RelationOnlyModel


StructureArm = Literal["c0_decoder", "c1_token_mean", "c2_projector"]
STRUCTURE_ARMS: tuple[StructureArm, ...] = (
    "c0_decoder",
    "c1_token_mean",
    "c2_projector",
)


@dataclass
class StructureRelationOutput:
    """Relation-loss and downstream representations from one forward pass."""

    # ``z``/``z_raw`` intentionally preserve the parent training-module API:
    # relation mining and relation loss always consume ``z``.
    z: Tensor
    z_raw: Tensor
    token_embeddings: Tensor
    padding_mask: Tensor
    backbone_raw: Tensor
    backbone_z: Tensor
    downstream_z: Tensor
    projector_raw: Tensor | None = None
    projector_z: Tensor | None = None


def masked_token_mean(h: Tensor, padding_mask: Tensor) -> Tensor:
    """Mean only over valid tokens; every sample must have at least one."""

    if h.ndim != 3 or padding_mask.shape != h.shape[:2]:
        raise ValueError("h/padding_mask must have shapes [B,L,D] and [B,L]")
    if padding_mask.dtype != torch.bool:
        raise TypeError("padding_mask must be bool")
    valid = (~padding_mask).to(dtype=h.dtype).unsqueeze(-1)
    denominator = valid.sum(dim=1)
    if bool((denominator <= 0).any()):
        raise ValueError("every sample must contain at least one valid token")
    return (h * valid).sum(dim=1) / denominator


class RelationProjector(nn.Module):
    """Frozen C2 package: 256 -> 512 -> 256 with GELU and LayerNorm."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model != 256:
            raise ValueError("the frozen C2 projector contract requires d_model=256")
        self.network = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, d_model),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.network(h)


class StructureRelationModel(nn.Module):
    """Reuse an initialized C0 model while changing only sample packaging.

    Constructing every arm first creates the exact same initialized historical
    ``RelationOnlyModel(main)``.  The shared stem, encoder and final token norm
    are moved into this module, which guarantees common tensors are identical.
    C0 also reuses the exact learned decoder; C1 discards it; C2 replaces it
    with the preregistered relation projector.
    """

    def __init__(self, base: RelationOnlyModel, arm: StructureArm) -> None:
        super().__init__()
        if arm not in STRUCTURE_ARMS:
            raise ValueError(f"unknown structure arm: {arm!r}")
        if base.config.decoder_kind != "main":
            raise ValueError("all structure arms must start from the main C0 initialization")
        self.config = base.config
        self.structure_arm = arm
        self.input_stem = base.input_stem
        self.encoder = base.encoder
        self.final_token_norm = base.final_token_norm
        if arm == "c0_decoder":
            self.decoder: nn.Module | None = base.decoder
        else:
            self.decoder = None
        if arm == "c2_projector":
            self.relation_projector: RelationProjector | None = RelationProjector(
                self.config.d_model
            )
        else:
            self.relation_projector = None

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> StructureRelationOutput:
        # Reuse the historical fail-closed input validation exactly.
        validator = RelationOnlyModel._validate_inputs
        padding_mask = validator(self, genus_ids, rclr, key_padding_mask)
        h = self.input_stem(genus_ids, rclr, padding_mask)
        h = self.encoder(h, key_padding_mask=padding_mask, attn_bias=None)
        h = self.final_token_norm(h)
        h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        backbone_raw = masked_token_mean(h, padding_mask)
        backbone_z = F.normalize(backbone_raw.float(), p=2.0, dim=-1, eps=1e-12)
        projector_raw: Tensor | None = None
        projector_z: Tensor | None = None

        if self.structure_arm == "c0_decoder":
            if self.decoder is None:
                raise RuntimeError("C0 decoder is missing")
            relation_raw = self.decoder(h, padding_mask)
            relation_z = F.normalize(relation_raw.float(), p=2.0, dim=-1, eps=1e-12)
            downstream_z = relation_z
        elif self.structure_arm == "c1_token_mean":
            relation_raw = backbone_raw
            relation_z = backbone_z
            downstream_z = backbone_z
        else:
            if self.relation_projector is None:
                raise RuntimeError("C2 relation projector is missing")
            projector_raw = self.relation_projector(backbone_raw)
            projector_z = F.normalize(projector_raw.float(), p=2.0, dim=-1, eps=1e-12)
            relation_raw = projector_raw
            relation_z = projector_z
            # C2 primary downstream representation is explicitly pre-projector.
            downstream_z = backbone_z

        return StructureRelationOutput(
            z=relation_z,
            z_raw=relation_raw,
            token_embeddings=h,
            padding_mask=padding_mask,
            backbone_raw=backbone_raw,
            backbone_z=backbone_z,
            downstream_z=downstream_z,
            projector_raw=projector_raw,
            projector_z=projector_z,
        )

