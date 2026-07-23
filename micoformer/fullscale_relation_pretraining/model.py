"""User-approved full-size F1/F2 relation student.

The final sample interface is the valid-token mean before and after L2
normalization.  Teacher-specific projection heads and the masked-rclr head are
training-time components; neither replaces the final sample representation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from micoformer.models.attn_bias import (
    BiasedTransformerEncoder,
    BiasedTransformerEncoderLayer,
)
from micoformer.models.heads import AbundanceRegressionHead
from micoformer.relation_pretraining.model import (
    DEFAULT_VOCAB_SIZE,
    PAD_TOKEN_ID,
)


FullscaleRelationArm = Literal["f1_shared", "f2_dual"]
FULLSCALE_RELATION_ARMS: tuple[FullscaleRelationArm, ...] = (
    "f1_shared",
    "f2_dual",
)
TEACHER_NAMES: tuple[str, str] = ("protein", "unifrac")


@dataclass(frozen=True)
class FullscaleRelationModelConfig:
    """Full-size defaults plus smaller values for structural tests only."""

    arm: FullscaleRelationArm = "f2_dual"
    vocab_size: int = DEFAULT_VOCAB_SIZE
    d_model: int = 512
    rclr_hidden_dim: int = 64
    num_layers: int = 12
    encoder_heads: int = 16
    encoder_ffn_dim: int = 2_048
    projection_dim: int = 256
    shared_head_hidden_dim: int = 2_048
    dual_head_hidden_dim: int = 1_024
    mlm_head_hidden_dim: int = 64
    dropout: float = 0.1
    max_seq_len: int = 512
    grad_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.arm not in FULLSCALE_RELATION_ARMS:
            raise ValueError(f"arm must be one of {FULLSCALE_RELATION_ARMS}")
        positive = {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "rclr_hidden_dim": self.rclr_hidden_dim,
            "num_layers": self.num_layers,
            "encoder_heads": self.encoder_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "projection_dim": self.projection_dim,
            "shared_head_hidden_dim": self.shared_head_hidden_dim,
            "dual_head_hidden_dim": self.dual_head_hidden_dim,
            "mlm_head_hidden_dim": self.mlm_head_hidden_dim,
            "max_seq_len": self.max_seq_len,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.vocab_size < 3:
            raise ValueError("vocab_size must include PAD, reserved id 1, and real genera")
        if self.d_model % self.encoder_heads != 0:
            raise ValueError("d_model must be divisible by encoder_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class FullscaleRelationModelOutput:
    h_raw: Tensor
    h_unit: Tensor
    teacher_z: dict[str, Tensor]
    token_embeddings: Tensor
    padding_mask: Tensor
    mlm_prediction: Tensor | None = None


def masked_token_mean(h: Tensor, padding_mask: Tensor) -> Tensor:
    """Average only real genus tokens and fail on empty samples."""

    if h.ndim != 3 or padding_mask.shape != h.shape[:2]:
        raise ValueError("h/padding_mask must have shapes [B,L,D] and [B,L]")
    if padding_mask.dtype != torch.bool:
        raise TypeError("padding_mask must use torch.bool")
    valid = (~padding_mask).to(dtype=h.dtype).unsqueeze(-1)
    denominator = valid.sum(dim=1)
    if bool((denominator <= 0).any()):
        raise ValueError("every sample must contain at least one valid token")
    return (h * valid).sum(dim=1) / denominator


class FullscaleRelationTokenStem(nn.Module):
    """Genus identity plus retained-support no-sigma rclr with an explicit mask flag."""

    def __init__(self, config: FullscaleRelationModelConfig) -> None:
        super().__init__()
        self.genus_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=PAD_TOKEN_ID,
        )
        self.rclr_mlp = nn.Sequential(
            nn.Linear(1, config.rclr_hidden_dim),
            nn.GELU(),
            nn.Linear(config.rclr_hidden_dim, config.d_model),
            nn.LayerNorm(config.d_model),
        )
        self.abundance_mask_embedding = nn.Parameter(torch.empty(config.d_model))
        nn.init.normal_(self.abundance_mask_embedding, mean=0.0, std=0.02)

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
            raise TypeError("abundance_mask must be bool with the same [B,L] shape")
        if bool((abundance_mask & padding_mask).any()):
            raise ValueError("abundance_mask may select only valid genus tokens")
        masked_rclr = rclr.masked_fill(abundance_mask, 0.0)
        abundance_embedding = self.rclr_mlp(masked_rclr.unsqueeze(-1))
        token_embedding = self.genus_embedding(genus_ids) + abundance_embedding
        token_embedding = token_embedding + (
            abundance_mask.unsqueeze(-1).to(token_embedding.dtype)
            * self.abundance_mask_embedding
        )
        return token_embedding.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class RelationProjectionHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, h_raw: Tensor) -> Tensor:
        projected = self.network(h_raw)
        return F.normalize(projected.float(), p=2.0, dim=-1, eps=1e-12)


class FullscaleRelationModel(nn.Module):
    """Shared 12L/d512 encoder with parameter-matched F1/F2 teacher heads."""

    def __init__(self, config: FullscaleRelationModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or FullscaleRelationModelConfig()
        self.input_stem = FullscaleRelationTokenStem(self.config)
        layer = BiasedTransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.encoder_heads,
            dim_feedforward=self.config.encoder_ffn_dim,
            dropout=self.config.dropout,
        )
        self.encoder = BiasedTransformerEncoder(
            layer,
            num_layers=self.config.num_layers,
            grad_checkpointing=self.config.grad_checkpointing,
        )
        self.final_token_norm = nn.LayerNorm(self.config.d_model)
        # Initialize every shared component before arm-specific heads so the
        # same seed gives F1/F2 an identical backbone, mask token and MLM head.
        self.mlm_head = AbundanceRegressionHead(
            d_model=self.config.d_model,
            hidden=self.config.mlm_head_hidden_dim,
        )
        if self.config.arm == "f1_shared":
            self.shared_teacher_head: RelationProjectionHead | None = RelationProjectionHead(
                self.config.d_model,
                self.config.shared_head_hidden_dim,
                self.config.projection_dim,
            )
            self.teacher_heads: nn.ModuleDict | None = None
        else:
            self.shared_teacher_head = None
            self.teacher_heads = nn.ModuleDict(
                {
                    name: RelationProjectionHead(
                        self.config.d_model,
                        self.config.dual_head_hidden_dim,
                        self.config.projection_dim,
                    )
                    for name in TEACHER_NAMES
                }
            )
    def _validate_inputs(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Tensor | None,
    ) -> Tensor:
        if genus_ids.ndim != 2:
            raise ValueError(f"genus_ids must have shape [B,L], got {tuple(genus_ids.shape)}")
        if genus_ids.dtype != torch.long:
            raise TypeError("genus_ids must use torch.long")
        if rclr.shape != genus_ids.shape or not torch.is_floating_point(rclr):
            raise TypeError("rclr must be floating point with the same [B,L] shape")
        if genus_ids.shape[1] > self.config.max_seq_len or genus_ids.numel() == 0:
            raise ValueError("input is empty or exceeds max_seq_len")
        if bool(((genus_ids < 0) | (genus_ids >= self.config.vocab_size)).any()):
            raise ValueError("genus_ids contain an id outside the configured vocabulary")
        if not bool(torch.isfinite(rclr).all()):
            raise ValueError("rclr contains NaN or infinity")
        inferred = genus_ids.eq(PAD_TOKEN_ID)
        if key_padding_mask is None:
            padding_mask = inferred
        else:
            if key_padding_mask.shape != genus_ids.shape or key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must be bool with the same [B,L] shape")
            if not torch.equal(key_padding_mask, inferred):
                raise ValueError("key_padding_mask must mark exactly id-0 PAD positions")
            padding_mask = key_padding_mask
        if bool(padding_mask.all(dim=1).any()):
            raise ValueError("every sample must contain at least one non-PAD token")
        return padding_mask

    def _encode(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
        abundance_mask: Tensor | None,
    ) -> Tensor:
        h = self.input_stem(genus_ids, rclr, padding_mask, abundance_mask)
        h = self.encoder(h, key_padding_mask=padding_mask, attn_bias=None)
        h = self.final_token_norm(h)
        return h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

    def _teacher_outputs(self, h_raw: Tensor) -> dict[str, Tensor]:
        if self.shared_teacher_head is not None:
            shared = self.shared_teacher_head(h_raw)
            return {name: shared for name in TEACHER_NAMES}
        if self.teacher_heads is None:
            raise RuntimeError("dual teacher heads are missing")
        return {name: self.teacher_heads[name](h_raw) for name in TEACHER_NAMES}

    def forward_relation(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> FullscaleRelationModelOutput:
        """Unmasked relation forward; this path never invokes the MLM head."""

        padding_mask = self._validate_inputs(genus_ids, rclr, key_padding_mask)
        h = self._encode(genus_ids, rclr, padding_mask, abundance_mask=None)
        h_raw = masked_token_mean(h, padding_mask)
        h_unit = F.normalize(h_raw.float(), p=2.0, dim=-1, eps=1e-12)
        return FullscaleRelationModelOutput(
            h_raw=h_raw,
            h_unit=h_unit,
            teacher_z=self._teacher_outputs(h_raw),
            token_embeddings=h,
            padding_mask=padding_mask,
        )

    def forward_mlm(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        abundance_mask: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> FullscaleRelationModelOutput:
        """Masked-rclr forward; genus ids remain visible and only masked targets score."""

        padding_mask = self._validate_inputs(genus_ids, rclr, key_padding_mask)
        if abundance_mask.shape != genus_ids.shape or abundance_mask.dtype != torch.bool:
            raise TypeError("abundance_mask must be bool with the same [B,L] shape")
        if not bool(abundance_mask.any()):
            raise ValueError("MLM forward requires at least one masked valid token")
        h = self._encode(genus_ids, rclr, padding_mask, abundance_mask)
        h_raw = masked_token_mean(h, padding_mask)
        h_unit = F.normalize(h_raw.float(), p=2.0, dim=-1, eps=1e-12)
        return FullscaleRelationModelOutput(
            h_raw=h_raw,
            h_unit=h_unit,
            teacher_z={},
            token_embeddings=h,
            padding_mask=padding_mask,
            mlm_prediction=self.mlm_head(h),
        )

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Tensor | None = None,
        *,
        mode: Literal["relation", "mlm"] = "relation",
        abundance_mask: Tensor | None = None,
    ) -> FullscaleRelationModelOutput:
        if mode == "relation":
            if abundance_mask is not None:
                raise ValueError("relation forward may not receive abundance_mask")
            return self.forward_relation(genus_ids, rclr, key_padding_mask)
        if mode == "mlm":
            if abundance_mask is None:
                raise ValueError("MLM forward requires abundance_mask")
            return self.forward_mlm(genus_ids, rclr, abundance_mask, key_padding_mask)
        raise ValueError("mode must be 'relation' or 'mlm'")
