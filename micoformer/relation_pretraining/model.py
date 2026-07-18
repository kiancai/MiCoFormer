"""Relation-only MiCoFormer student model.

The module is intentionally independent from the MLM/pretraining model.  The
only student inputs are a genus token id and the retained-support, no-sigma
``rclr`` scalar associated with that token.  Project, site and both teacher
distances remain outside the model and are consumed only by relation mining.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from micoformer.models.attn_bias import (
    BiasedTransformerEncoder,
    BiasedTransformerEncoderLayer,
)


PAD_TOKEN_ID = 0
RESERVED_TOKEN_ID = 1
REAL_GENUS_COUNT = 8_114
DEFAULT_VOCAB_SIZE = REAL_GENUS_COUNT + 2


@dataclass(frozen=True)
class RelationModelConfig:
    """Configuration for the reduced relation-only mechanism pilot.

    The defaults are the locked 6L/d256 pilot.  Smaller values are accepted so
    that structural tests can run cheaply, but a production workflow should
    serialize this configuration and verify it against the experiment contract.
    """

    vocab_size: int = DEFAULT_VOCAB_SIZE
    d_model: int = 256
    rclr_hidden_dim: int = 64
    num_layers: int = 6
    encoder_heads: int = 8
    encoder_ffn_dim: int = 1_024
    decoder_heads: int = 4
    decoder_ffn_dim: int = 1_024
    dropout: float = 0.1
    max_seq_len: int = 512
    decoder_kind: Literal["main", "pma"] = "main"
    grad_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size < 3:
            raise ValueError("vocab_size must include PAD, reserved id 1, and real genera")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.rclr_hidden_dim <= 0 or self.encoder_ffn_dim <= 0 or self.decoder_ffn_dim <= 0:
            raise ValueError("all feed-forward widths must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.d_model % self.encoder_heads != 0:
            raise ValueError("d_model must be divisible by encoder_heads")
        if self.d_model % self.decoder_heads != 0:
            raise ValueError("d_model must be divisible by decoder_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.decoder_kind not in {"main", "pma"}:
            raise ValueError("decoder_kind must be 'main' or 'pma'")


@dataclass
class RelationModelOutput:
    """One shared representation for relation loss and downstream evaluation."""

    z: Tensor
    z_raw: Tensor
    token_embeddings: Tensor
    padding_mask: Tensor


class RelationTokenStem(nn.Module):
    """Add trainable genus identity and scalar retained-support rclr paths."""

    def __init__(self, config: RelationModelConfig) -> None:
        super().__init__()
        self.vocab_size = config.vocab_size
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

    def forward(self, genus_ids: Tensor, rclr: Tensor, padding_mask: Tensor) -> Tensor:
        abundance_embedding = self.rclr_mlp(rclr.unsqueeze(-1))
        token_embedding = self.genus_embedding(genus_ids) + abundance_embedding
        # The scalar MLP has biases, so explicitly make PAD positions inert.
        return token_embedding.masked_fill(padding_mask.unsqueeze(-1), 0.0)


class LearnedSeedDecoder(nn.Module):
    """Locked main decoder: seed cross-attention plus two residual blocks."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, d_model))
        nn.init.normal_(self.seed, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.sample_ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.sample_norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, padding_mask: Tensor) -> Tensor:
        batch_size = h.shape[0]
        q = self.seed.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(
            q,
            h,
            h,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        sample = self.attention_norm(q + self.attention_dropout(attended))
        z_raw = self.sample_norm(sample + self.sample_ffn(sample))
        return z_raw.squeeze(1)


class MatchedPMADecoder(nn.Module):
    """Matched PMA ablation without the main decoder's two residual additions."""

    def __init__(self, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty(1, d_model))
        nn.init.normal_(self.seed, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, h: Tensor, padding_mask: Tensor) -> Tensor:
        batch_size = h.shape[0]
        q = self.seed.unsqueeze(0).expand(batch_size, -1, -1)
        attended, _ = self.cross_attention(
            q,
            h,
            h,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return self.output_norm(attended).squeeze(1)


class RelationOnlyModel(nn.Module):
    """Reduced relation-only Set Transformer.

    Token ids follow ``0=PAD``, ``1=reserved/UNK`` and
    ``real genus id = V3 var index + 2``.  There is deliberately no mask-token
    parameter, positional encoding, prior, register token, MLM head or metadata
    path in this module.
    """

    def __init__(self, config: Optional[RelationModelConfig] = None) -> None:
        super().__init__()
        self.config = config or RelationModelConfig()
        self.input_stem = RelationTokenStem(self.config)
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
        if self.config.decoder_kind == "main":
            self.decoder: nn.Module = LearnedSeedDecoder(
                d_model=self.config.d_model,
                nhead=self.config.decoder_heads,
                dim_feedforward=self.config.decoder_ffn_dim,
                dropout=self.config.dropout,
            )
        else:
            self.decoder = MatchedPMADecoder(
                d_model=self.config.d_model,
                nhead=self.config.decoder_heads,
                dropout=self.config.dropout,
            )

    def _validate_inputs(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        if genus_ids.ndim != 2:
            raise ValueError(f"genus_ids must have shape [B, L], got {tuple(genus_ids.shape)}")
        if genus_ids.dtype != torch.long:
            raise TypeError("genus_ids must use torch.long")
        if rclr.shape != genus_ids.shape:
            raise ValueError("rclr must have the same [B, L] shape as genus_ids")
        if not torch.is_floating_point(rclr):
            raise TypeError("rclr must be floating point")
        if genus_ids.shape[1] > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {genus_ids.shape[1]} exceeds max_seq_len={self.config.max_seq_len}"
            )
        if genus_ids.numel() == 0:
            raise ValueError("empty batch or zero-length sequence is not supported")
        if torch.any(genus_ids < 0) or torch.any(genus_ids >= self.config.vocab_size):
            raise ValueError("genus_ids contain an id outside the configured vocabulary")
        if not torch.isfinite(rclr).all():
            raise ValueError("rclr contains NaN or infinity")

        inferred_mask = genus_ids.eq(PAD_TOKEN_ID)
        if key_padding_mask is None:
            padding_mask = inferred_mask
        else:
            if key_padding_mask.shape != genus_ids.shape:
                raise ValueError("key_padding_mask must have the same [B, L] shape as genus_ids")
            if key_padding_mask.dtype != torch.bool:
                raise TypeError("key_padding_mask must use torch.bool")
            if not torch.equal(key_padding_mask, inferred_mask):
                raise ValueError("key_padding_mask must mark exactly the id-0 PAD positions")
            padding_mask = key_padding_mask
        if torch.any(padding_mask.all(dim=1)):
            raise ValueError("every sample must contain at least one non-PAD genus token")
        return padding_mask

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> RelationModelOutput:
        padding_mask = self._validate_inputs(genus_ids, rclr, key_padding_mask)
        h = self.input_stem(genus_ids, rclr, padding_mask)
        h = self.encoder(h, key_padding_mask=padding_mask, attn_bias=None)
        h = self.final_token_norm(h)
        h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        z_raw = self.decoder(h, padding_mask)
        # Relation mining/loss are explicitly outside AMP: the shared final
        # representation and its squared-L2 geometry are at least float32.
        z = F.normalize(z_raw.float(), p=2.0, dim=-1, eps=1e-12)
        return RelationModelOutput(
            z=z,
            z_raw=z_raw,
            token_embeddings=h,
            padding_mask=padding_mask,
        )
