"""Frozen MPC encoder and its training-only Query heads.

The deliverable sample representation is the valid-token mean from an unmasked
forward.  The MLM and Query heads exist only to shape the shared encoder.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from micoformer.fullscale_relation_pretraining.model import (
    FullscaleRelationModel,
    FullscaleRelationModelConfig,
    masked_token_mean,
)


@dataclass(frozen=True)
class MPCModelConfig:
    vocab_size: int = 8_116
    d_model: int = 512
    rclr_hidden_dim: int = 64
    num_layers: int = 12
    encoder_heads: int = 16
    encoder_ffn_dim: int = 2_048
    mlm_head_hidden_dim: int = 64
    dropout: float = 0.1
    max_seq_len: int = 512
    grad_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size != 8_116:
            raise ValueError("the first full-data MPC contract requires vocab_size=8116")
        if (
            self.d_model,
            self.num_layers,
            self.encoder_heads,
            self.encoder_ffn_dim,
            self.max_seq_len,
        ) != (512, 12, 16, 2_048, 512):
            raise ValueError("the first full-data MPC backbone contract drifted")
        if self.dropout != 0.1 or self.grad_checkpointing:
            raise ValueError("the first full-data MPC run requires dropout=.1 and no checkpointing")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class QueryOutput:
    positive_presence: Tensor
    negative_presence: Tensor
    positive_continuous: Tensor
    positive_target: Tensor
    positive_count: int


class MPCForwardOutput(NamedTuple):
    """DDP-visible losses and counts.

    ``DistributedDataParallel(static_graph=True)`` routes output tensors through
    an autograd sink.  A ``NamedTuple`` is a pytree-recognized tuple, whereas a
    plain dataclass is treated as an opaque leaf by PyTorch 2.6's DDP pytree
    helper and silently bypasses gradient reduction.
    """

    mlm_loss: Tensor
    continuous_loss: Tensor
    presence_loss: Tensor
    mlm_count: int
    continuous_count: int
    presence_count: int


class ConcatQueryHead(nn.Module):
    """The frozen first-run concat-MLP scorer for sample/candidate pairs."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.presence = nn.Linear(d_model, 1)
        self.continuous = nn.Linear(d_model, 1)

    def forward(self, sample: Tensor, candidate: Tensor) -> tuple[Tensor, Tensor]:
        if sample.ndim != 2 or sample.shape != candidate.shape:
            raise ValueError("sample/candidate must be aligned [N,D]")
        features = self.trunk(torch.cat((sample, candidate), dim=-1))
        return self.presence(features).squeeze(-1), self.continuous(features).squeeze(-1)


class MPCPretrainingModel(nn.Module):
    """Learned-ID encoder plus fixed PHY+PROT candidate Query supervision."""

    def __init__(
        self,
        candidate_table: Tensor,
        config: MPCModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MPCModelConfig()
        backbone_config = FullscaleRelationModelConfig(
            arm="f1_shared",
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            rclr_hidden_dim=self.config.rclr_hidden_dim,
            num_layers=self.config.num_layers,
            encoder_heads=self.config.encoder_heads,
            encoder_ffn_dim=self.config.encoder_ffn_dim,
            mlm_head_hidden_dim=self.config.mlm_head_hidden_dim,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_seq_len,
            grad_checkpointing=self.config.grad_checkpointing,
        )
        self.backbone = FullscaleRelationModel(backbone_config)
        # The relation projector is inherited from the reusable backbone but is not
        # part of MPC.  Removing it is required for DDP static-graph correctness.
        self.backbone.shared_teacher_head = None
        if candidate_table.shape != (self.config.vocab_size, self.config.d_model):
            raise ValueError("fixed candidate table must be [8116,512]")
        if not bool(torch.isfinite(candidate_table).all()):
            raise ValueError("fixed candidate table contains non-finite values")
        self.register_buffer(
            "candidate_table",
            candidate_table.detach().float().clone(),
            persistent=False,
        )
        self.query_head = ConcatQueryHead(self.config.d_model)

    def shared_encoder_parameters(self) -> tuple[nn.Parameter, ...]:
        """Parameters used to define the frozen initialization gradient dose."""
        modules = (
            self.backbone.input_stem,
            self.backbone.encoder,
            self.backbone.final_token_norm,
        )
        return tuple(
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        )

    @staticmethod
    def _recenter_visible_rclr(
        rclr: Tensor,
        padding_mask: Tensor,
        query_mask: Tensor,
        mlm_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if not (rclr.shape == padding_mask.shape == query_mask.shape == mlm_mask.shape):
            raise ValueError("rclr and masks must share [B,L]")
        if any(mask.dtype != torch.bool for mask in (padding_mask, query_mask, mlm_mask)):
            raise TypeError("padding/query/mlm masks must be bool")
        if bool((query_mask & mlm_mask).any()):
            raise ValueError("Query and MLM masks may not overlap")
        if bool(((query_mask | mlm_mask) & padding_mask).any()):
            raise ValueError("a corruption mask selected padding")
        encoder_padding = padding_mask | query_mask
        visible_numerical = ~(encoder_padding | mlm_mask)
        counts = visible_numerical.sum(1, keepdim=True)
        if bool((counts <= 0).any()):
            raise ValueError("a sample lost all numerical context")
        means = (
            (rclr * visible_numerical.to(rclr.dtype)).sum(1, keepdim=True)
            / counts.to(rclr.dtype)
        )
        centered = (rclr - means).masked_fill(~visible_numerical, 0.0)
        return centered, encoder_padding

    def _mlm_loss(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
        mlm_mask: Tensor,
    ) -> tuple[Tensor, int]:
        no_query = torch.zeros_like(mlm_mask)
        centered, encoder_padding = self._recenter_visible_rclr(
            rclr, padding_mask, no_query, mlm_mask
        )
        token_embeddings = self.backbone._encode(
            genus_ids,
            centered,
            encoder_padding,
            abundance_mask=mlm_mask,
        )
        prediction = self.backbone.mlm_head(token_embeddings)
        count = int(mlm_mask.sum().item())
        if count <= 0:
            raise RuntimeError("MLM branch has no targets")
        loss = F.smooth_l1_loss(
            prediction[mlm_mask].float(),
            rclr[mlm_mask].float(),
            beta=1.0,
        )
        return loss, count

    def _query_output(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
        query_mask: Tensor,
        negative_genus_ids: Tensor,
    ) -> QueryOutput:
        no_mlm = torch.zeros_like(query_mask)
        centered, encoder_padding = self._recenter_visible_rclr(
            rclr, padding_mask, query_mask, no_mlm
        )
        token_embeddings = self.backbone._encode(
            genus_ids,
            centered,
            encoder_padding,
            abundance_mask=no_mlm,
        )
        sample_h = masked_token_mean(token_embeddings, encoder_padding)
        sample_indices = torch.where(query_mask)[0]
        positive_ids = genus_ids[query_mask]
        negative_ids = negative_genus_ids[query_mask]
        if bool((positive_ids < 2).any()) or bool((negative_ids < 2).any()):
            raise ValueError("Query candidates must be real genus IDs")
        positive_candidate = F.embedding(positive_ids, self.candidate_table, padding_idx=0)
        negative_candidate = F.embedding(negative_ids, self.candidate_table, padding_idx=0)
        positive_presence, positive_continuous = self.query_head(
            sample_h[sample_indices], positive_candidate
        )
        negative_presence, _ = self.query_head(sample_h[sample_indices], negative_candidate)
        return QueryOutput(
            positive_presence=positive_presence,
            negative_presence=negative_presence,
            positive_continuous=positive_continuous,
            positive_target=rclr[query_mask].float(),
            positive_count=int(positive_ids.numel()),
        )

    @staticmethod
    def _query_losses(output: QueryOutput) -> tuple[Tensor, Tensor, int, int]:
        if output.positive_count <= 0:
            raise RuntimeError("Query branch has no targets")
        continuous = F.smooth_l1_loss(
            output.positive_continuous.float(),
            output.positive_target.float(),
            beta=1.0,
        )
        logits = torch.cat((output.positive_presence, output.negative_presence)).float()
        labels = torch.cat((
            torch.ones_like(output.positive_presence),
            torch.zeros_like(output.negative_presence),
        )).float()
        presence = F.binary_cross_entropy_with_logits(logits, labels)
        return continuous, presence, output.positive_count, int(labels.numel())

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
        query_mask: Tensor,
        mlm_mask: Tensor,
        negative_genus_ids: Tensor,
    ) -> MPCForwardOutput:
        padding = self.backbone._validate_inputs(genus_ids, rclr, padding_mask)
        if negative_genus_ids.shape != genus_ids.shape or negative_genus_ids.dtype != torch.long:
            raise TypeError("negative_genus_ids must be long [B,L]")
        mlm_loss, mlm_count = self._mlm_loss(genus_ids, rclr, padding, mlm_mask)
        query = self._query_output(
            genus_ids, rclr, padding, query_mask, negative_genus_ids
        )
        continuous, presence, continuous_count, presence_count = self._query_losses(query)
        return MPCForwardOutput(
            mlm_loss=mlm_loss,
            continuous_loss=continuous,
            presence_loss=presence,
            mlm_count=mlm_count,
            continuous_count=continuous_count,
            presence_count=presence_count,
        )

    def component_loss(
        self,
        component: str,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
        query_mask: Tensor,
        mlm_mask: Tensor,
        negative_genus_ids: Tensor,
    ) -> Tensor:
        """Run only the branch needed by initialization calibration."""
        padding = self.backbone._validate_inputs(genus_ids, rclr, padding_mask)
        if component == "mlm":
            return self._mlm_loss(genus_ids, rclr, padding, mlm_mask)[0]
        query = self._query_output(
            genus_ids, rclr, padding, query_mask, negative_genus_ids
        )
        continuous, presence, _, _ = self._query_losses(query)
        if component == "continuous":
            return continuous
        if component == "presence":
            return presence
        raise ValueError("component must be mlm, continuous, or presence")

    def unmasked_representations(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        padding = self.backbone._validate_inputs(genus_ids, rclr, padding_mask)
        token_embeddings = self.backbone._encode(
            genus_ids, rclr, padding, abundance_mask=None
        )
        return token_embeddings, masked_token_mean(token_embeddings, padding)
