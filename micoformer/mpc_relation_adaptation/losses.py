"""UniFrac ordinal shaping and explicit preservation primitives."""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class OrdinalLossConfig:
    """Per-step ordinal semantics inherited from the prior matched gate."""

    tie_fraction: float = 0.05
    temperature: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.tie_fraction < 1.0:
            raise ValueError("tie_fraction must be in [0,1)")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


class OrdinalLossOutput(NamedTuple):
    loss: Tensor
    valid_anchors: int
    comparisons: int
    mean_teacher_span: float


def _squared_distances(embedding: Tensor) -> Tensor:
    if embedding.ndim != 2 or embedding.shape[0] < 3:
        raise ValueError("embedding must be [B,D] with B>=3")
    if not torch.is_floating_point(embedding) or not bool(torch.isfinite(embedding).all()):
        raise ValueError("embedding must be finite floating point")
    value = embedding.float()
    return (value[:, None, :] - value[None, :, :]).square().sum(dim=-1)


def _validate_teacher(teacher: Tensor, batch_size: int) -> Tensor:
    if teacher.shape != (batch_size, batch_size) or not torch.is_floating_point(teacher):
        raise TypeError("teacher_distances must be floating point [B,B]")
    value = teacher.float()
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("teacher_distances must be finite and non-negative")
    if not torch.allclose(value, value.T, rtol=1e-5, atol=1e-6):
        raise ValueError("teacher_distances must be symmetric")
    if not torch.allclose(
        value.diagonal(), torch.zeros(batch_size, device=value.device), atol=1e-6
    ):
        raise ValueError("teacher_distances must have a zero diagonal")
    return value


def unifrac_ordinal_loss(
    embedding: Tensor,
    teacher_distances: Tensor,
    relation_block_ids: Tensor,
    config: OrdinalLossConfig | None = None,
) -> OrdinalLossOutput:
    """Match cross-block UniFrac order without regressing absolute distance values.

    The executable contract must define whether a block is a Study, Project, or
    another leakage-control unit.  Database labels are not accepted separately.
    """

    settings = config or OrdinalLossConfig()
    student = _squared_distances(embedding)
    if teacher_distances.device != student.device:
        raise ValueError("teacher_distances and embedding must share a device")
    teacher = _validate_teacher(teacher_distances, student.shape[0])
    if (
        relation_block_ids.ndim != 1
        or relation_block_ids.shape[0] != student.shape[0]
        or relation_block_ids.device != student.device
    ):
        raise ValueError("relation_block_ids must be aligned [B] on the embedding device")
    if relation_block_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("relation_block_ids must use an integer dtype")
    cross_block = relation_block_ids[:, None].ne(relation_block_ids[None, :])
    cross_block.fill_diagonal_(False)
    losses: list[Tensor] = []
    spans: list[Tensor] = []
    comparisons = 0
    quantiles = torch.tensor([0.1, 0.9], device=teacher.device)
    for anchor in range(student.shape[0]):
        candidates = torch.where(cross_block[anchor])[0]
        if candidates.numel() < 2:
            continue
        teacher_values = teacher[anchor, candidates]
        student_values = student[anchor, candidates]
        q10, q90 = torch.quantile(teacher_values, quantiles)
        span = (q90 - q10).clamp_min(torch.finfo(torch.float32).eps)
        teacher_delta = teacher_values[:, None] - teacher_values[None, :]
        student_delta = student_values[:, None] - student_values[None, :]
        upper = torch.triu(
            torch.ones_like(teacher_delta, dtype=torch.bool), diagonal=1
        )
        keep = upper & (teacher_delta.abs() > settings.tie_fraction * span)
        if not bool(keep.any()):
            continue
        signs = teacher_delta[keep].sign()
        weights = (teacher_delta[keep].abs() / span).clamp(max=1.0)
        values = F.softplus(-signs * student_delta[keep] / settings.temperature)
        losses.append((values * weights).sum() / weights.sum().clamp_min(1e-12))
        spans.append(span)
        comparisons += int(keep.sum().item())
    if not losses:
        raise RuntimeError("ordinal batch contains no valid cross-block comparisons")
    return OrdinalLossOutput(
        loss=torch.stack(losses).mean(),
        valid_anchors=len(losses),
        comparisons=comparisons,
        mean_teacher_span=float(torch.stack(spans).mean().detach().cpu()),
    )


def cosine_anchor_loss(h_unit: Tensor, z_unit: Tensor) -> Tensor:
    """Penalize adapter drift from the frozen checkpoint without labels."""

    if h_unit.shape != z_unit.shape or h_unit.ndim != 2:
        raise ValueError("h_unit and z_unit must be aligned [B,D]")
    if not bool(torch.isfinite(h_unit).all()) or not bool(torch.isfinite(z_unit).all()):
        raise ValueError("anchor embeddings must be finite")
    return (1.0 - F.cosine_similarity(
        h_unit.float(), z_unit.float(), dim=-1
    )).clamp_min(0.0).mean()
