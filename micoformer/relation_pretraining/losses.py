"""Relation-only squared-L2 triplet objective and explicit reductions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .mining import RelationMiningOutput, TeacherMiningResult


DEFAULT_STUDENT_ATOL = 1e-7


@dataclass
class TeacherLossStats:
    """Differentiable loss terms plus integer conservation diagnostics."""

    loss: Tensor
    loss_sum: Tensor
    valid_count: int
    skipped_count: int
    active_count: int
    inactive_count: int
    positive_squared_distance_sum: Tensor
    negative_squared_distance_sum: Tensor
    teacher_order_recovered_count: int

    def counters(self) -> dict[str, int]:
        counters = {
            "valid_anchor": self.valid_count,
            "skipped_anchor": self.skipped_count,
            "active_hinge": self.active_count,
            "inactive_hinge": self.inactive_count,
            "teacher_order_recovered": self.teacher_order_recovered_count,
        }
        if self.active_count + self.inactive_count != self.valid_count:
            raise RuntimeError("active/inactive hinge counter conservation failed")
        return counters


@dataclass
class RelationLossOutput:
    """Equal-priority teacher reduction for one physical batch."""

    loss: Tensor
    teacher_stats: dict[str, TeacherLossStats]
    teacher_weights: dict[str, float]
    has_relation_update: bool
    counters: dict[str, int]


def squared_l2_triplet_hinge(
    anchor: Tensor,
    positive: Tensor,
    negative: Tensor,
    *,
    margin: float = 0.10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-relation hinge and its two normalized-space distances."""

    if anchor.shape != positive.shape or anchor.shape != negative.shape:
        raise ValueError("anchor, positive and negative must have identical shapes")
    if anchor.ndim != 2:
        raise ValueError("triplet tensors must have shape [N, D]")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite non-negative number")
    positive_squared_distance = (anchor - positive).square().sum(dim=-1)
    negative_squared_distance = (anchor - negative).square().sum(dim=-1)
    hinge = F.relu(positive_squared_distance - negative_squared_distance + margin)
    return hinge, positive_squared_distance, negative_squared_distance


def _teacher_triplet_loss(
    z: Tensor,
    result: TeacherMiningResult,
    *,
    margin: float,
    student_atol: float,
) -> TeacherLossStats:
    batch_size = int(z.shape[0])
    if result.batch_size != batch_size:
        raise ValueError(
            f"teacher {result.name!r} mining batch {result.batch_size} does not match z batch {batch_size}"
        )
    valid = result.valid_mask.to(device=z.device)
    valid_count = int(valid.sum().item())
    skipped_count = batch_size - valid_count
    connected_zero = z.sum() * 0.0
    if valid_count == 0:
        stats = TeacherLossStats(
            loss=connected_zero,
            loss_sum=connected_zero,
            valid_count=0,
            skipped_count=skipped_count,
            active_count=0,
            inactive_count=0,
            positive_squared_distance_sum=connected_zero,
            negative_squared_distance_sum=connected_zero,
            teacher_order_recovered_count=0,
        )
        stats.counters()
        return stats

    anchors = torch.nonzero(valid, as_tuple=False).flatten()
    positives = result.positive_index.to(device=z.device)[anchors]
    negatives = result.negative_index.to(device=z.device)[anchors]
    hinge, positive_distance, negative_distance = squared_l2_triplet_hinge(
        z[anchors],
        z[positives],
        z[negatives],
        margin=margin,
    )
    loss_sum = hinge.sum()
    active_count = int((hinge.detach() > 0).sum().item())
    # Keep the diagnostic on the same strict-order boundary as mining.  A
    # numerically tied pair must not be reported as recovered merely because
    # float32 produced a tiny positive delta.
    teacher_order_recovered_count = int(
        (
            negative_distance.detach()
            > positive_distance.detach() + student_atol
        ).sum().item()
    )
    stats = TeacherLossStats(
        loss=loss_sum / valid_count,
        loss_sum=loss_sum,
        valid_count=valid_count,
        skipped_count=skipped_count,
        active_count=active_count,
        inactive_count=valid_count - active_count,
        positive_squared_distance_sum=positive_distance.sum(),
        negative_squared_distance_sum=negative_distance.sum(),
        teacher_order_recovered_count=teacher_order_recovered_count,
    )
    stats.counters()
    return stats


def relation_triplet_loss(
    z: Tensor,
    mining: RelationMiningOutput | Mapping[str, TeacherMiningResult],
    *,
    margin: float = 0.10,
    student_atol: float = DEFAULT_STUDENT_ATOL,
) -> RelationLossOutput:
    """Compute the locked valid-anchor and equal-priority teacher reduction.

    Each teacher first takes a mean over only its valid triplets.  A valid
    zero-hinge relation remains in that denominator.  If both teachers are
    present their means receive 0.5/0.5 weight; if only one has any valid
    relation it is renormalized to 1.0; if none is present the returned zero is
    graph-connected to ``z`` and ``has_relation_update`` is false.
    """

    if z.ndim != 2:
        raise ValueError("z must have shape [B, D]")
    if z.dtype not in {torch.float32, torch.float64} or not torch.isfinite(z).all():
        raise ValueError("z must be finite and use at least float32 precision")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("margin must be a finite non-negative number")
    if not math.isfinite(student_atol) or student_atol < 0:
        raise ValueError("student_atol must be a finite non-negative number")
    teachers = mining.teachers if isinstance(mining, RelationMiningOutput) else dict(mining)
    if not teachers:
        raise ValueError("at least one teacher mining result is required")
    if len(teachers) > 2:
        raise ValueError("the v1 loss supports at most the two locked teachers")

    teacher_stats = {
        name: _teacher_triplet_loss(
            z,
            result,
            margin=margin,
            student_atol=student_atol,
        )
        for name, result in teachers.items()
    }
    present = [name for name, stats in teacher_stats.items() if stats.valid_count > 0]
    teacher_weights = {name: 0.0 for name in teacher_stats}
    if present:
        weight = 1.0 / len(present)
        for name in present:
            teacher_weights[name] = weight
        loss = sum(
            (teacher_stats[name].loss * teacher_weights[name] for name in present),
            start=z.sum() * 0.0,
        )
    else:
        loss = z.sum() * 0.0

    counters: dict[str, int] = {}
    valid_total = 0
    skipped_total = 0
    active_total = 0
    inactive_total = 0
    for name, stats in teacher_stats.items():
        local = stats.counters()
        counters.update({f"teacher/{name}/{key}": value for key, value in local.items()})
        valid_total += stats.valid_count
        skipped_total += stats.skipped_count
        active_total += stats.active_count
        inactive_total += stats.inactive_count
    teacher_anchor_total = int(z.shape[0]) * len(teacher_stats)
    counters.update(
        {
            "all/teacher_anchor_total": teacher_anchor_total,
            "all/valid_anchor": valid_total,
            "all/skipped_anchor": skipped_total,
            "all/active_hinge": active_total,
            "all/inactive_hinge": inactive_total,
            "all/present_teacher": len(present),
            "all/missing_teacher": len(teacher_stats) - len(present),
            "all/relation_update": int(bool(present)),
        }
    )
    if valid_total + skipped_total != teacher_anchor_total:
        raise RuntimeError("loss valid/skip counter conservation failed")
    if active_total + inactive_total != valid_total:
        raise RuntimeError("loss active/inactive counter conservation failed")

    return RelationLossOutput(
        loss=loss,
        teacher_stats=teacher_stats,
        teacher_weights=teacher_weights,
        has_relation_update=bool(present),
        counters=counters,
    )
