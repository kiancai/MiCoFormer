"""Equal-priority relation loss for shared or teacher-specific head outputs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor

from micoformer.relation_pretraining.losses import (
    DEFAULT_STUDENT_ATOL,
    TeacherLossStats,
    relation_triplet_loss,
)
from micoformer.relation_pretraining.mining import TeacherMiningResult

from .mining import MultiHeadRelationMiningOutput


@dataclass
class MultiHeadRelationLossOutput:
    loss: Tensor
    teacher_stats: dict[str, TeacherLossStats]
    teacher_losses: dict[str, Tensor]
    teacher_weights: dict[str, float]
    has_relation_update: bool
    counters: dict[str, int]


def multi_head_relation_triplet_loss(
    teacher_z: Mapping[str, Tensor],
    mining: MultiHeadRelationMiningOutput | Mapping[str, TeacherMiningResult],
    *,
    margin: float = 0.10,
    student_atol: float = DEFAULT_STUDENT_ATOL,
) -> MultiHeadRelationLossOutput:
    results = mining.teachers if isinstance(mining, MultiHeadRelationMiningOutput) else dict(mining)
    if set(teacher_z) != set(results) or not teacher_z:
        raise ValueError("teacher_z and mining results must have identical non-empty keys")
    if len(teacher_z) > 2:
        raise ValueError("at most the two frozen teachers are supported")

    single = {
        name: relation_triplet_loss(
            teacher_z[name],
            {name: results[name]},
            margin=margin,
            student_atol=student_atol,
        )
        for name in teacher_z
    }
    stats = {name: output.teacher_stats[name] for name, output in single.items()}
    teacher_losses = {name: output.loss for name, output in single.items()}
    present = [name for name, item in stats.items() if item.valid_count > 0]
    weights = {name: 0.0 for name in teacher_z}
    connected_zero = sum((z.sum() * 0.0 for z in teacher_z.values()))
    loss = connected_zero
    if present:
        weight = 1.0 / len(present)
        for name in present:
            weights[name] = weight
            loss = loss + teacher_losses[name] * weight

    counters: dict[str, int] = {}
    valid_total = 0
    skipped_total = 0
    active_total = 0
    inactive_total = 0
    for name, item in stats.items():
        local = item.counters()
        counters.update({f"teacher/{name}/{key}": value for key, value in local.items()})
        valid_total += item.valid_count
        skipped_total += item.skipped_count
        active_total += item.active_count
        inactive_total += item.inactive_count
    batch_size = int(next(iter(teacher_z.values())).shape[0])
    total = batch_size * len(teacher_z)
    counters.update(
        {
            "all/teacher_anchor_total": total,
            "all/valid_anchor": valid_total,
            "all/skipped_anchor": skipped_total,
            "all/active_hinge": active_total,
            "all/inactive_hinge": inactive_total,
            "all/present_teacher": len(present),
            "all/missing_teacher": len(teacher_z) - len(present),
            "all/relation_update": int(bool(present)),
        }
    )
    if valid_total + skipped_total != total or active_total + inactive_total != valid_total:
        raise RuntimeError("multi-head relation loss counter conservation failed")
    return MultiHeadRelationLossOutput(
        loss=loss,
        teacher_stats=stats,
        teacher_losses=teacher_losses,
        teacher_weights=weights,
        has_relation_update=bool(present),
        counters=counters,
    )
