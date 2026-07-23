"""Teacher-specific student-space mining for F1/F2.

Positive selection remains entirely teacher-defined.  Each teacher then uses
its own detached head geometry for next-farther negative selection while
protecting the other teacher's positive, matching the historical selector
except for the explicitly approved dual-head student geometry.
"""
from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from micoformer.relation_pretraining.mining import (
    MiningConfig,
    TeacherMiningResult,
    _comparison_tolerance,
    _project_difference_matrix,
    _validate_teacher_inputs,
    select_output_negative,
    select_positive,
)


@dataclass
class MultiHeadRelationMiningOutput:
    teachers: dict[str, TeacherMiningResult]
    student_squared_distances: dict[str, Tensor]

    @property
    def batch_size(self) -> int:
        first = next(iter(self.student_squared_distances.values()))
        return int(first.shape[0])

    def counters(self) -> dict[str, int]:
        counters: dict[str, int] = {}
        valid_total = 0
        skipped_total = 0
        for name, result in self.teachers.items():
            local = result.counters()
            counters.update({f"teacher/{name}/{key}": value for key, value in local.items()})
            valid_total += local["valid_relation"]
            skipped_total += local["skipped_relation"]
        total = self.batch_size * len(self.teachers)
        counters.update(
            {
                "all/teacher_anchor_total": total,
                "all/valid_relation": valid_total,
                "all/skipped_relation": skipped_total,
            }
        )
        if valid_total + skipped_total != total:
            raise RuntimeError("multi-head mining counter conservation failed")
        return counters


@torch.no_grad()
def mine_relations_by_teacher(
    teacher_z: Mapping[str, Tensor],
    teacher_distances: Mapping[str, Tensor],
    row_ids: Tensor | Sequence[int],
    project_ids: Tensor | Sequence[Hashable],
    *,
    teacher_validity: Mapping[str, Tensor] | None = None,
    config: MiningConfig | None = None,
) -> MultiHeadRelationMiningOutput:
    config = config or MiningConfig()
    if set(teacher_z) != set(teacher_distances) or not teacher_z:
        raise ValueError("teacher_z and teacher_distances must have identical non-empty keys")
    if len(teacher_z) > 2:
        raise ValueError("at most the two frozen teachers are supported")

    first = next(iter(teacher_z.values()))
    if first.ndim != 2 or first.shape[0] < 2:
        raise ValueError("teacher embeddings must have shape [B,D] with B >= 2")
    batch_size = int(first.shape[0])
    device = first.device
    for name, z in teacher_z.items():
        if z.ndim != 2 or z.shape[0] != batch_size or z.device != device:
            raise ValueError(f"teacher {name!r} embedding batch/device mismatch")
        if z.dtype not in {torch.float32, torch.float64} or not bool(torch.isfinite(z).all()):
            raise ValueError(f"teacher {name!r} embedding must be finite float32/float64")

    rows = torch.as_tensor(row_ids, dtype=torch.long, device=device)
    if rows.ndim != 1 or rows.numel() != batch_size:
        raise ValueError("row_ids must have shape [B]")
    if torch.unique(rows).numel() != batch_size:
        raise ValueError("row_ids must be unique within a physical batch")
    different_project = _project_difference_matrix(
        project_ids,
        batch_size=batch_size,
        device=device,
    )
    distances, validities = _validate_teacher_inputs(
        teacher_distances,
        teacher_validity,
        batch_size=batch_size,
        device=device,
    )
    student_distances = {
        name: (z.detach()[:, None, :] - z.detach()[None, :, :]).square().sum(dim=-1)
        for name, z in teacher_z.items()
    }

    positive_indices: dict[str, Tensor] = {}
    positive_ties: dict[str, Tensor] = {}
    for teacher_name in distances:
        indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        ties = torch.zeros(batch_size, dtype=torch.long, device=device)
        for anchor in range(batch_size):
            index, tie_count = select_positive(
                distances[teacher_name],
                validities[teacher_name],
                rows,
                different_project,
                anchor,
                config=config,
            )
            indices[anchor] = index
            ties[anchor] = tie_count
        positive_indices[teacher_name] = indices
        positive_ties[teacher_name] = ties

    teacher_names = list(distances)
    results: dict[str, TeacherMiningResult] = {}
    for teacher_name in teacher_names:
        distance = distances[teacher_name]
        validity = validities[teacher_name]
        student = student_distances[teacher_name]
        positive_index = positive_indices[teacher_name]
        negative_index = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        statuses: list[str] = []
        before_project = torch.zeros(batch_size, dtype=torch.long, device=device)
        after_project = torch.zeros(batch_size, dtype=torch.long, device=device)
        after_protection = torch.zeros(batch_size, dtype=torch.long, device=device)
        other_protected = torch.zeros(batch_size, dtype=torch.bool, device=device)
        nan_teacher = torch.full((batch_size,), float("nan"), dtype=distance.dtype, device=device)
        positive_teacher_distance = nan_teacher.clone()
        negative_teacher_distance = nan_teacher.clone()
        nan_student = torch.full((batch_size,), float("nan"), dtype=student.dtype, device=device)
        positive_student_distance = nan_student.clone()
        negative_student_distance = nan_student.clone()
        other_names = [name for name in teacher_names if name != teacher_name]

        for anchor in range(batch_size):
            positive = int(positive_index[anchor].item())
            if positive < 0:
                statuses.append("no_positive")
                continue
            teacher_positive = distance[anchor, positive]
            student_positive = student[anchor, positive]
            positive_teacher_distance[anchor] = teacher_positive
            positive_student_distance[anchor] = student_positive

            tolerance = _comparison_tolerance(
                distance[anchor],
                teacher_positive,
                rtol=config.teacher_rtol,
                atol=config.teacher_atol,
                scale_floor=config.teacher_scale_floor,
            )
            raw_far = validity[anchor] & (distance[anchor] > teacher_positive + tolerance)
            raw_far = raw_far.clone()
            raw_far[anchor] = False
            before_project[anchor] = raw_far.sum()
            eligible = raw_far & different_project[anchor]
            after_project[anchor] = eligible.sum()
            eligible = eligible.clone()
            eligible[positive] = False
            for other_name in other_names:
                other_positive = int(positive_indices[other_name][anchor].item())
                if other_positive >= 0:
                    if bool(eligible[other_positive]):
                        other_protected[anchor] = True
                    eligible[other_positive] = False
            after_protection[anchor] = eligible.sum()
            if not bool(eligible.any()):
                statuses.append("no_teacher_far")
                continue
            negative, status = select_output_negative(
                student_positive,
                student[anchor],
                eligible,
                rows,
                config=config,
            )
            statuses.append(status)
            if negative < 0:
                continue
            negative_index[anchor] = negative
            negative_teacher_distance[anchor] = distance[anchor, negative]
            negative_student_distance[anchor] = student[anchor, negative]

        result = TeacherMiningResult(
            name=teacher_name,
            positive_index=positive_index,
            negative_index=negative_index,
            status=tuple(statuses),
            positive_tie_count=positive_ties[teacher_name],
            teacher_far_before_project_count=before_project,
            teacher_far_after_project_count=after_project,
            teacher_far_after_protection_count=after_protection,
            other_positive_protected=other_protected,
            positive_teacher_distance=positive_teacher_distance,
            negative_teacher_distance=negative_teacher_distance,
            positive_student_distance=positive_student_distance,
            negative_student_distance=negative_student_distance,
        )
        result.counters()
        results[teacher_name] = result

    output = MultiHeadRelationMiningOutput(
        teachers=results,
        student_squared_distances=student_distances,
    )
    output.counters()
    return output
