"""Deterministic two-teacher relation mining inside a physical batch.

Teacher distances and Project_ID are selector-only inputs.  They are never
forwarded to the student model.  Mining is deliberately non-differentiable;
the selected identities are used by :mod:`losses` to recompute differentiable
student distances.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, Sequence

import torch
from torch import Tensor


STATUS_NAMES = (
    "no_positive",
    "no_teacher_far",
    "valid_next",
    "no_next",
    "valid_fallback",
)
VALID_STATUS_NAMES = frozenset({"valid_next", "valid_fallback"})
NO_NEXT_MODES = frozenset({"skip", "closest_radius_inside"})


@dataclass(frozen=True)
class MiningConfig:
    """Numerical and no-next policy for relation selection.

    Tolerances are explicit and should be serialized in every run manifest.
    The defaults implement the frozen runtime comparisons exactly: teacher
    tolerance is ``max(1e-12, 1e-10 * max(abs(a), abs(b), 1))`` and student
    squared-L2 comparisons use absolute tolerance ``1e-7``.
    """

    no_next_mode: str = "skip"
    teacher_rtol: float = 1e-10
    teacher_atol: float = 1e-12
    teacher_scale_floor: float = 1.0
    student_rtol: float = 0.0
    student_atol: float = 1e-7

    def __post_init__(self) -> None:
        if self.no_next_mode not in NO_NEXT_MODES:
            raise ValueError(
                f"no_next_mode must be one of {sorted(NO_NEXT_MODES)}, got {self.no_next_mode!r}"
            )
        for name in (
            "teacher_rtol",
            "teacher_atol",
            "teacher_scale_floor",
            "student_rtol",
            "student_atol",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


@dataclass
class TeacherMiningResult:
    """Per-anchor mining identities and diagnostics for one teacher."""

    name: str
    positive_index: Tensor
    negative_index: Tensor
    status: tuple[str, ...]
    positive_tie_count: Tensor
    teacher_far_before_project_count: Tensor
    teacher_far_after_project_count: Tensor
    teacher_far_after_protection_count: Tensor
    other_positive_protected: Tensor
    positive_teacher_distance: Tensor
    negative_teacher_distance: Tensor
    positive_student_distance: Tensor
    negative_student_distance: Tensor

    def __post_init__(self) -> None:
        batch_size = int(self.positive_index.numel())
        if self.positive_index.ndim != 1 or self.negative_index.shape != self.positive_index.shape:
            raise ValueError("positive_index and negative_index must both have shape [B]")
        if len(self.status) != batch_size:
            raise ValueError("status must contain exactly one entry per teacher-anchor")
        unknown = set(self.status).difference(STATUS_NAMES)
        if unknown:
            raise ValueError(f"unknown mining statuses: {sorted(unknown)}")
        tensor_fields = (
            self.positive_tie_count,
            self.teacher_far_before_project_count,
            self.teacher_far_after_project_count,
            self.teacher_far_after_protection_count,
            self.other_positive_protected,
            self.positive_teacher_distance,
            self.negative_teacher_distance,
            self.positive_student_distance,
            self.negative_student_distance,
        )
        if any(t.ndim != 1 or t.numel() != batch_size for t in tensor_fields):
            raise ValueError("all per-anchor diagnostic tensors must have shape [B]")
        valid_from_status = torch.tensor(
            [s in VALID_STATUS_NAMES for s in self.status],
            dtype=torch.bool,
            device=self.negative_index.device,
        )
        valid_from_indices = (self.positive_index >= 0) & (self.negative_index >= 0)
        if not torch.equal(valid_from_status, valid_from_indices):
            raise ValueError("valid status and selected positive/negative indices disagree")

    @property
    def batch_size(self) -> int:
        return int(self.positive_index.numel())

    @property
    def valid_mask(self) -> Tensor:
        return (self.positive_index >= 0) & (self.negative_index >= 0)

    def counters(self) -> dict[str, int]:
        status_counter = Counter(self.status)
        counters = {f"status/{name}": int(status_counter.get(name, 0)) for name in STATUS_NAMES}
        counters.update(
            {
                "anchor_total": self.batch_size,
                "status_total": int(sum(status_counter.values())),
                "valid_relation": int(self.valid_mask.sum().item()),
                "skipped_relation": int((~self.valid_mask).sum().item()),
                "positive_tie_anchor": int((self.positive_tie_count > 1).sum().item()),
                "teacher_far_before_project_total": int(
                    self.teacher_far_before_project_count.sum().item()
                ),
                "teacher_far_after_project_total": int(
                    self.teacher_far_after_project_count.sum().item()
                ),
                "teacher_far_after_protection_total": int(
                    self.teacher_far_after_protection_count.sum().item()
                ),
                "other_positive_protected": int(self.other_positive_protected.sum().item()),
            }
        )
        if counters["status_total"] != counters["anchor_total"]:
            raise RuntimeError("status counter conservation failed")
        if counters["valid_relation"] + counters["skipped_relation"] != counters["anchor_total"]:
            raise RuntimeError("valid/skip counter conservation failed")
        return counters


@dataclass
class RelationMiningOutput:
    """Complete mining output for one physical batch."""

    teachers: dict[str, TeacherMiningResult]
    student_squared_distances: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.student_squared_distances.shape[0])

    def counters(self) -> dict[str, int]:
        counters: dict[str, int] = {}
        aggregate_status = Counter()
        valid_total = 0
        skipped_total = 0
        for name, result in self.teachers.items():
            teacher_counters = result.counters()
            counters.update({f"teacher/{name}/{key}": value for key, value in teacher_counters.items()})
            aggregate_status.update(result.status)
            valid_total += teacher_counters["valid_relation"]
            skipped_total += teacher_counters["skipped_relation"]
        teacher_anchor_total = self.batch_size * len(self.teachers)
        counters["all/teacher_anchor_total"] = teacher_anchor_total
        for status in STATUS_NAMES:
            counters[f"all/status/{status}"] = int(aggregate_status.get(status, 0))
        counters["all/status_total"] = int(sum(aggregate_status.values()))
        counters["all/valid_relation"] = valid_total
        counters["all/skipped_relation"] = skipped_total
        if counters["all/status_total"] != teacher_anchor_total:
            raise RuntimeError("global status counter conservation failed")
        if valid_total + skipped_total != teacher_anchor_total:
            raise RuntimeError("global valid/skip counter conservation failed")
        return counters


def _comparison_tolerance(
    left: Tensor,
    right: Tensor,
    *,
    rtol: float,
    atol: float,
    scale_floor: float = 0.0,
) -> Tensor:
    """Frozen max-form tolerance for scalar or elementwise comparisons."""

    floor = torch.as_tensor(scale_floor, dtype=left.dtype, device=left.device)
    scale = torch.maximum(torch.maximum(left.abs(), right.abs()), floor)
    relative = rtol * scale
    absolute = torch.as_tensor(atol, dtype=left.dtype, device=left.device)
    return torch.maximum(relative, absolute)


def _canonical_min_index(
    values: Tensor,
    eligible: Tensor,
    row_ids: Tensor,
    *,
    rtol: float,
    atol: float,
    scale_floor: float = 0.0,
) -> tuple[int, int]:
    """Return min-value candidate, breaking numerical ties by canonical row id."""

    if not bool(eligible.any()):
        return -1, 0
    best = values[eligible].min()
    tied = eligible & (
        (values - best).abs()
        <= _comparison_tolerance(
            values,
            best,
            rtol=rtol,
            atol=atol,
            scale_floor=scale_floor,
        )
    )
    tie_indices = torch.nonzero(tied, as_tuple=False).flatten()
    tied_rows = row_ids[tie_indices]
    chosen_position = int(torch.argmin(tied_rows).item())
    return int(tie_indices[chosen_position].item()), int(tie_indices.numel())


def select_positive(
    teacher_distances: Tensor,
    valid_pairs: Tensor,
    row_ids: Tensor,
    different_project: Tensor,
    anchor_index: int,
    *,
    config: MiningConfig,
) -> tuple[int, int]:
    """Select a different-Project teacher argmin with row-id tie-breaking."""

    eligible = valid_pairs[anchor_index] & different_project[anchor_index]
    eligible = eligible.clone()
    eligible[anchor_index] = False
    return _canonical_min_index(
        teacher_distances[anchor_index],
        eligible,
        row_ids,
        rtol=config.teacher_rtol,
        atol=config.teacher_atol,
        scale_floor=config.teacher_scale_floor,
    )


def select_output_negative(
    positive_student_distance: Tensor,
    candidate_student_distances: Tensor,
    teacher_eligible: Tensor,
    row_ids: Tensor,
    *,
    config: MiningConfig,
) -> tuple[int, str]:
    """Select output-space next-farther, or the configured no-next policy."""

    eligible = teacher_eligible & torch.isfinite(candidate_student_distances)
    if not bool(eligible.any()):
        return -1, "no_teacher_far"
    radius_tolerance = _comparison_tolerance(
        candidate_student_distances,
        positive_student_distance,
        rtol=config.student_rtol,
        atol=config.student_atol,
    )
    outside = eligible & (
        candidate_student_distances > positive_student_distance + radius_tolerance
    )
    if bool(outside.any()):
        index, _ = _canonical_min_index(
            candidate_student_distances,
            outside,
            row_ids,
            rtol=config.student_rtol,
            atol=config.student_atol,
        )
        return index, "valid_next"
    if config.no_next_mode == "skip":
        return -1, "no_next"
    radius_delta = (candidate_student_distances - positive_student_distance).abs()
    index, _ = _canonical_min_index(
        radius_delta,
        eligible,
        row_ids,
        rtol=config.student_rtol,
        atol=config.student_atol,
    )
    return index, "valid_fallback"


def _project_difference_matrix(
    project_ids: Tensor | Sequence[Hashable],
    *,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if isinstance(project_ids, Tensor):
        if project_ids.ndim != 1 or project_ids.numel() != batch_size:
            raise ValueError("project_ids tensor must have shape [B]")
        if torch.is_floating_point(project_ids) and not torch.isfinite(project_ids).all():
            raise ValueError("project_ids contains NaN or infinity")
        projects = project_ids.to(device=device)
        return projects[:, None] != projects[None, :]

    projects_list = list(project_ids)
    if len(projects_list) != batch_size:
        raise ValueError("project_ids sequence must have length B")
    for project in projects_list:
        if project is None or (isinstance(project, float) and math.isnan(project)):
            raise ValueError("project_ids contains a missing value")
    return torch.tensor(
        [[left != right for right in projects_list] for left in projects_list],
        dtype=torch.bool,
        device=device,
    )


def _validate_teacher_inputs(
    teacher_distances: Mapping[str, Tensor],
    teacher_validity: Optional[Mapping[str, Tensor]],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    if not teacher_distances:
        raise ValueError("at least one teacher distance matrix is required")
    if len(teacher_distances) > 2:
        raise ValueError("the v1 selector supports at most the two locked teachers")
    if teacher_validity is not None and set(teacher_validity) != set(teacher_distances):
        raise ValueError("teacher_validity keys must exactly match teacher_distances keys")

    distances: dict[str, Tensor] = {}
    validities: dict[str, Tensor] = {}
    for name, matrix in teacher_distances.items():
        if not isinstance(name, str) or not name:
            raise ValueError("teacher names must be non-empty strings")
        if matrix.shape != (batch_size, batch_size):
            raise ValueError(f"teacher {name!r} must have shape [B, B]")
        if matrix.dtype != torch.float64:
            raise TypeError(f"teacher {name!r} distances must use frozen float64 precision")
        distance = matrix.detach().to(device=device)
        validity = torch.isfinite(distance)
        if teacher_validity is not None:
            supplied = teacher_validity[name]
            if supplied.shape != (batch_size, batch_size) or supplied.dtype != torch.bool:
                raise ValueError(f"teacher {name!r} validity must be bool [B, B]")
            validity &= supplied.detach().to(device=device)
        if bool((distance[validity] < 0).any()):
            raise ValueError(f"teacher {name!r} contains a negative valid distance")
        distances[name] = distance
        validities[name] = validity
    return distances, validities


@torch.no_grad()
def mine_relations(
    z: Tensor,
    teacher_distances: Mapping[str, Tensor],
    row_ids: Tensor | Sequence[int],
    project_ids: Tensor | Sequence[Hashable],
    *,
    teacher_validity: Optional[Mapping[str, Tensor]] = None,
    config: Optional[MiningConfig] = None,
) -> RelationMiningOutput:
    """Mine independent relations for every teacher-anchor in one batch.

    Args:
        z: Final shared sample embedding ``[B, D]``.  Mining reads a detached
            squared-L2 matrix; the loss later recomputes selected distances.
        teacher_distances: One or two named ``[B, B]`` teacher matrices.
            Invalid pairs must be non-finite or false in ``teacher_validity``.
        row_ids: Unique canonical V3 row identities used for every tie-break.
        project_ids: Project_ID values or pre-encoded integer project ids.
        teacher_validity: Optional explicit pair-validity masks.
        config: Explicit tolerances and ``skip``/``closest_radius_inside`` mode.
    """

    config = config or MiningConfig()
    if z.ndim != 2 or z.shape[0] < 2:
        raise ValueError("z must have shape [B, D] with B >= 2")
    if z.dtype not in {torch.float32, torch.float64} or not torch.isfinite(z).all():
        raise ValueError("z must be finite and use at least float32 precision")
    batch_size = int(z.shape[0])
    device = z.device
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

    delta = z.detach()[:, None, :] - z.detach()[None, :, :]
    student_squared_distances = delta.square().sum(dim=-1)

    positive_indices: dict[str, Tensor] = {}
    positive_ties: dict[str, Tensor] = {}
    for teacher_name in distances:
        indices = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        tie_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        for anchor in range(batch_size):
            index, ties = select_positive(
                distances[teacher_name],
                validities[teacher_name],
                rows,
                different_project,
                anchor,
                config=config,
            )
            indices[anchor] = index
            tie_counts[anchor] = ties
        positive_indices[teacher_name] = indices
        positive_ties[teacher_name] = tie_counts

    teacher_names = list(distances)
    results: dict[str, TeacherMiningResult] = {}
    for teacher_name in teacher_names:
        distance = distances[teacher_name]
        validity = validities[teacher_name]
        positive_index = positive_indices[teacher_name]
        negative_index = torch.full((batch_size,), -1, dtype=torch.long, device=device)
        statuses: list[str] = []
        before_project = torch.zeros(batch_size, dtype=torch.long, device=device)
        after_project = torch.zeros(batch_size, dtype=torch.long, device=device)
        after_protection = torch.zeros(batch_size, dtype=torch.long, device=device)
        other_protected = torch.zeros(batch_size, dtype=torch.bool, device=device)
        nan_template = torch.full(
            (batch_size,),
            float("nan"),
            dtype=distance.dtype,
            device=device,
        )
        positive_teacher_distance = nan_template.clone()
        negative_teacher_distance = nan_template.clone()
        positive_student_distance = torch.full(
            (batch_size,),
            float("nan"),
            dtype=student_squared_distances.dtype,
            device=device,
        )
        negative_student_distance = positive_student_distance.clone()

        other_names = [name for name in teacher_names if name != teacher_name]
        for anchor in range(batch_size):
            positive = int(positive_index[anchor].item())
            if positive < 0:
                statuses.append("no_positive")
                continue
            teacher_positive_distance = distance[anchor, positive]
            student_positive_distance = student_squared_distances[anchor, positive]
            positive_teacher_distance[anchor] = teacher_positive_distance
            positive_student_distance[anchor] = student_positive_distance

            far_tolerance = _comparison_tolerance(
                distance[anchor],
                teacher_positive_distance,
                rtol=config.teacher_rtol,
                atol=config.teacher_atol,
                scale_floor=config.teacher_scale_floor,
            )
            raw_far = validity[anchor] & (
                distance[anchor] > teacher_positive_distance + far_tolerance
            )
            raw_far = raw_far.clone()
            raw_far[anchor] = False
            before_project[anchor] = raw_far.sum()
            eligible = raw_far & different_project[anchor]
            after_project[anchor] = eligible.sum()

            # Protect this teacher's positive and every available other-teacher
            # positive.  v1 has exactly two teachers, but this remains safe for
            # one-teacher unit/smoke runs.
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
                student_positive_distance,
                student_squared_distances[anchor],
                eligible,
                rows,
                config=config,
            )
            statuses.append(status)
            if negative < 0:
                continue
            if not bool(different_project[anchor, negative]):
                raise RuntimeError("selector returned a same-Project negative")
            negative_index[anchor] = negative
            negative_teacher_distance[anchor] = distance[anchor, negative]
            negative_student_distance[anchor] = student_squared_distances[anchor, negative]

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
        # Fail immediately rather than allowing incomplete status accounting to
        # leak into training metrics.
        result.counters()
        results[teacher_name] = result

    output = RelationMiningOutput(
        teachers=results,
        student_squared_distances=student_squared_distances,
    )
    output.counters()
    return output
