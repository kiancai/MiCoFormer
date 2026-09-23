"""Soft-neighborhood relation loss used by the fixed-PP continuation pilot."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class SoftNeighborhoodConfig:
    teacher_temperature: float = 0.02
    student_temperature: float = 0.10
    diagonal_penalty: float = 1e-6

    def __post_init__(self) -> None:
        if self.teacher_temperature <= 0 or self.student_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.diagonal_penalty <= 0:
            raise ValueError("diagonal_penalty must be positive")


def soft_neighborhood_loss(
    student: Tensor,
    teacher_distance: Tensor,
    config: SoftNeighborhoodConfig | None = None,
) -> Tensor:
    """Match candidate-neighborhood distributions without hard positives.

    ``student`` is [B,D] and ``teacher_distance`` is a symmetric [B,B] matrix.
    The teacher distribution is detached; only the student distribution
    receives gradients.
    """

    settings = config or SoftNeighborhoodConfig()
    if student.ndim != 2:
        raise ValueError("student must be [B,D]")
    if teacher_distance.shape != (student.shape[0], student.shape[0]):
        raise ValueError("teacher_distance must be [B,B]")
    if not torch.isfinite(student).all() or not torch.isfinite(teacher_distance).all():
        raise ValueError("student and teacher must be finite")
    if torch.any(teacher_distance < 0):
        raise ValueError("teacher distances must be non-negative")
    if not torch.allclose(teacher_distance, teacher_distance.transpose(0, 1), atol=1e-5, rtol=0):
        raise ValueError("teacher distance matrix must be symmetric")

    batch_size = student.shape[0]
    if batch_size < 3:
        raise ValueError("soft-neighborhood loss needs at least three samples")
    unit = F.normalize(student.float(), p=2.0, dim=-1, eps=1e-12)
    similarity = unit @ unit.transpose(0, 1)
    mask = ~torch.eye(batch_size, dtype=torch.bool, device=student.device)
    teacher_logits = (-teacher_distance.float() / settings.teacher_temperature).masked_fill(~mask, -1e9)
    student_logits = (similarity / settings.student_temperature).masked_fill(~mask, -1e9)
    teacher_probability = torch.softmax(teacher_logits, dim=1).detach()
    student_log_probability = torch.log_softmax(student_logits, dim=1)
    return -(teacher_probability * student_log_probability).sum(dim=1).mean()
