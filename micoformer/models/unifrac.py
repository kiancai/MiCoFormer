"""Normalized weighted UniFrac and batch-local soft-neighborhood supervision.
归一化加权 UniFrac，以及批内软邻域监督。

Ported from dev@28de0ee4f2b363cb4130f86064a542c3e3666fea,
adapted from teacher.py to the reviewed MiCoFormer interface.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor
import torch.nn.functional as F


class NeighborhoodLoss(NamedTuple):
    loss: Tensor
    teacher_entropy: Tensor
    kl: Tensor


def weighted_unifrac(edge_mass: Tensor, branch_lengths: Tensor) -> Tensor:
    """Return full-composition normalized weighted UniFrac distances [B, B].
    根据完整组成的树枝质量，计算归一化加权 UniFrac 距离 [B, B]。
    """
    # Teachers are fixed targets; calculate in float32 even inside BF16 autocast.
    # teacher 不参与反传；即使外层启用 BF16，也使用 float32 计算。
    with torch.autocast(device_type=edge_mass.device.type, enabled=False):
        mass = edge_mass.detach().float()
        lengths = branch_lengths.detach().float()
        weighted = mass * lengths
        total = weighted.sum(1)
        distances = torch.cdist(weighted, weighted, p=1) / (
            total[:, None] + total[None, :]
        )
        return distances


def neighborhood_loss(
    sample_embeddings: Tensor,
    distances: Tensor,
    *,
    teacher_temperature: float = 0.03,
    student_temperature: float = 0.05,
) -> NeighborhoodLoss:
    """Match off-diagonal teacher neighborhoods using sample cosine similarities.
    用样本余弦相似度拟合 teacher 的非对角软邻域，返回 CE、熵和 KL。
    """
    count = sample_embeddings.shape[0]
    with torch.autocast(device_type=sample_embeddings.device.type, enabled=False):
        diagonal = torch.eye(count, device=sample_embeddings.device, dtype=torch.bool)
        teacher_logits = (-distances.detach().float() / teacher_temperature).masked_fill(
            diagonal, -torch.inf
        )
        probability = torch.softmax(teacher_logits, dim=1)
        normalized = F.normalize(sample_embeddings.float(), dim=1)
        logits = ((normalized @ normalized.T) / student_temperature).masked_fill(
            diagonal, -torch.inf
        )
        # Zero diagonal log probabilities before multiplying by zero target mass.
        # 对角线目标概率为零；先置零对应 log 概率，避免 0 * -inf。
        log_probability = torch.log_softmax(logits, dim=1).masked_fill(diagonal, 0.0)
        loss = -(probability * log_probability).sum(1).mean()
        with torch.no_grad():
            teacher_log_probability = torch.log_softmax(
                teacher_logits, dim=1
            ).masked_fill(diagonal, 0.0)
            entropy = -(probability * teacher_log_probability).sum(1).mean()
            kl = (
                probability * (teacher_log_probability - log_probability.detach())
            ).sum(1).mean()
        return NeighborhoodLoss(loss, entropy, kl)
