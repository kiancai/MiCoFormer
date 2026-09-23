"""Exact full-composition weighted UniFrac teacher for pilot batches."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sparse
import torch
from torch import Tensor, nn


TEACHER_DEFINITION_ID = (
    "normalized_weighted_unifrac_v2_branch_weighted_numerator_denominator"
)


class UniFracTeacher(nn.Module):
    """Compute normalized weighted UniFrac on the model device.

    The teacher uses the full raw composition supplied by the batch, not the
    top-512 student input.  The edge incidence and branch lengths are fixed
    buffers and never enter the student graph.
    """

    def __init__(
        self,
        *,
        incidence_path: str | Path,
        branch_lengths_path: str | Path,
        expected_n_genus: int = 8_114,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        incidence = sparse.load_npz(incidence_path).tocsr()
        branches = np.asarray(np.load(branch_lengths_path), dtype=np.float32)
        if incidence.shape != (expected_n_genus, branches.size):
            raise ValueError("UniFrac assets are not aligned to the fixed genus universe")
        if incidence.nnz == 0:
            raise ValueError("UniFrac incidence asset is empty")
        if not np.isfinite(incidence.data).all() or np.any(incidence.data < 0):
            raise ValueError("UniFrac incidence values must be finite and non-negative")
        if not np.all(np.isin(incidence.data, (0, 1))):
            raise ValueError("UniFrac incidence must be a binary descendant matrix")
        if not np.isfinite(branches).all() or np.any(branches < 0):
            raise ValueError("UniFrac branch lengths must be finite and non-negative")
        if not np.any(branches > 0):
            raise ValueError("UniFrac branch lengths must contain a positive edge")
        dense = np.asarray(incidence.toarray(), dtype=np.float32)
        self.register_buffer("incidence", torch.from_numpy(dense), persistent=False)
        self.register_buffer(
            "branch_lengths", torch.from_numpy(branches), persistent=False
        )

    @torch.no_grad()
    def forward(self, composition: Tensor) -> Tensor:
        if composition.ndim != 2 or composition.shape[1] != self.incidence.shape[0]:
            raise ValueError("composition must be [B,8114] on the teacher genus universe")
        if composition.device != self.incidence.device:
            raise RuntimeError("composition and UniFrac assets must share a device")
        if not torch.isfinite(composition).all() or torch.any(composition < 0):
            raise ValueError("composition must be finite non-negative abundance")
        # The frozen teacher is a distance on sample-relative composition.  Keep
        # its matrix multiplication in float32 even inside student bf16 autocast.
        with torch.autocast(device_type=composition.device.type, enabled=False):
            composition = composition.float()
            totals = composition.sum(dim=1, keepdim=True)
            if bool((totals <= 0).any()) or not bool(torch.isfinite(totals).all()):
                raise ValueError("composition rows must have positive finite totals")
            profiles = (composition / totals) @ self.incidence.float()
            weighted = profiles * self.branch_lengths.float()
            denominator = weighted.sum(dim=1)[:, None] + weighted.sum(dim=1)[None, :]
            if bool((denominator <= 0).any()) or not bool(torch.isfinite(denominator).all()):
                raise RuntimeError("UniFrac denominator is non-positive or non-finite")
            chunks: list[Tensor] = []
            for start in range(0, profiles.shape[0], self.chunk_size):
                end = min(start + self.chunk_size, profiles.shape[0])
                # sum_e length_e * |p_ie - p_je|, not unweighted edge mass.
                numerator = (
                    weighted[start:end, None, :] - weighted[None, :, :]
                ).abs().sum(dim=-1)
                chunks.append((numerator / denominator[start:end]).clamp_(0.0, 1.0))
            result = torch.cat(chunks, dim=0)
            result.fill_diagonal_(0.0)
            return result
