"""GPU teachers for the explicitly non-formal fast full-corpus run.

The protein distance is a barycenter lower bound under the same centered-cosine
ground features as the formal balanced OT teacher.  UniFrac is exact for the
retained top-512 composition.  Neither may be relabeled as the formal
full-composition exact teacher.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from scipy import sparse
from torch import Tensor

from .module import FullscaleRelationPretrainingModule, _RelationStep


FAST_TEACHER_KIND = "top512_protein_barycenter_lb_and_top512_weighted_unifrac"


class FastFullCorpusTeacher(nn.Module):
    """Compute both fast teacher matrices on the model device."""

    def __init__(
        self,
        *,
        protein_features: str | Path,
        protein_valid_mask: str | Path,
        genus_to_edge: str | Path,
        branch_lengths: str | Path,
        expected_n_genus: int = 8_114,
    ) -> None:
        super().__init__()
        features = np.asarray(np.load(protein_features), dtype=np.float32)
        valid = np.asarray(np.load(protein_valid_mask), dtype=np.bool_)
        incidence_csr = sparse.load_npz(genus_to_edge).tocsr()
        branches = np.asarray(np.load(branch_lengths), dtype=np.float32)
        if features.ndim != 2 or features.shape[0] != expected_n_genus:
            raise ValueError("protein feature asset is not aligned to the frozen genus universe")
        if valid.shape != (expected_n_genus,):
            raise ValueError("protein valid mask is not aligned to the frozen genus universe")
        if incidence_csr.shape[0] != expected_n_genus or incidence_csr.shape[1] != branches.size:
            raise ValueError("UniFrac assets are not aligned to the frozen genus universe")
        if not np.isfinite(features).all() or not np.isfinite(branches).all():
            raise ValueError("fast teacher assets must be finite")
        if np.any(branches < 0):
            raise ValueError("UniFrac branch lengths must be non-negative")
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        unit = features / np.maximum(norms, 1e-8)
        unit[~valid] = 0.0
        incidence = incidence_csr.toarray().astype(np.float32, copy=False)
        self.register_buffer("protein_unit", torch.from_numpy(unit), persistent=False)
        self.register_buffer("protein_valid", torch.from_numpy(valid), persistent=False)
        self.register_buffer("genus_to_edge", torch.from_numpy(incidence), persistent=False)
        self.register_buffer("branch_lengths", torch.from_numpy(branches), persistent=False)

    @torch.no_grad()
    def forward(
        self,
        var_indices: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor]]:
        if var_indices.ndim != 2 or rclr.shape != var_indices.shape or padding_mask.shape != var_indices.shape:
            raise ValueError("fast teacher inputs must be aligned [B,L]")
        if var_indices.device != self.protein_unit.device:
            raise RuntimeError("fast teacher assets and inputs are on different devices")
        batch_size = int(var_indices.shape[0])
        if batch_size < 2:
            raise ValueError("fast relation teacher requires at least two samples")
        safe_indices = var_indices.masked_fill(padding_mask, 0)
        if bool(((safe_indices < 0) | (safe_indices >= self.protein_unit.shape[0])).any()):
            raise ValueError("fast teacher received an invalid genus index")
        logits = rclr.float().masked_fill(padding_mask, float("-inf"))
        weights = torch.softmax(logits, dim=1).masked_fill(padding_mask, 0.0)
        composition = torch.zeros(
            (batch_size, self.protein_unit.shape[0]),
            dtype=torch.float32,
            device=var_indices.device,
        )
        composition.scatter_add_(1, safe_indices, weights)
        if not torch.allclose(
            composition.sum(dim=1),
            torch.ones(batch_size, device=composition.device),
            atol=2e-6,
            rtol=2e-6,
        ):
            raise RuntimeError("retained fast-teacher composition is not normalized")

        protein_mu = composition @ self.protein_unit.float()
        protein_norm = protein_mu.square().sum(dim=1)
        protein = 0.5 * (
            protein_norm[:, None]
            + protein_norm[None, :]
            - 2.0 * (protein_mu @ protein_mu.transpose(0, 1))
        )
        protein = protein.clamp_min_(0.0)
        protein.fill_diagonal_(0.0)
        valid_mass = composition @ self.protein_valid.float()
        endpoint_valid = valid_mass >= 0.90
        protein_validity = endpoint_valid[:, None] & endpoint_valid[None, :]

        profiles = composition @ self.genus_to_edge.float()
        weighted_profiles = profiles * self.branch_lengths.float()
        numerator = (
            profiles[:, None, :] - profiles[None, :, :]
        ).abs().mul_(self.branch_lengths.float()).sum(dim=-1)
        profile_mass = weighted_profiles.sum(dim=1)
        denominator = profile_mass[:, None] + profile_mass[None, :]
        if bool((denominator <= 0).any()) or not bool(torch.isfinite(denominator).all()):
            raise RuntimeError("fast UniFrac denominator is non-positive or non-finite")
        unifrac = (numerator / denominator).clamp_(0.0, 1.0)
        unifrac.fill_diagonal_(0.0)
        unifrac_validity = torch.ones_like(unifrac, dtype=torch.bool)
        diagnostics = {
            "protein_valid_mass_min": valid_mass.min(),
            "protein_valid_fraction": endpoint_valid.float().mean(),
        }
        return (
            {"protein": protein, "unifrac": unifrac},
            {"protein": protein_validity, "unifrac": unifrac_validity},
            diagnostics,
        )


class FastFullCorpusPretrainingModule(FullscaleRelationPretrainingModule):
    """Inject the bounded fast GPU teachers before the unchanged F2 loss."""

    def __init__(self, *, fast_teacher_assets: Mapping[str, str], **kwargs: Any) -> None:
        if kwargs.get("arm") != "f2_dual":
            raise ValueError("the fast full-corpus contract permits only f2_dual")
        super().__init__(**kwargs)
        required = {
            "protein_features",
            "protein_valid_mask",
            "genus_to_edge",
            "branch_lengths",
        }
        if set(fast_teacher_assets) != required:
            raise ValueError("fast_teacher_assets keys drifted")
        self.fast_teacher_assets = dict(fast_teacher_assets)
        self.fast_teacher = FastFullCorpusTeacher(**self.fast_teacher_assets)

    def _relation_forward(self, batch: Mapping[str, Any], stage: str) -> _RelationStep:
        if "teacher_distances" in batch or "teacher_validity" in batch:
            raise RuntimeError("fast full-corpus batch may not inject cached teachers")
        var_indices = torch.as_tensor(batch["var_indices"], device=self.device, dtype=torch.long)
        rclr = torch.as_tensor(batch["rclr"], device=self.device, dtype=torch.float32)
        padding = torch.as_tensor(batch["padding_mask"], device=self.device, dtype=torch.bool)
        with torch.autocast(device_type=self.device.type, enabled=False):
            distances, validity, diagnostics = self.fast_teacher(var_indices, rclr, padding)
        enriched = dict(batch)
        enriched["teacher_distances"] = distances
        enriched["teacher_validity"] = validity
        batch_size = int(var_indices.shape[0])
        for name, value in diagnostics.items():
            self.log(f"{stage}/fast_teacher/{name}", value, batch_size=batch_size)
        return super()._relation_forward(enriched, stage)
