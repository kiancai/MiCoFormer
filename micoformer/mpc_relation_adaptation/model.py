"""A small residual adapter on top of a strictly frozen MPC sample embedding."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from micoformer.mpc_pretraining.model import MPCPretrainingModel


@dataclass(frozen=True)
class ResidualAdapterConfig:
    """Architecture-only configuration; training dose belongs to the run contract."""

    d_model: int = 512
    bottleneck_dim: int = 64

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.bottleneck_dim <= 0:
            raise ValueError("adapter dimensions must be positive")
        if self.bottleneck_dim >= self.d_model:
            raise ValueError("the residual adapter must use a strict bottleneck")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class ResidualAdapterOutput(NamedTuple):
    """Baseline and adapted sample representations from the same frozen MPC pass."""

    h_raw: Tensor
    h_unit: Tensor
    z_raw: Tensor
    z_unit: Tensor


class ResidualAdapter(nn.Module):
    """Pre-norm bottleneck adapter with exact zero-update parity at initialization."""

    def __init__(self, config: ResidualAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResidualAdapterConfig()
        self.norm = nn.LayerNorm(self.config.d_model)
        self.down = nn.Linear(self.config.d_model, self.config.bottleneck_dim)
        self.up = nn.Linear(self.config.bottleneck_dim, self.config.d_model)
        # Exact h -> z parity is a load/smoke invariant, not an approximate hope.
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h_raw: Tensor) -> ResidualAdapterOutput:
        if h_raw.ndim != 2 or h_raw.shape[1] != self.config.d_model:
            raise ValueError(
                f"h_raw must have shape [B,{self.config.d_model}], got {tuple(h_raw.shape)}"
            )
        if not torch.is_floating_point(h_raw) or not bool(torch.isfinite(h_raw).all()):
            raise ValueError("h_raw must be finite floating point")
        delta = self.up(F.gelu(self.down(self.norm(h_raw))))
        z_raw = h_raw + delta
        return ResidualAdapterOutput(
            h_raw=h_raw,
            h_unit=F.normalize(h_raw.float(), p=2.0, dim=-1, eps=1e-12),
            z_raw=z_raw,
            z_unit=F.normalize(z_raw.float(), p=2.0, dim=-1, eps=1e-12),
        )


class FrozenMPCResidualAdapter(nn.Module):
    """Keep MPC in eval/no-grad mode and expose only adapter parameters to training."""

    def __init__(
        self,
        mpc: MPCPretrainingModel,
        adapter_config: ResidualAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        config = adapter_config or ResidualAdapterConfig(d_model=mpc.config.d_model)
        if config.d_model != mpc.config.d_model:
            raise ValueError("adapter d_model must match the MPC sample embedding dimension")
        self.mpc = mpc
        self.adapter = ResidualAdapter(config)
        for parameter in self.mpc.parameters():
            parameter.requires_grad_(False)
        self.mpc.eval()

    def train(self, mode: bool = True) -> "FrozenMPCResidualAdapter":
        super().train(mode)
        # super().train() recurses into all children, so restore the frozen encoder
        # to eval mode to disable its dropout even while the adapter is training.
        self.mpc.eval()
        return self

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        padding_mask: Tensor,
    ) -> ResidualAdapterOutput:
        with torch.no_grad():
            _, h_raw = self.mpc.unmasked_representations(genus_ids, rclr, padding_mask)
        return self.adapter(h_raw.detach())

    def trainable_parameter_names(self) -> tuple[str, ...]:
        return tuple(name for name, value in self.named_parameters() if value.requires_grad)

    def assert_gradient_boundary(self) -> None:
        leaked = [
            name
            for name, parameter in self.mpc.named_parameters()
            if parameter.requires_grad or parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError(f"frozen MPC received trainable state or gradients: {leaked[:5]}")
        missing = [
            name
            for name, parameter in self.adapter.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if missing:
            raise RuntimeError(f"adapter gradients are missing: {missing}")
