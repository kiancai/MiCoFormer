"""Checkpoint-initialized MPC relation adaptation primitives."""

from .checkpoint import MPCCheckpointContract, load_frozen_mpc_adapter
from .losses import (
    OrdinalLossConfig,
    OrdinalLossOutput,
    cosine_anchor_loss,
    unifrac_ordinal_loss,
)
from .model import (
    FrozenMPCResidualAdapter,
    ResidualAdapter,
    ResidualAdapterConfig,
    ResidualAdapterOutput,
)

__all__ = [
    "FrozenMPCResidualAdapter",
    "MPCCheckpointContract",
    "OrdinalLossConfig",
    "OrdinalLossOutput",
    "ResidualAdapter",
    "ResidualAdapterConfig",
    "ResidualAdapterOutput",
    "cosine_anchor_loss",
    "load_frozen_mpc_adapter",
    "unifrac_ordinal_loss",
]
