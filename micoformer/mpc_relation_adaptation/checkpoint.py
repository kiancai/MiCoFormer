"""Fail-closed loading of a full-data learned-ID MPC checkpoint for adaptation."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from micoformer.mpc_pretraining.model import MPCModelConfig, MPCPretrainingModel

from .model import FrozenMPCResidualAdapter, ResidualAdapterConfig


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class MPCCheckpointContract:
    checkpoint_sha256: str
    prior_sha256: str
    training_manifest_sha256: str
    checkpoint_kind: str = "mpc_model_milestone"
    completed_epoch: int = 50
    global_step: int = 264_450

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "prior_sha256",
            "training_manifest_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA256 hex digest")
        if self.checkpoint_kind not in {"mpc_model_milestone", "mpc_full_resume"}:
            raise ValueError("unsupported MPC checkpoint kind")
        if self.completed_epoch <= 0 or self.global_step <= 0:
            raise ValueError("checkpoint epoch and global_step must be positive")


def load_frozen_mpc_adapter(
    checkpoint_path: str | Path,
    prior_assets_path: str | Path,
    contract: MPCCheckpointContract,
    adapter_config: ResidualAdapterConfig | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> FrozenMPCResidualAdapter:
    checkpoint = Path(checkpoint_path).resolve()
    prior = Path(prior_assets_path).resolve()
    if not checkpoint.is_file() or not prior.is_file():
        raise FileNotFoundError("checkpoint and fixed candidate prior must both exist")
    if _sha256_file(checkpoint) != contract.checkpoint_sha256:
        raise RuntimeError("MPC checkpoint SHA256 differs from the adaptation contract")
    if _sha256_file(prior) != contract.prior_sha256:
        raise RuntimeError("fixed candidate prior SHA256 differs from the adaptation contract")
    payload = torch.load(checkpoint, map_location=map_location, weights_only=False)
    if (
        payload.get("checkpoint_kind") != contract.checkpoint_kind
        or payload.get("manifest_sha256") != contract.training_manifest_sha256
        or int(payload.get("epoch", -1)) + 1 != contract.completed_epoch
        or int(payload.get("global_step", -1)) != contract.global_step
    ):
        raise RuntimeError("MPC checkpoint identity differs from the adaptation contract")
    config = MPCModelConfig(**payload["model_config"])
    with np.load(prior, allow_pickle=False) as archive:
        if "pp_table" not in archive.files:
            raise RuntimeError("fixed candidate prior archive is missing pp_table")
        candidate_table = torch.from_numpy(
            np.asarray(archive["pp_table"], dtype=np.float32)
        )
    model = MPCPretrainingModel(candidate_table, config=config)
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("MPC checkpoint failed strict state restore")
    wrapped = FrozenMPCResidualAdapter(model, adapter_config)
    if any(parameter.requires_grad for parameter in wrapped.mpc.parameters()):
        raise RuntimeError("MPC parameters remained trainable after wrapping")
    return wrapped
