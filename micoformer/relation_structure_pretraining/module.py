"""Lightning modules and strict provenance for fresh C0/C1/C2 arms."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import lightning as L
import torch
from torch import Tensor

from micoformer.relation_pretraining.mining import MiningConfig
from micoformer.relation_pretraining.model import RelationModelConfig
from micoformer.relation_pretraining.module import (
    RESUME_DETERMINISM_CONTRACT,
    RelationOptimizationConfig,
    RelationPretrainingModule,
    _json_copy,
    _state_dict_sha256,
    _validate_relation_rng_state,
    build_relation_source_manifest,
    capture_relation_rng_state,
    full_initialization_sha256,
    restore_relation_rng_state,
)

from .model import STRUCTURE_ARMS, StructureArm, StructureRelationModel


STRUCTURE_SOURCE_PATHS = (
    "micoformer/relation_structure_pretraining/__init__.py",
    "micoformer/relation_structure_pretraining/model.py",
    "micoformer/relation_structure_pretraining/module.py",
    "micoformer/relation_structure_pretraining/workflow.py",
    "micoformer/relation_structure_pretraining/extract.py",
    "scripts/2.train_relation_structure.py",
)
COMMON_PREFIXES = ("input_stem.", "encoder.", "final_token_norm.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def build_structure_source_manifest() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    files: dict[str, str] = {}
    for relative in STRUCTURE_SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise RuntimeError(f"required structure source is missing: {path}")
        files[relative] = _file_sha256(path)
    body: dict[str, Any] = {
        "schema_version": 1,
        "repository_root": str(repository),
        "files": files,
        "parent_relation_source_manifest": build_relation_source_manifest(),
    }
    body["manifest_sha256"] = _canonical_sha256(body)
    return body


def validate_structure_source_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("structure source manifest is missing or malformed")
    recorded = manifest.get("manifest_sha256")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if recorded != _canonical_sha256(body):
        raise RuntimeError("structure source manifest self-hash is invalid")
    current = build_structure_source_manifest()
    if manifest != current:
        raise RuntimeError("structure source manifest differs from the live source tree")
    return current


def common_initialization_sha256(model: StructureRelationModel) -> str:
    selected = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(COMMON_PREFIXES)
    }
    expected = {
        name
        for name in model.state_dict()
        if name.startswith(COMMON_PREFIXES)
    }
    if set(selected) != expected or not selected:
        raise RuntimeError("failed to isolate the common relation backbone")
    return _state_dict_sha256(selected)


class StructureRelationPretrainingModule(RelationPretrainingModule):
    """Reuse the exact relation step while swapping the sample package."""

    def __init__(
        self,
        *,
        structure_arm: StructureArm,
        data_binding: Mapping[str, Any],
        model_config: RelationModelConfig | Mapping[str, Any] | None = None,
        optimization_config: RelationOptimizationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        if structure_arm not in STRUCTURE_ARMS:
            raise ValueError(f"structure_arm must be one of {STRUCTURE_ARMS}")
        if model_config is None:
            model_config = RelationModelConfig(decoder_kind="main")
        elif isinstance(model_config, Mapping):
            model_config = RelationModelConfig(**dict(model_config))
        if model_config.decoder_kind != "main":
            raise ValueError("all C0/C1/C2 arms require the same main-decoder init stream")

        super().__init__(
            arm_name="main_skip",
            model_config=model_config,
            data_binding=data_binding,
            mining_config=MiningConfig(no_next_mode="skip"),
            optimization_config=optimization_config,
        )
        base = self.model
        self.model = StructureRelationModel(base, structure_arm)
        self.structure_arm = structure_arm
        self.arm_name = structure_arm
        self.shared_initialization_sha256 = common_initialization_sha256(self.model)
        self.full_initialization_sha256 = full_initialization_sha256(self.model)
        structure_manifest = build_structure_source_manifest()
        self.relation_contract = {
            "schema_version": 1,
            "arm_name": self.arm_name,
            "model_config": asdict(self.model_config),
            "mining_config": asdict(self.mining_config),
            "optimization_config": asdict(self.optimization_config),
            "data_binding": copy.deepcopy(self.data_binding),
            "shared_initialization_sha256": self.shared_initialization_sha256,
            "full_initialization_sha256": self.full_initialization_sha256,
            # Parent methods validate this unchanged historical dependency.
            "source_manifest": structure_manifest["parent_relation_source_manifest"],
            "resume_determinism": copy.deepcopy(RESUME_DETERMINISM_CONTRACT),
            "structure": {
                "schema_version": 1,
                "arm": structure_arm,
                "relation_readout": (
                    "decoder_z"
                    if structure_arm == "c0_decoder"
                    else "backbone_z"
                    if structure_arm == "c1_token_mean"
                    else "projector_z"
                ),
                "downstream_primary": (
                    "decoder_z" if structure_arm == "c0_decoder" else "backbone_z"
                ),
                "projector": (
                    "Linear(256,512)->GELU->LayerNorm(512)->Linear(512,256)"
                    if structure_arm == "c2_projector"
                    else None
                ),
                "source_manifest": structure_manifest,
            },
        }
        # Lightning's hparams are descriptive; the explicit relation_contract
        # remains the fail-closed checkpoint source of truth.
        self.save_hyperparameters(copy.deepcopy(self.relation_contract))

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        validate_structure_source_manifest(
            self.relation_contract["structure"]["source_manifest"]
        )
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        contract = checkpoint.get("relation_contract")
        if not isinstance(contract, dict):
            raise RuntimeError("structure checkpoint relation contract is missing")
        structure = contract.get("structure")
        if not isinstance(structure, dict):
            raise RuntimeError("checkpoint is not a structure-relation checkpoint")
        validate_structure_source_manifest(structure.get("source_manifest"))
        super().on_load_checkpoint(checkpoint)


def build_structure_module(
    *,
    structure_arm: StructureArm,
    data_binding: Mapping[str, Any],
    seed: int = 42,
) -> StructureRelationPretrainingModule:
    if seed != 42:
        raise ValueError("the matched C0/C1/C2 experiment requires seed=42")
    L.seed_everything(seed, workers=True)
    return StructureRelationPretrainingModule(
        structure_arm=structure_arm,
        data_binding=data_binding,
        model_config=RelationModelConfig(decoder_kind="main"),
        optimization_config=RelationOptimizationConfig(),
    )


def load_structure_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_data_binding: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng_state: bool = False,
) -> StructureRelationPretrainingModule:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("structure checkpoint root must be a mapping")
    contract = checkpoint.get("relation_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise RuntimeError("structure checkpoint relation contract is missing")
    structure = contract.get("structure")
    if not isinstance(structure, dict) or structure.get("arm") not in STRUCTURE_ARMS:
        raise RuntimeError("checkpoint is not a known C0/C1/C2 structure arm")
    validate_structure_source_manifest(structure.get("source_manifest"))
    data_binding = contract.get("data_binding")
    if not isinstance(data_binding, dict):
        raise RuntimeError("structure checkpoint data binding is missing")
    if expected_data_binding is not None and data_binding != _json_copy(expected_data_binding):
        raise RuntimeError("structure checkpoint data binding mismatch")

    caller_rng = capture_relation_rng_state() if not restore_rng_state else None
    try:
        module = StructureRelationPretrainingModule(
            structure_arm=structure["arm"],
            data_binding=data_binding,
            model_config=contract["model_config"],
            optimization_config=contract["optimization_config"],
        )
    finally:
        if caller_rng is not None:
            restore_relation_rng_state(caller_rng)
    module.shared_initialization_sha256 = contract["shared_initialization_sha256"]
    module.full_initialization_sha256 = contract["full_initialization_sha256"]
    module.relation_contract = copy.deepcopy(contract)
    module._restore_checkpoint_metadata(checkpoint, restore_rng_state=restore_rng_state)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("structure checkpoint state_dict is missing")
    incompatible = module.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict structure checkpoint reload reported incompatible keys")
    module._validate_restored_runtime_state()
    if restore_rng_state:
        state = module._pending_resume_rng_state
        _validate_relation_rng_state(state)
        restore_relation_rng_state(state)
        module._pending_resume_rng_state = None
    return module

