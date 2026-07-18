"""Strict row-addressed export of all preregistered C0/C1/C2 readouts."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from micoformer.relation_pretraining.data import (
    RelationAnnDataDataset,
    _collate_relation_samples,
    sha256_array,
    sha256_file,
)
from micoformer.relation_pretraining.module import assert_final_z

from .model import StructureRelationModel
from .module import (
    StructureRelationPretrainingModule,
    build_structure_module,
    load_structure_checkpoint,
    validate_structure_source_manifest,
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_export_module(path: Path) -> tuple[StructureRelationPretrainingModule, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and payload.get("checkpoint_kind") == "structure_epoch0":
        if payload.get("schema_version") != 1:
            raise RuntimeError("unsupported structure epoch0 schema")
        contract = payload.get("relation_contract")
        if not isinstance(contract, dict):
            raise RuntimeError("structure epoch0 contract is missing")
        structure = contract.get("structure")
        if not isinstance(structure, dict):
            raise RuntimeError("structure epoch0 arm metadata is missing")
        validate_structure_source_manifest(structure.get("source_manifest"))
        module = build_structure_module(
            structure_arm=structure["arm"],
            data_binding=contract["data_binding"],
            seed=42,
        )
        if module.relation_contract != contract:
            raise RuntimeError("live structure epoch0 contract differs from artifact")
        state = payload.get("model_state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("structure epoch0 model state is missing")
        incompatible = module.model.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("structure epoch0 model strict reload failed")
        return module, "structure_epoch0"
    return load_structure_checkpoint(path, map_location="cpu"), "structure_lightning"


def extract_structure_embeddings(
    *,
    checkpoint_path: str | os.PathLike[str],
    h5ad_path: str | os.PathLike[str],
    row_ids: np.ndarray,
    output_path: str | os.PathLike[str],
    row_source_path: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cuda:0",
    batch_size: int = 32,
    num_workers: int = 0,
) -> Path:
    checkpoint_path = Path(checkpoint_path).resolve()
    h5ad_path = Path(h5ad_path).resolve()
    output_path = Path(output_path).resolve()
    rows = np.asarray(row_ids)
    if rows.dtype != np.int64 or rows.ndim != 1:
        raise TypeError("row_ids must be int64 [N]")
    if rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError("row_ids must be non-empty and unique")
    if batch_size <= 0 or batch_size > 32 or num_workers < 0:
        raise ValueError("batch_size must be 1..32 and num_workers non-negative")

    module, checkpoint_kind = _load_export_module(checkpoint_path)
    if module.data_binding.get("corpus_sha256") != sha256_file(h5ad_path):
        raise RuntimeError("checkpoint is bound to a different corpus")
    if row_source_path is not None:
        source = Path(row_source_path).resolve()
        if not np.array_equal(np.load(source, allow_pickle=False), rows):
            raise RuntimeError("row source differs from supplied rows")
        row_source = {"path": str(source), "sha256": sha256_file(source)}
    else:
        row_source = None

    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested but CUDA is unavailable")
    dataset = RelationAnnDataDataset(
        h5ad_path,
        split_rows=rows,
        project_codes=None,
        site_codes=None,
        require_metadata=False,
        max_tokens=512,
        expected_n_vars=module.model_config.vocab_size - 2,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=resolved_device.type == "cuda",
        collate_fn=_collate_relation_samples,
        persistent_workers=False,
    )
    model: StructureRelationModel = module.model.to(resolved_device).eval()
    arrays = {
        "relation_z": np.empty((rows.size, 256), dtype=np.float32),
        "backbone_z": np.empty((rows.size, 256), dtype=np.float32),
        "primary_z": np.empty((rows.size, 256), dtype=np.float32),
    }
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            genus_ids = torch.as_tensor(batch["genus_ids"], device=resolved_device, dtype=torch.long)
            rclr = torch.as_tensor(batch["rclr"], device=resolved_device, dtype=torch.float32)
            padding = torch.as_tensor(
                batch["padding_mask"], device=resolved_device, dtype=torch.bool
            )
            with torch.autocast(
                device_type=resolved_device.type,
                dtype=torch.float16,
                enabled=resolved_device.type == "cuda",
            ):
                output = model(genus_ids, rclr, padding)
            assert_final_z(output.z)
            assert_final_z(output.backbone_z)
            assert_final_z(output.downstream_z)
            count = int(output.z.shape[0])
            arrays["relation_z"][cursor : cursor + count] = output.z.cpu().numpy()
            arrays["backbone_z"][cursor : cursor + count] = output.backbone_z.cpu().numpy()
            arrays["primary_z"][cursor : cursor + count] = output.downstream_z.cpu().numpy()
            cursor += count
    dataset.close()
    if cursor != rows.size:
        raise RuntimeError("structure exporter did not consume the exact row list")
    for name, value in arrays.items():
        if not np.isfinite(value).all() or not np.allclose(
            np.linalg.norm(value, axis=1), 1.0, rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError(f"exported {name} is nonfinite or not unit-normalized")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez(
            handle,
            row_ids=rows,
            relation_z=arrays["relation_z"],
            backbone_z=arrays["backbone_z"],
            primary_z=arrays["primary_z"],
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output_path)
    manifest = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "kind": checkpoint_kind,
        },
        "arm": module.structure_arm,
        "relation_contract": module.relation_contract,
        "corpus": {"path": str(h5ad_path), "sha256": sha256_file(h5ad_path)},
        "rows": {
            "count": int(rows.size),
            "array_sha256": sha256_array(rows),
            "source": row_source,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "arrays": {
                "row_ids": sha256_array(rows),
                **{name: sha256_array(value) for name, value in arrays.items()},
            },
        },
        "semantics": {
            "relation_z": "representation consumed by relation mining/loss",
            "backbone_z": "L2-normalized mask-aware final-token mean",
            "primary_z": "preregistered downstream representation",
        },
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return output_path

