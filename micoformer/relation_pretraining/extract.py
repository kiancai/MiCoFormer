"""Strict, row-addressed embedding export for epoch-0 and trained checkpoints."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import (
    RelationAnnDataDataset,
    _collate_relation_samples,
    sha256_array,
    sha256_file,
)
from .model import RelationModelConfig, RelationOnlyModel
from .module import (
    assert_final_z,
    full_initialization_sha256,
    load_relation_checkpoint,
)


@dataclass
class _EmbeddingCheckpoint:
    model: RelationOnlyModel
    model_config: RelationModelConfig
    data_binding: dict[str, Any]
    checkpoint_kind: str
    checkpoint_contract: dict[str, Any]


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


def _load_embedding_checkpoint(path: Path) -> _EmbeddingCheckpoint:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise RuntimeError("embedding checkpoint root must be a mapping")
    if raw.get("checkpoint_kind") == "relation_architecture_epoch0":
        required = {
            "schema_version",
            "checkpoint_kind",
            "architecture",
            "seed",
            "model_config",
            "data_binding",
            "shared_initialization_sha256",
            "full_initialization_sha256",
            "model_state_dict",
        }
        if set(raw) != required or raw.get("schema_version") != 1 or raw.get("seed") != 42:
            raise RuntimeError("epoch0 architecture checkpoint schema drifted")
        config = RelationModelConfig(**raw["model_config"])
        if raw["architecture"] != config.decoder_kind:
            raise RuntimeError("epoch0 checkpoint architecture/config mismatch")
        model = RelationOnlyModel(config)
        incompatible = model.load_state_dict(raw["model_state_dict"], strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("epoch0 model failed strict state reload")
        if full_initialization_sha256(model) != raw["full_initialization_sha256"]:
            raise RuntimeError("epoch0 model state does not match its initialization hash")
        return _EmbeddingCheckpoint(
            model=model,
            model_config=config,
            data_binding=dict(raw["data_binding"]),
            checkpoint_kind="relation_architecture_epoch0",
            checkpoint_contract={key: value for key, value in raw.items() if key != "model_state_dict"},
        )

    module = load_relation_checkpoint(path, map_location="cpu")
    return _EmbeddingCheckpoint(
        model=module.model,
        model_config=module.model_config,
        data_binding=module.data_binding,
        checkpoint_kind="relation_lightning",
        checkpoint_contract=module.relation_contract,
    )


def _resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested but CUDA is unavailable")
    return resolved


def extract_relation_embeddings(
    *,
    checkpoint_path: str | os.PathLike[str],
    h5ad_path: str | os.PathLike[str],
    row_ids: np.ndarray,
    output_path: str | os.PathLike[str],
    row_source_path: Optional[str | os.PathLike[str]] = None,
    device: str | torch.device = "cuda:0",
    batch_size: int = 32,
    num_workers: int = 0,
    require_d_model_256: bool = True,
) -> Path:
    """Export ``row_ids`` in exactly their supplied order as ``row_ids,z`` NPZ."""

    checkpoint_path = Path(checkpoint_path).resolve()
    h5ad_path = Path(h5ad_path).resolve()
    output_path = Path(output_path).resolve()
    rows = np.asarray(row_ids)
    if rows.dtype != np.int64 or rows.ndim != 1:
        raise TypeError("embedding row_ids must be int64 [N]")
    if rows.size == 0 or np.unique(rows).size != rows.size:
        raise ValueError("embedding row_ids must be non-empty and unique")
    if batch_size <= 0 or batch_size > 32:
        raise ValueError("embedding batch_size must be in [1, 32]")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    loaded = _load_embedding_checkpoint(checkpoint_path)
    if require_d_model_256 and loaded.model_config.d_model != 256:
        raise RuntimeError("the frozen pilot export requires a 256-dimensional final z")
    corpus_hash = sha256_file(h5ad_path)
    if loaded.data_binding.get("corpus_sha256") != corpus_hash:
        raise RuntimeError("checkpoint is bound to a different corpus")
    if row_source_path is not None:
        source = Path(row_source_path).resolve()
        source_rows = np.load(source, allow_pickle=False)
        if not np.array_equal(source_rows, rows):
            raise RuntimeError("row_source_path contents differ from supplied row_ids")
        row_source = {"path": str(source), "sha256": sha256_file(source)}
    else:
        row_source = None

    dataset = RelationAnnDataDataset(
        h5ad_path,
        split_rows=rows,
        project_codes=None,
        site_codes=None,
        require_metadata=False,
        max_tokens=512,
        expected_n_vars=loaded.model_config.vocab_size - 2,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.device(device).type == "cuda",
        collate_fn=_collate_relation_samples,
        persistent_workers=False,
    )
    resolved_device = _resolve_device(device)
    model = loaded.model.to(resolved_device).eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as scratch:
        z_path = Path(scratch) / "z.float32.memmap"
        z = np.memmap(
            z_path,
            mode="w+",
            dtype=np.float32,
            shape=(rows.size, loaded.model_config.d_model),
        )
        cursor = 0
        with torch.inference_mode():
            for batch in loader:
                genus_ids = torch.as_tensor(
                    batch["genus_ids"], device=resolved_device, dtype=torch.long
                )
                rclr = torch.as_tensor(
                    batch["rclr"], device=resolved_device, dtype=torch.float32
                )
                padding_mask = torch.as_tensor(
                    batch["padding_mask"], device=resolved_device, dtype=torch.bool
                )
                with torch.autocast(
                    device_type=resolved_device.type,
                    dtype=torch.float16,
                    enabled=resolved_device.type == "cuda",
                ):
                    output = model(genus_ids, rclr, padding_mask)
                assert_final_z(output.z)
                count = int(output.z.shape[0])
                z[cursor : cursor + count] = output.z.detach().cpu().numpy()
                cursor += count
        dataset.close()
        if cursor != rows.size:
            raise RuntimeError("embedding exporter did not consume the exact row list")
        z.flush()
        norms = np.linalg.norm(z, axis=1)
        if not np.isfinite(z).all() or not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise RuntimeError("exported z is nonfinite or not row-wise unit normalized")
        temporary_npz = Path(scratch) / "embeddings.npz"
        with temporary_npz.open("wb") as handle:
            np.savez(handle, row_ids=rows, z=z)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_npz, output_path)
        z_hash = sha256_array(z)
        # Release the mmap before TemporaryDirectory removes its backing file;
        # otherwise NFS can leave a busy .nfs placeholder behind.
        mmap_handle = getattr(z, "_mmap", None)
        if mmap_handle is not None:
            mmap_handle.close()
        del z

    manifest = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "kind": loaded.checkpoint_kind,
            "contract": loaded.checkpoint_contract,
        },
        "model_config": asdict(loaded.model_config),
        "corpus": {"path": str(h5ad_path), "sha256": corpus_hash},
        "rows": {
            "count": int(rows.size),
            "array_sha256": sha256_array(rows),
            "source": row_source,
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "array_sha256": {
                "row_ids": sha256_array(rows),
                "z": z_hash,
            },
            "z_shape": [int(rows.size), int(loaded.model_config.d_model)],
            "z_dtype": "float32",
            "z_unit_norm": True,
        },
        "transform": {
            "student_input": "top512 raw-abundance ranking; retained-support no-sigma rclr",
            "student_output": "RelationModelOutput.z",
        },
    }
    _atomic_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return output_path
