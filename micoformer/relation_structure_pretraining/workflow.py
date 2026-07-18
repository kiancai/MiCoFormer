"""Fail-closed preparation, smoke and single-GPU training for C0/C1/C2."""
from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from micoformer.relation_pretraining.data import (
    RelationDataModule,
    sha256_array,
    sha256_file,
)
from micoformer.relation_pretraining.module import (
    assert_final_z,
    full_initialization_sha256,
)
from micoformer.relation_pretraining.workflow import build_relation_module

from .model import STRUCTURE_ARMS
from .module import (
    StructureRelationPretrainingModule,
    build_structure_module,
    build_structure_source_manifest,
    common_initialization_sha256,
    load_structure_checkpoint,
    validate_structure_source_manifest,
)


_ENDPOINT_PATTERN = re.compile(r"^epoch(?P<epoch>\d+)-step(?P<step>\d+)\.ckpt$")


@dataclass(frozen=True)
class StructureRunConfig:
    h5ad_path: Path
    schedule_root: Path
    cache_root: Path
    output_root: Path
    arm: str
    disease_rows_path: Path
    smoke_dir: Path
    device_index: int = 0
    num_workers: int = 0
    resume_checkpoint: Path | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        if self.arm not in STRUCTURE_ARMS:
            raise ValueError(f"arm must be one of {STRUCTURE_ARMS}")
        if self.seed != 42 or self.device_index < 0 or self.num_workers < 0:
            raise ValueError("frozen seed=42, non-negative device/workers required")


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


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _cuda_preflight(device_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; training may not fall back to CPU")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError("requested CUDA device is not visible")
    device = torch.device("cuda", device_index)
    probe = torch.ones(8, device=device)
    if float((probe * probe).sum()) != 8.0:
        raise RuntimeError("CUDA arithmetic preflight failed")
    properties = torch.cuda.get_device_properties(device)
    return {
        "visible_device_count": int(torch.cuda.device_count()),
        "device_index": device_index,
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _epoch0_payload(module: StructureRelationPretrainingModule) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_kind": "structure_epoch0",
        "arm": module.structure_arm,
        "seed": 42,
        "relation_contract": module.relation_contract,
        "model_state_dict": module.model.state_dict(),
    }


def _load_epoch0(path: Path, module: StructureRelationPretrainingModule) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_kind") != "structure_epoch0"
        or payload.get("relation_contract") != module.relation_contract
    ):
        raise RuntimeError("structure epoch0 artifact contract mismatch")
    incompatible = module.model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("structure epoch0 strict state reload failed")
    if full_initialization_sha256(module.model) != module.full_initialization_sha256:
        raise RuntimeError("structure epoch0 full initialization hash mismatch")


def prepare_structure_initialization(
    *,
    h5ad_path: str | os.PathLike[str],
    schedule_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    disease_rows_path: str | os.PathLike[str],
    num_workers: int = 0,
) -> Path:
    output_root = Path(output_root).resolve()
    data = RelationDataModule(
        h5ad_path=h5ad_path,
        schedule_root=schedule_root,
        cache_root=cache_root,
        num_workers=num_workers,
        pin_memory=False,
    )
    cache_records = data.store.validate_all_caches()
    disease_rows = data.store.validate_disease_rows(disease_rows_path)
    modules = {
        arm: build_structure_module(
            structure_arm=arm, data_binding=data.checkpoint_binding, seed=42
        )
        for arm in STRUCTURE_ARMS
    }
    shared = {module.shared_initialization_sha256 for module in modules.values()}
    if len(shared) != 1:
        raise RuntimeError("C0/C1/C2 common backbone initialization hashes differ")

    # C0 must remain an exact fresh copy of the historical main_skip model.
    historical = build_relation_module(
        arm_name="main_skip", data_binding=data.checkpoint_binding, seed=42
    )
    c0 = modules["c0_decoder"]
    if historical.model.state_dict().keys() != c0.model.state_dict().keys():
        raise RuntimeError("fresh C0 model keys differ from historical main_skip")
    for name, value in historical.model.state_dict().items():
        if not torch.equal(value, c0.model.state_dict()[name]):
            raise RuntimeError(f"fresh C0 initialization differs at {name}")
    data.setup("fit")
    first_batch = data.store.epoch_dataset("train", 0, data.datasets["train"])[0]
    historical.eval()
    c0.eval()
    with torch.inference_mode():
        arguments = (
            torch.as_tensor(first_batch["genus_ids"], dtype=torch.long),
            torch.as_tensor(first_batch["rclr"], dtype=torch.float32),
            torch.as_tensor(first_batch["padding_mask"], dtype=torch.bool),
        )
        old_output = historical.model(*arguments)
        c0_output = c0.model(*arguments)
    if not torch.equal(old_output.z, c0_output.z):
        raise RuntimeError("fresh C0 forward differs from historical main_skip")

    artifacts: dict[str, dict[str, str]] = {}
    for arm, module in modules.items():
        path = output_root / "epoch0" / f"{arm}.ckpt"
        if not path.exists():
            _atomic_torch(path, _epoch0_payload(module))
        verifier = build_structure_module(
            structure_arm=arm, data_binding=data.checkpoint_binding, seed=42
        )
        _load_epoch0(path, verifier)
        artifacts[arm] = {"path": str(path), "sha256": sha256_file(path)}
    source_manifest = validate_structure_source_manifest(
        c0.relation_contract["structure"]["source_manifest"]
    )
    manifest = {
        "schema_version": 1,
        "seed": 42,
        "source_manifest": source_manifest,
        "data_binding": data.checkpoint_binding,
        "common_initialization_sha256": next(iter(shared)),
        "c0_historical_parity": True,
        "artifacts": artifacts,
        "cache_records": cache_records,
        "disease_rows": {
            "path": str(Path(disease_rows_path).resolve()),
            "sha256": sha256_file(disease_rows_path),
            "array_sha256": sha256_array(disease_rows),
            "count": int(disease_rows.size),
        },
    }
    path = output_root / "epoch0" / "manifest.json"
    _atomic_json(path, manifest)
    return path


def _load_init_manifest(
    output_root: Path,
    data_binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = output_root / "epoch0" / "manifest.json"
    if not path.is_file():
        raise RuntimeError("run prepare-init before smoke or training")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("data_binding") != dict(data_binding):
        raise RuntimeError("structure init data binding differs")
    validate_structure_source_manifest(manifest.get("source_manifest"))
    return manifest


def run_structure_cuda_smoke(
    *,
    h5ad_path: str | os.PathLike[str],
    schedule_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    device_index: int = 0,
) -> Path:
    output_root = Path(output_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("structure smoke output directory must be fresh")
    output_dir.mkdir(parents=True, exist_ok=True)
    cuda = _cuda_preflight(device_index)
    data = RelationDataModule(
        h5ad_path=h5ad_path,
        schedule_root=schedule_root,
        cache_root=cache_root,
        num_workers=0,
        pin_memory=True,
    )
    data.store.validate_all_caches()
    manifest = _load_init_manifest(output_root, data.checkpoint_binding)
    data.setup("fit")
    batch = data.store.epoch_dataset("train", 0, data.datasets["train"])[0]
    device = torch.device("cuda", device_index)
    results: dict[str, Any] = {}
    for arm in STRUCTURE_ARMS:
        module = build_structure_module(
            structure_arm=arm, data_binding=data.checkpoint_binding, seed=42
        )
        artifact = Path(manifest["artifacts"][arm]["path"])
        if sha256_file(artifact) != manifest["artifacts"][arm]["sha256"]:
            raise RuntimeError("structure epoch0 artifact hash drifted")
        _load_epoch0(artifact, module)
        module = module.to(device).train()
        # The smoke invokes the exact shared step, while suppressing Lightning
        # logger I/O because no Trainer is attached.
        module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
        module._trainer = SimpleNamespace(estimated_stepping_batches=21_560)
        configured = module.configure_optimizers()
        module._trainer = None
        optimizer = configured["optimizer"]
        scheduler = configured["lr_scheduler"]["scheduler"]
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            step = module._shared_step(batch, "train")
        if not step.has_relation_update or not torch.isfinite(step.loss):
            raise RuntimeError(f"{arm} smoke has no finite relation update")
        step.loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in module.parameters()
        ):
            raise RuntimeError(f"{arm} smoke produced a nonfinite gradient")
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            output = module.model(
                torch.as_tensor(batch["genus_ids"], device=device, dtype=torch.long),
                torch.as_tensor(batch["rclr"], device=device, dtype=torch.float32),
                torch.as_tensor(batch["padding_mask"], device=device, dtype=torch.bool),
            )
            assert_final_z(output.z)
            assert_final_z(output.backbone_z)
            assert_final_z(output.downstream_z)
        results[arm] = {
            "loss": float(step.loss.detach().cpu()),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "common_initialization_sha256": module.shared_initialization_sha256,
            "full_initialization_sha256": module.full_initialization_sha256,
        }
        del module, optimizer, scheduler, output
        torch.cuda.empty_cache()
    if len({record["common_initialization_sha256"] for record in results.values()}) != 1:
        raise RuntimeError("structure smoke arms lost matched common initialization")
    payload = {
        "schema_version": 1,
        "status": "passed",
        "source_manifest": build_structure_source_manifest(),
        "data_binding": data.checkpoint_binding,
        "initialization_manifest": {
            "path": str(output_root / "epoch0/manifest.json"),
            "sha256": sha256_file(output_root / "epoch0/manifest.json"),
        },
        "cuda": cuda,
        "arms": results,
    }
    path = output_dir / "smoke_manifest.json"
    _atomic_json(path, payload)
    (output_dir / ".complete").touch(exist_ok=False)
    return path


def _validate_smoke(
    smoke_dir: Path,
    *,
    data_binding: Mapping[str, Any],
    initialization_manifest_path: Path,
) -> dict[str, Any]:
    if not (smoke_dir / ".complete").is_file():
        raise RuntimeError("passed structure smoke sentinel is missing")
    payload = json.loads((smoke_dir / "smoke_manifest.json").read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("data_binding") != dict(data_binding):
        raise RuntimeError("structure smoke status/data binding mismatch")
    validate_structure_source_manifest(payload.get("source_manifest"))
    if payload.get("initialization_manifest", {}).get("sha256") != sha256_file(
        initialization_manifest_path
    ):
        raise RuntimeError("structure smoke binds a different initialization manifest")
    return payload


def run_structure_pretraining(config: StructureRunConfig) -> Path:
    output_root = config.output_root.resolve()
    data = RelationDataModule(
        h5ad_path=config.h5ad_path,
        schedule_root=config.schedule_root,
        cache_root=config.cache_root,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    cache_records = data.store.validate_all_caches()
    disease_rows = data.store.validate_disease_rows(config.disease_rows_path)
    initialization = _load_init_manifest(output_root, data.checkpoint_binding)
    initialization_path = output_root / "epoch0/manifest.json"
    smoke = _validate_smoke(
        config.smoke_dir.resolve(),
        data_binding=data.checkpoint_binding,
        initialization_manifest_path=initialization_path,
    )
    module = build_structure_module(
        structure_arm=config.arm, data_binding=data.checkpoint_binding, seed=config.seed
    )
    artifact_record = initialization["artifacts"][config.arm]
    artifact = Path(artifact_record["path"])
    if sha256_file(artifact) != artifact_record["sha256"]:
        raise RuntimeError("structure epoch0 artifact hash drifted")
    _load_epoch0(artifact, module)
    if module.shared_initialization_sha256 != initialization["common_initialization_sha256"]:
        raise RuntimeError("structure arm common initialization differs from manifest")
    cuda = _cuda_preflight(config.device_index)

    resume_path: str | None = None
    if config.resume_checkpoint is not None:
        restored = load_structure_checkpoint(
            config.resume_checkpoint,
            expected_data_binding=data.checkpoint_binding,
            map_location="cpu",
        )
        if restored.relation_contract != module.relation_contract:
            raise RuntimeError("resume checkpoint structure contract drifted")
        data.store.verify_cache_provenance(restored.consumed_teacher_caches)
        resume_path = str(config.resume_checkpoint.resolve())

    arm_dir = output_root / config.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": 1,
        "arm": config.arm,
        "seed": config.seed,
        "source_manifest": build_structure_source_manifest(),
        "relation_contract": module.relation_contract,
        "data_binding": data.checkpoint_binding,
        "cache_records": cache_records,
        "initialization": {
            "manifest_path": str(initialization_path),
            "manifest_sha256": sha256_file(initialization_path),
            "artifact_path": str(artifact),
            "artifact_sha256": sha256_file(artifact),
        },
        "smoke_manifest_sha256": sha256_file(config.smoke_dir / "smoke_manifest.json"),
        "cuda": cuda,
        "disease_rows": {
            "path": str(config.disease_rows_path.resolve()),
            "sha256": sha256_file(config.disease_rows_path),
            "array_sha256": sha256_array(disease_rows),
            "count": int(disease_rows.size),
        },
        "resume_checkpoint": resume_path,
    }
    manifest_path = arm_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != run_manifest:
            raise RuntimeError("existing structure run manifest differs")
    else:
        _atomic_json(manifest_path, run_manifest)
    # Revalidate smoke immediately before Trainer creation.
    _validate_smoke(
        config.smoke_dir.resolve(),
        data_binding=data.checkpoint_binding,
        initialization_manifest_path=initialization_path,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=arm_dir / "checkpoints",
        filename="epoch{epoch:02d}-step{step}",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=[config.device_index],
        strategy="auto",
        precision="16-mixed",
        max_epochs=10,
        accumulate_grad_batches=1,
        deterministic=True,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="step")],
        logger=CSVLogger(save_dir=arm_dir, name="logs"),
        default_root_dir=arm_dir,
        reload_dataloaders_every_n_epochs=1,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
    )
    trainer.fit(module, datamodule=data, ckpt_path=resume_path)
    candidates = []
    for path in (arm_dir / "checkpoints").glob("epoch*-step*.ckpt"):
        match = _ENDPOINT_PATTERN.fullmatch(path.name)
        if match:
            candidates.append((int(match.group("epoch")), int(match.group("step")), path))
    final = [item for item in candidates if item[0] == 9]
    if len(final) != 1:
        raise RuntimeError(f"expected one immutable epoch09 checkpoint, found {len(final)}")
    epoch, step, endpoint = final[0]
    restored = load_structure_checkpoint(
        endpoint,
        expected_data_binding=data.checkpoint_binding,
        map_location="cpu",
    )
    if restored.relation_contract != module.relation_contract:
        raise RuntimeError("final structure checkpoint contract drifted")
    endpoint_manifest = {
        "schema_version": 1,
        "arm": config.arm,
        "endpoint": {
            "path": str(endpoint.resolve()),
            "sha256": sha256_file(endpoint),
            "epoch": epoch,
            "global_step": step,
        },
        "run_manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "relation_contract": module.relation_contract,
        "runtime_counts": restored.relation_runtime_counts,
    }
    _atomic_json(arm_dir / "endpoints/arm_completion.json", endpoint_manifest)
    return endpoint.resolve()

