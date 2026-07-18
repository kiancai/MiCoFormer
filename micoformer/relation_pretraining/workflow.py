"""Fail-closed orchestration for one frozen relation-only training arm."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from .data import RelationDataModule, sha256_array, sha256_file
from .extract import _load_embedding_checkpoint, extract_relation_embeddings
from .model import RelationModelConfig
from .module import (
    ARM_SPECS,
    RelationOptimizationConfig,
    RelationPretrainingModule,
    assert_final_z,
    build_relation_source_manifest,
    full_initialization_sha256,
    load_relation_checkpoint,
    validate_relation_source_manifest,
)


_IMMUTABLE_EPOCH_CHECKPOINT = re.compile(
    r"^epoch(?P<epoch>\d+)-step(?P<step>\d+)\.ckpt$"
)
SMOKE_SOURCE_PATHS = (
    "micoformer/relation_pretraining/model.py",
    "micoformer/relation_pretraining/mining.py",
    "micoformer/relation_pretraining/losses.py",
    "micoformer/relation_pretraining/data.py",
    "micoformer/relation_pretraining/module.py",
    "micoformer/relation_pretraining/workflow.py",
    "micoformer/relation_pretraining/extract.py",
    "micoformer/relation_pretraining/smoke.py",
    "scripts/2.train_relation.py",
)
SMOKE_REQUIRED_PASS_FIELDS = (
    "b32_launch_gate_passed",
    "real_50_step_gate_passed",
    "three_arm_gate_passed",
    "matched_trajectory_gate_passed",
    "resume_determinism_gate_passed",
    "resume_exact_gate_passed",
    "host_rss_gate_passed",
    "gpu_memory_gate_passed",
)
_SMOKE_ARM_ORDER = ("main_skip", "main_radius", "pma_skip")
_SMOKE_REAL_STEPS = 50
_SMOKE_GPU_LIMIT_BYTES = 32 * 1024**3
_SMOKE_HOST_RSS_LIMIT_BYTES = 45 * 1024**3
_SMOKE_RESUME_COMPONENTS = {
    "second_step_record",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "grad_scaler_state",
    "runtime_counters",
    "consumed_teacher_caches",
    "rng_after_second_step",
}


@dataclass(frozen=True)
class RelationRunConfig:
    h5ad_path: Path
    schedule_root: Path
    cache_root: Path
    output_root: Path
    arm_name: str
    disease_rows_path: Path
    smoke_dir: Path
    resume_checkpoint: Optional[Path] = None
    seed: int = 42
    num_workers: int = 0
    device_index: int = 0

    def __post_init__(self) -> None:
        if self.arm_name not in ARM_SPECS:
            raise ValueError(f"arm_name must be one of {sorted(ARM_SPECS)}")
        if self.seed != 42:
            raise ValueError("the frozen pilot requires seed=42")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")


def build_relation_module(
    *,
    arm_name: str,
    data_binding: Mapping[str, Any],
    seed: int = 42,
    model_config_overrides: Optional[Mapping[str, Any]] = None,
) -> RelationPretrainingModule:
    """Reset the initialization stream and build one exactly named arm."""

    if seed != 42:
        raise ValueError("the frozen pilot requires seed=42")
    if arm_name not in ARM_SPECS:
        raise ValueError(f"arm_name must be one of {sorted(ARM_SPECS)}")
    L.seed_everything(seed, workers=True)
    decoder_kind, _ = ARM_SPECS[arm_name]
    fields = dict(model_config_overrides or {})
    if "decoder_kind" in fields and fields["decoder_kind"] != decoder_kind:
        raise ValueError("model_config_overrides may not change the arm decoder")
    fields["decoder_kind"] = decoder_kind
    model_config = RelationModelConfig(**fields)
    return RelationPretrainingModule(
        arm_name=arm_name,
        model_config=model_config,
        data_binding=data_binding,
        optimization_config=RelationOptimizationConfig(),
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


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_smoke_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"smoke authorization {label} must be one regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"smoke authorization {label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"smoke authorization {label} root must be an object")
    return payload


def _require_smoke_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"smoke authorization {label} is not a lowercase SHA256")
    return value


def _require_smoke_true(payload: Mapping[str, Any], name: str, label: str) -> None:
    if payload.get(name) is not True:
        raise RuntimeError(f"smoke authorization {label}.{name} is not true")


def _validate_smoke_gpu_peak(payload: Any, label: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"smoke authorization {label} must be an object")
    _require_smoke_true(payload, "peak_reserved_strictly_below_limit", label)
    if payload.get("peak_reserved_limit_bytes") != _SMOKE_GPU_LIMIT_BYTES:
        raise RuntimeError(f"smoke authorization {label} GPU limit drifted")
    peak = payload.get("peak_reserved_bytes")
    if type(peak) is not int or peak < 0 or peak >= _SMOKE_GPU_LIMIT_BYTES:
        raise RuntimeError(f"smoke authorization {label} peak reserved memory is invalid")


def _validate_smoke_step(
    payload: Any,
    *,
    label: str,
    expected_index: int,
    expected_batch_size: int,
    expected_sequence_length: Optional[int] = None,
) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"smoke authorization {label} must be an object")
    if payload.get("scheduled_index") != expected_index:
        raise RuntimeError(f"smoke authorization {label} scheduled index drifted")
    if payload.get("schedule_batch_index") != expected_index:
        raise RuntimeError(f"smoke authorization {label} batch index drifted")
    if payload.get("batch_size") != expected_batch_size:
        raise RuntimeError(f"smoke authorization {label} batch size drifted")
    if expected_sequence_length is not None and (
        payload.get("dynamic_sequence_length") != expected_sequence_length
    ):
        raise RuntimeError(f"smoke authorization {label} sequence length drifted")
    if not isinstance(payload.get("has_relation_update"), bool):
        raise RuntimeError(f"smoke authorization {label} update status is malformed")
    _require_smoke_hash(payload.get("row_ids_sha256"), f"{label}.row_ids_sha256")
    loss = payload.get("loss")
    if not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
        raise RuntimeError(f"smoke authorization {label} loss is nonfinite")
    for lr_name in ("lr_before", "lr_after"):
        values = payload.get(lr_name)
        if (
            not isinstance(values, list)
            or not values
            or any(
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in values
            )
        ):
            raise RuntimeError(f"smoke authorization {label}.{lr_name} is malformed")
    counters = payload.get("runtime_counters")
    if not isinstance(counters, dict) or set(counters) != {
        "scheduled_batches",
        "optimizer_steps",
        "skipped_batches",
        "scheduler_steps",
    }:
        raise RuntimeError(f"smoke authorization {label} runtime counters are malformed")
    if any(type(counters[name]) is not int or counters[name] < 0 for name in counters):
        raise RuntimeError(f"smoke authorization {label} runtime counters are invalid")
    if counters["scheduled_batches"] != (
        counters["optimizer_steps"] + counters["skipped_batches"]
    ) or counters["scheduler_steps"] != counters["optimizer_steps"]:
        raise RuntimeError(f"smoke authorization {label} runtime counters violate invariants")
    if (
        counters["scheduled_batches"] != expected_index + 1
        or payload.get("optimizer_step_count") != counters["optimizer_steps"]
        or payload.get("scheduler_step_count") != counters["scheduler_steps"]
    ):
        raise RuntimeError(f"smoke authorization {label} step/runtime counters drifted")


def _validate_smoke_arm(arm_name: str, payload: Any) -> None:
    label = f"arms.{arm_name}"
    if not isinstance(payload, dict) or set(payload) != {
        "synthetic_worst_case",
        "real_first_50",
        "interrupted_resume",
        "single_process_peak_host_rss_bytes",
    }:
        raise RuntimeError(f"smoke authorization {label} schema drifted")
    host_peak = payload["single_process_peak_host_rss_bytes"]
    if type(host_peak) is not int or host_peak <= 0:
        raise RuntimeError(f"smoke authorization {label} host RSS is invalid")

    synthetic = payload["synthetic_worst_case"]
    if not isinstance(synthetic, dict) or synthetic.get("shape") != [32, 512]:
        raise RuntimeError(f"smoke authorization {label} synthetic B32x512 gate drifted")
    _validate_smoke_step(
        synthetic.get("step"),
        label=f"{label}.synthetic_worst_case.step",
        expected_index=0,
        expected_batch_size=32,
        expected_sequence_length=512,
    )
    if synthetic["step"].get("has_relation_update") is not True:
        raise RuntimeError(f"smoke authorization {label} synthetic gate did not update")
    if (
        synthetic["step"].get("optimizer_step_count") != 1
        or synthetic["step"].get("scheduler_step_count") != 1
        or synthetic.get("optimizer_state_finite") is not True
    ):
        raise RuntimeError(f"smoke authorization {label} synthetic optimizer gate failed")
    _validate_smoke_gpu_peak(synthetic, f"{label}.synthetic_worst_case")

    real = payload["real_first_50"]
    if not isinstance(real, dict) or real.get("scheduled_steps") != _SMOKE_REAL_STEPS:
        raise RuntimeError(f"smoke authorization {label} real 50-step gate drifted")
    steps = real.get("steps")
    if not isinstance(steps, list) or len(steps) != _SMOKE_REAL_STEPS:
        raise RuntimeError(f"smoke authorization {label} lacks exactly 50 real steps")
    for index, step in enumerate(steps):
        _validate_smoke_step(
            step,
            label=f"{label}.real_first_50.steps[{index}]",
            expected_index=index,
            expected_batch_size=32,
        )
    optimizer_steps = real.get("optimizer_steps")
    scheduler_steps = real.get("scheduler_steps")
    skipped_steps = real.get("skipped_steps")
    if (
        type(optimizer_steps) is not int
        or type(scheduler_steps) is not int
        or type(skipped_steps) is not int
        or optimizer_steps < 0
        or scheduler_steps != optimizer_steps
        or skipped_steps != _SMOKE_REAL_STEPS - optimizer_steps
        or real.get("optimizer_state_finite") is not True
    ):
        raise RuntimeError(f"smoke authorization {label} real-step counters are invalid")
    final_counters = steps[-1]["runtime_counters"]
    if (
        final_counters["scheduled_batches"] != _SMOKE_REAL_STEPS
        or final_counters["optimizer_steps"] != optimizer_steps
        or final_counters["scheduler_steps"] != scheduler_steps
        or final_counters["skipped_batches"] != skipped_steps
    ):
        raise RuntimeError(f"smoke authorization {label} real final counters drifted")
    initialization = real.get("initialization")
    if not isinstance(initialization, dict) or set(initialization) != {
        "shared_initialization_sha256",
        "full_initialization_sha256",
    }:
        raise RuntimeError(f"smoke authorization {label} initialization hashes are missing")
    for name, value in initialization.items():
        _require_smoke_hash(value, f"{label}.initialization.{name}")
    _validate_smoke_gpu_peak(real, f"{label}.real_first_50")

    resume = payload["interrupted_resume"]
    if not isinstance(resume, dict) or resume.get("arm_name") != arm_name:
        raise RuntimeError(f"smoke authorization {label} resume arm identity drifted")
    _require_smoke_true(resume, "passed", f"{label}.interrupted_resume")
    if resume.get("comparison") != "bitwise/exact; no tolerance relaxation":
        raise RuntimeError(f"smoke authorization {label} resume comparison is not exact")
    component_hashes = resume.get("component_sha256")
    if not isinstance(component_hashes, dict) or set(component_hashes) != _SMOKE_RESUME_COMPONENTS:
        raise RuntimeError(f"smoke authorization {label} resume components are incomplete")
    for name, value in component_hashes.items():
        _require_smoke_hash(value, f"{label}.interrupted_resume.component_sha256.{name}")
    _require_smoke_hash(
        resume.get("temporary_checkpoint_sha256"),
        f"{label}.interrupted_resume.temporary_checkpoint_sha256",
    )
    _validate_smoke_step(
        resume.get("first_step"),
        label=f"{label}.interrupted_resume.first_step",
        expected_index=0,
        expected_batch_size=32,
    )
    second_loss = resume.get("second_step_loss")
    resume_counters = resume.get("runtime_counters")
    if (
        not isinstance(second_loss, (int, float))
        or not math.isfinite(float(second_loss))
        or not isinstance(resume_counters, dict)
        or set(resume_counters)
        != {"scheduled_batches", "optimizer_steps", "skipped_batches", "scheduler_steps"}
        or any(type(value) is not int or value < 0 for value in resume_counters.values())
        or resume_counters["scheduled_batches"] != 2
        or resume_counters["scheduled_batches"]
        != resume_counters["optimizer_steps"] + resume_counters["skipped_batches"]
        or resume_counters["scheduler_steps"] != resume_counters["optimizer_steps"]
    ):
        raise RuntimeError(f"smoke authorization {label} resume second-step evidence drifted")
    if (
        resume.get("temporary_checkpoint_removed") is not True
        or resume.get("deterministic_algorithms") is not True
        or resume.get("cudnn_benchmark") is not False
        or resume.get("cublas_workspace_config") != ":4096:8"
    ):
        raise RuntimeError(f"smoke authorization {label} resume runtime contract drifted")
    _validate_smoke_gpu_peak(resume, f"{label}.interrupted_resume")


def _validate_smoke_matched_trajectories(arms: Mapping[str, Any]) -> None:
    reference = arms[_SMOKE_ARM_ORDER[0]]["real_first_50"]["steps"]
    fields = (
        "scheduled_index",
        "schedule_batch_index",
        "row_ids_sha256",
        "batch_size",
        "dynamic_sequence_length",
        "has_relation_update",
        "optimizer_step_count",
        "scheduler_step_count",
    )
    for arm_name in _SMOKE_ARM_ORDER[1:]:
        candidate = arms[arm_name]["real_first_50"]["steps"]
        for index, (left, right) in enumerate(zip(reference, candidate)):
            if any(left.get(name) != right.get(name) for name in fields):
                raise RuntimeError(
                    f"smoke authorization matched trajectory drifted at {arm_name} step {index}"
                )
            for lr_name in ("lr_before", "lr_after"):
                left_lr = left.get(lr_name)
                right_lr = right.get(lr_name)
                if len(left_lr) != len(right_lr) or any(
                    not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)
                    for a, b in zip(left_lr, right_lr)
                ):
                    raise RuntimeError(
                        f"smoke authorization matched LR drifted at {arm_name} step {index}"
                    )


def validate_smoke_launch_authorization(
    smoke_dir: str | os.PathLike[str],
    *,
    h5ad_path: str | os.PathLike[str],
    schedule_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the immutable smoke evidence required before any arm output exists."""

    directory = Path(smoke_dir).resolve()
    if not directory.is_dir():
        raise RuntimeError(f"required smoke authorization directory is missing: {directory}")
    manifest_path = directory / "smoke_manifest.json"
    complete_path = directory / ".complete"
    manifest = _load_smoke_json(manifest_path, "manifest")
    complete = _load_smoke_json(complete_path, "sentinel")
    manifest_sha256 = sha256_file(manifest_path)
    complete_sha256 = sha256_file(complete_path)

    expected_complete_keys = {
        "schema_version",
        "status",
        "manifest_sha256",
        "completed_at",
        *SMOKE_REQUIRED_PASS_FIELDS,
    }
    if set(complete) != expected_complete_keys:
        raise RuntimeError("smoke authorization sentinel schema drifted")
    if complete.get("schema_version") != 1 or complete.get("status") != "passed":
        raise RuntimeError("smoke authorization sentinel is not schema-v1 passed")
    if complete.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("smoke authorization sentinel manifest hash mismatch")
    for name in SMOKE_REQUIRED_PASS_FIELDS:
        _require_smoke_true(complete, name, "sentinel")

    if (
        manifest.get("schema_version") != 1
        or manifest.get("gate") != "relation_b32_real_cuda_launch_gate"
        or manifest.get("status") != "passed"
        or manifest.get("completed_at") != complete.get("completed_at")
    ):
        raise RuntimeError("smoke authorization manifest identity/status drifted")
    result = manifest.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("smoke authorization manifest result is missing")
    for name in SMOKE_REQUIRED_PASS_FIELDS:
        _require_smoke_true(result, name, "manifest.result")
    if (
        result.get("production_training_authorized_by_this_gate") is not True
        or result.get("b16_implemented_or_triggered") is not False
    ):
        raise RuntimeError("smoke authorization did not explicitly authorize B32 production")

    expected_inputs = {
        "h5ad_path": Path(h5ad_path).resolve(),
        "schedule_root": Path(schedule_root).resolve(),
        "cache_root": Path(cache_root).resolve(),
        "output_dir": directory,
    }
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != set(expected_inputs):
        raise RuntimeError("smoke authorization input-path schema drifted")
    for name, expected in expected_inputs.items():
        value = inputs.get(name)
        if not isinstance(value, str) or Path(value).resolve() != expected:
            raise RuntimeError(
                f"smoke authorization input path drifted for {name}: {value!r} != {expected}"
            )

    contract = manifest.get("contract")
    expected_contract = {
        "arms": list(_SMOKE_ARM_ORDER),
        "synthetic_shape": [32, 512],
        "real_schedule": "train_epoch_000 fixed first 50 batches",
        "physical_batch_size": 32,
        "peak_reserved_strict_limit_bytes": _SMOKE_GPU_LIMIT_BYTES,
        "aggregate_host_rss_strict_limit_bytes": _SMOKE_HOST_RSS_LIMIT_BYTES,
        "cuda_required": True,
        "cpu_fallback_allowed": False,
        "b16_implemented_or_triggered": False,
        "production_checkpoint_or_embedding_write": False,
    }
    if contract != expected_contract:
        raise RuntimeError("smoke authorization frozen B32 contract drifted")
    cuda = manifest.get("cuda")
    if not isinstance(cuda, dict) or cuda.get("cpu_fallback_allowed") is not False:
        raise RuntimeError("smoke authorization lacks a real-CUDA-only result")
    determinism = manifest.get("determinism")
    if (
        not isinstance(determinism, dict)
        or determinism.get("torch_deterministic_algorithms") is not True
        or determinism.get("cudnn_benchmark") is not False
        or determinism.get("cublas_workspace_config") != ":4096:8"
    ):
        raise RuntimeError("smoke authorization deterministic runtime drifted")

    matched_schedule = manifest.get("matched_schedule")
    if (
        not isinstance(matched_schedule, dict)
        or matched_schedule.get("schedule_kind") != "train"
        or matched_schedule.get("schedule_index") != 0
        or matched_schedule.get("steps") != _SMOKE_REAL_STEPS
        or not isinstance(matched_schedule.get("batch_row_id_sha256"), list)
        or len(matched_schedule["batch_row_id_sha256"]) != _SMOKE_REAL_STEPS
    ):
        raise RuntimeError("smoke authorization matched 50-batch schedule drifted")
    for index, value in enumerate(matched_schedule["batch_row_id_sha256"]):
        _require_smoke_hash(value, f"matched_schedule.batch_row_id_sha256[{index}]")

    arms = manifest.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(_SMOKE_ARM_ORDER):
        raise RuntimeError("smoke authorization must contain exactly the three frozen arms")
    for arm_name in _SMOKE_ARM_ORDER:
        _validate_smoke_arm(arm_name, arms[arm_name])
    shared_hashes = {
        arms[name]["real_first_50"]["initialization"]["shared_initialization_sha256"]
        for name in _SMOKE_ARM_ORDER
    }
    if len(shared_hashes) != 1:
        raise RuntimeError("smoke authorization three-arm shared initialization drifted")
    if (
        arms["main_skip"]["real_first_50"]["initialization"][
            "full_initialization_sha256"
        ]
        != arms["main_radius"]["real_first_50"]["initialization"][
            "full_initialization_sha256"
        ]
    ):
        raise RuntimeError("smoke authorization main-arm initialization drifted")
    _validate_smoke_matched_trajectories(arms)

    matched = manifest.get("matched_trajectory_gate")
    if (
        not isinstance(matched, dict)
        or matched.get("passed") is not True
        or matched.get("reference_arm") != _SMOKE_ARM_ORDER[0]
        or matched.get("scheduled_steps") != _SMOKE_REAL_STEPS
        or matched.get("optimizer_steps")
        != arms[_SMOKE_ARM_ORDER[0]]["real_first_50"]["optimizer_steps"]
        or matched.get("skipped_steps")
        != arms[_SMOKE_ARM_ORDER[0]]["real_first_50"]["skipped_steps"]
    ):
        raise RuntimeError("smoke authorization matched-trajectory gate drifted")

    host = manifest.get("host_rss_gate")
    peaks = host.get("per_single_arm_peak_bytes") if isinstance(host, dict) else None
    if not isinstance(peaks, dict) or set(peaks) != set(_SMOKE_ARM_ORDER):
        raise RuntimeError("smoke authorization host RSS arm schema drifted")
    normalized_peaks: dict[str, int] = {}
    for name in _SMOKE_ARM_ORDER:
        value = peaks[name]
        if type(value) is not int or value <= 0:
            raise RuntimeError("smoke authorization host RSS peak is invalid")
        if value != arms[name]["single_process_peak_host_rss_bytes"]:
            raise RuntimeError("smoke authorization host RSS arm evidence drifted")
        normalized_peaks[name] = value
    aggregate = sum(normalized_peaks.values())
    if (
        host.get("aggregate_limit_bytes") != _SMOKE_HOST_RSS_LIMIT_BYTES
        or host.get("aggregate_strictly_below_limit") is not True
        or host.get("conservative_aggregate_bytes") != aggregate
        or aggregate >= _SMOKE_HOST_RSS_LIMIT_BYTES
    ):
        raise RuntimeError("smoke authorization aggregate host RSS gate failed")

    repository = Path(__file__).resolve().parents[2]
    source_assets = manifest.get("source_assets")
    if not isinstance(source_assets, dict) or set(source_assets) != set(SMOKE_SOURCE_PATHS):
        raise RuntimeError("smoke authorization source asset set drifted")
    verified_sources: dict[str, str] = {}
    for relative in SMOKE_SOURCE_PATHS:
        record = source_assets[relative]
        expected_path = (repository / relative).resolve()
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise RuntimeError(f"smoke authorization source record drifted: {relative}")
        if Path(record.get("path", "")).resolve() != expected_path:
            raise RuntimeError(f"smoke authorization source path drifted: {relative}")
        recorded_hash = _require_smoke_hash(record.get("sha256"), f"source_assets.{relative}")
        if not expected_path.is_file() or sha256_file(expected_path) != recorded_hash:
            raise RuntimeError(f"smoke authorization source SHA256 drifted: {relative}")
        verified_sources[relative] = recorded_hash

    return {
        "schema_version": 1,
        "directory": str(directory),
        "manifest": {"path": str(manifest_path), "sha256": manifest_sha256},
        "complete": {"path": str(complete_path), "sha256": complete_sha256},
        "completed_at": manifest["completed_at"],
        "verified_inputs": {name: str(path) for name, path in expected_inputs.items()},
        "verified_pass_fields": list(SMOKE_REQUIRED_PASS_FIELDS),
        "source_assets": verified_sources,
    }


def _immutable_epoch_candidates(checkpoint_dir: Path) -> list[tuple[int, int, Path]]:
    candidates: list[tuple[int, int, Path]] = []
    for path in sorted(checkpoint_dir.glob("*.ckpt")):
        if path.name == "last.ckpt":
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"endpoint checkpoint must be one regular file: {path}")
        match = _IMMUTABLE_EPOCH_CHECKPOINT.fullmatch(path.name)
        if match is None:
            raise RuntimeError(
                "unexpected non-last checkpoint filename; immutable endpoints require "
                f"exact epochXX-stepYY.ckpt names: {path.name}"
            )
        candidates.append((int(match.group("epoch")), int(match.group("step")), path.resolve()))
    return candidates


def bind_immutable_epoch_endpoint(
    *,
    checkpoint_dir: str | os.PathLike[str],
    endpoint_manifest_path: str | os.PathLike[str],
    run_manifest_path: str | os.PathLike[str],
    expected_relation_contract: Mapping[str, Any],
    expected_data_binding: Mapping[str, Any],
    arm_name: str,
    endpoint_name: str,
    expected_epoch: Optional[int],
    require_amp_scaler: bool,
) -> Path:
    """Bind one endpoint to a unique immutable ``epochXX-stepYY.ckpt`` file."""

    checkpoint_dir = Path(checkpoint_dir).resolve()
    endpoint_manifest_path = Path(endpoint_manifest_path).resolve()
    run_manifest_path = Path(run_manifest_path).resolve()
    if not run_manifest_path.is_file():
        raise RuntimeError(f"endpoint run manifest is missing: {run_manifest_path}")
    if arm_name not in ARM_SPECS:
        raise ValueError(f"unknown relation arm: {arm_name!r}")
    if not endpoint_name or Path(endpoint_name).name != endpoint_name:
        raise ValueError(f"unsafe endpoint name: {endpoint_name!r}")
    source_manifest = validate_relation_source_manifest(
        expected_relation_contract.get("source_manifest")
    )
    try:
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"endpoint run manifest is unreadable: {run_manifest_path}") from exc
    if (
        not isinstance(run_manifest, dict)
        or run_manifest.get("schema_version") != 1
        or run_manifest.get("arm_name") != arm_name
        or run_manifest.get("source_manifest") != source_manifest
        or run_manifest.get("relation_contract") != dict(expected_relation_contract)
    ):
        raise RuntimeError("endpoint run manifest does not bind the exact arm/source/run contract")

    candidates = _immutable_epoch_candidates(checkpoint_dir)
    if not candidates:
        raise RuntimeError(f"no immutable epoch checkpoints found under {checkpoint_dir}")
    if expected_epoch is None:
        selected_epoch = max(epoch for epoch, _, _ in candidates)
    else:
        selected_epoch = int(expected_epoch)
    matching_epoch = [item for item in candidates if item[0] == selected_epoch]
    if not matching_epoch:
        raise RuntimeError(f"immutable checkpoint for epoch={selected_epoch} is missing")
    if len(matching_epoch) != 1:
        raise RuntimeError(
            "immutable endpoint checkpoint is ambiguous: "
            f"epoch={selected_epoch} matches={len(matching_epoch)}"
        )
    _, selected_step, checkpoint_path = matching_epoch[0]
    if checkpoint_path.name == "last.ckpt":
        raise RuntimeError("mutable last.ckpt may not define an endpoint")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("immutable endpoint checkpoint root is not a mapping")
    if payload.get("relation_contract") != dict(expected_relation_contract):
        raise RuntimeError("immutable endpoint checkpoint relation contract drifted")
    payload_epoch = payload.get("epoch")
    payload_global_step = payload.get("global_step")
    if type(payload_epoch) is not int or payload_epoch != selected_epoch:
        raise RuntimeError("immutable checkpoint filename/payload epoch mismatch")
    if type(payload_global_step) is not int or payload_global_step != selected_step:
        raise RuntimeError("immutable checkpoint filename/payload global_step mismatch")
    optimizer_states = payload.get("optimizer_states")
    scheduler_states = payload.get("lr_schedulers")
    if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
        raise RuntimeError("immutable checkpoint lacks the single AdamW state")
    if not isinstance(scheduler_states, list) or len(scheduler_states) != 1:
        raise RuntimeError("immutable checkpoint lacks the single warmup-cosine scheduler state")
    runtime_counters = payload.get("relation_runtime_counters")
    if not isinstance(runtime_counters, dict):
        raise RuntimeError("immutable checkpoint lacks relation runtime counters")
    expected_counter_keys = {
        "scheduled_batches",
        "optimizer_steps",
        "skipped_batches",
        "scheduler_steps",
        "current_lr",
    }
    if set(runtime_counters) != expected_counter_keys:
        raise RuntimeError("immutable checkpoint runtime counter schema drifted")
    for name in (
        "scheduled_batches",
        "optimizer_steps",
        "skipped_batches",
        "scheduler_steps",
    ):
        if type(runtime_counters[name]) is not int or runtime_counters[name] < 0:
            raise RuntimeError(f"immutable checkpoint runtime counter {name} is invalid")
    if (
        runtime_counters["scheduled_batches"]
        != runtime_counters["optimizer_steps"] + runtime_counters["skipped_batches"]
        or runtime_counters["scheduler_steps"] != runtime_counters["optimizer_steps"]
    ):
        raise RuntimeError("immutable checkpoint runtime counter invariants failed")
    if runtime_counters["optimizer_steps"] != selected_step:
        raise RuntimeError(
            "immutable checkpoint global_step differs from relation optimizer-step count"
        )
    scheduler_last_epoch = scheduler_states[0].get("last_epoch")
    if scheduler_last_epoch != runtime_counters["scheduler_steps"]:
        raise RuntimeError(
            "immutable checkpoint scheduler state differs from scheduler-step count"
        )
    current_lr = runtime_counters["current_lr"]
    if (
        not isinstance(current_lr, (int, float))
        or isinstance(current_lr, bool)
        or not math.isfinite(float(current_lr))
    ):
        raise RuntimeError("immutable endpoint checkpoint lacks a finite current LR")
    optimizer_groups = optimizer_states[0].get("param_groups")
    if not isinstance(optimizer_groups, list) or not optimizer_groups:
        raise RuntimeError("immutable endpoint optimizer state lacks parameter groups")
    optimizer_lrs = [
        group.get("lr") if isinstance(group, dict) else None
        for group in optimizer_groups
    ]
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in optimizer_lrs
    ) or any(float(value) != float(current_lr) for value in optimizer_lrs):
        raise RuntimeError("runtime current LR differs from the immutable optimizer state")
    amp_key_present = "MixedPrecision" in payload
    if require_amp_scaler and (
        not amp_key_present
        or not isinstance(payload["MixedPrecision"], dict)
        or not payload["MixedPrecision"]
    ):
        raise RuntimeError(
            "16-mixed immutable checkpoint lacks non-empty Lightning MixedPrecision state"
        )
    restored = load_relation_checkpoint(
        checkpoint_path,
        expected_data_binding=expected_data_binding,
        map_location="cpu",
        restore_rng_state=False,
    )
    if restored.relation_contract != dict(expected_relation_contract):
        raise RuntimeError("immutable endpoint failed strict relation checkpoint reconstruction")

    checkpoint_hash = sha256_file(checkpoint_path)
    record = {
        "schema_version": 1,
        "endpoint_kind": "frozen_relation_arm_completion",
        "endpoint_name": endpoint_name,
        "arm_name": arm_name,
        "architecture": ARM_SPECS[arm_name][0],
        "no_next_mode": ARM_SPECS[arm_name][1],
        "physical_batch_size": int(
            expected_relation_contract["optimization_config"]["physical_batch_size"]
        ),
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_hash,
            "epoch": selected_epoch,
            "global_step": selected_step,
            "immutable_filename": True,
            "last_ckpt_used": False,
        },
        "run_manifest": {
            "path": str(run_manifest_path),
            "sha256": sha256_file(run_manifest_path),
        },
        "source_manifest_sha256": expected_relation_contract["source_manifest"][
            "manifest_sha256"
        ],
        "relation_contract_sha256": _canonical_mapping_sha256(
            expected_relation_contract
        ),
        "resume_state": {
            "optimizer_states": len(optimizer_states),
            "scheduler_states": len(scheduler_states),
            "rng_state": "relation_rng_state",
            "amp_scaler_key": "MixedPrecision" if amp_key_present else None,
            "amp_equivalence_verification": (
                "real CUDA interrupted-resume smoke required"
            ),
        },
        "runtime_counters": runtime_counters,
    }
    if endpoint_manifest_path.exists():
        existing = json.loads(endpoint_manifest_path.read_text(encoding="utf-8"))
        if existing != record:
            raise RuntimeError("existing immutable endpoint manifest differs")
    else:
        _atomic_json(endpoint_manifest_path, record)
    if sha256_file(checkpoint_path) != checkpoint_hash:
        raise RuntimeError("immutable endpoint checkpoint changed while publishing its manifest")
    return checkpoint_path


def validate_matched_arm_completions(
    output_root: str | os.PathLike[str],
) -> Optional[Path]:
    """Publish a runtime-only matched gate once all three arm completions exist."""

    output_root = Path(output_root).resolve()
    paths = {
        arm: output_root / arm / "endpoints/arm_completion.json"
        for arm in sorted(ARM_SPECS)
    }
    if not all(path.is_file() for path in paths.values()):
        return None
    payloads = {
        arm: json.loads(path.read_text(encoding="utf-8"))
        for arm, path in paths.items()
    }
    counters: dict[str, dict[str, Any]] = {}
    for arm, payload in payloads.items():
        if (
            payload.get("schema_version") != 1
            or payload.get("endpoint_kind") != "frozen_relation_arm_completion"
            or payload.get("arm_name") != arm
            or payload.get("architecture") != ARM_SPECS[arm][0]
            or payload.get("no_next_mode") != ARM_SPECS[arm][1]
        ):
            raise RuntimeError(f"arm completion schema/identity drifted for {arm}")
        counters[arm] = payload.get("runtime_counters")
        if not isinstance(counters[arm], dict):
            raise RuntimeError(f"arm completion lacks runtime counters for {arm}")
    reference_arm = "main_skip"
    mismatched = {
        arm: value
        for arm, value in counters.items()
        if value != counters[reference_arm]
    }
    if mismatched:
        raise RuntimeError(
            "matched arm runtime counters/LR diverged; no common endpoint may be published: "
            f"reference={counters[reference_arm]} mismatched={mismatched}"
        )
    gate = {
        "schema_version": 1,
        "endpoint_kind": "matched_relation_runtime_gate",
        "physical_batch_size": payloads[reference_arm]["physical_batch_size"],
        "runtime_counters": counters[reference_arm],
        "arms": {
            arm: {
                "completion": {
                    "path": str(paths[arm].resolve()),
                    "sha256": sha256_file(paths[arm]),
                },
                "checkpoint": payloads[arm]["checkpoint"],
            }
            for arm in sorted(paths)
        },
    }
    gate_path = output_root / "endpoints/matched_runtime_gate.json"
    if gate_path.exists():
        if json.loads(gate_path.read_text(encoding="utf-8")) != gate:
            raise RuntimeError("existing matched runtime gate differs")
    else:
        _atomic_json(gate_path, gate)
    return gate_path


def _epoch0_payload(module: RelationPretrainingModule) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_kind": "relation_architecture_epoch0",
        "architecture": module.model_config.decoder_kind,
        "seed": 42,
        "model_config": asdict(module.model_config),
        "data_binding": module.data_binding,
        "shared_initialization_sha256": module.shared_initialization_sha256,
        "full_initialization_sha256": module.full_initialization_sha256,
        "model_state_dict": module.model.state_dict(),
    }


def _write_or_verify_epoch0(path: Path, module: RelationPretrainingModule) -> None:
    if not path.exists():
        _atomic_torch_save(path, _epoch0_payload(module))
    loaded = _load_embedding_checkpoint(path)
    if (
        loaded.model_config != module.model_config
        or loaded.data_binding != module.data_binding
        or full_initialization_sha256(loaded.model) != module.full_initialization_sha256
    ):
        raise RuntimeError(f"epoch0 artifact does not strictly match deterministic init: {path}")


def prepare_relation_initialization(
    *,
    h5ad_path: str | os.PathLike[str],
    schedule_root: str | os.PathLike[str],
    cache_root: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    disease_rows_path: str | os.PathLike[str],
    extraction_device: str | torch.device = "cuda:0",
    num_workers: int = 0,
) -> Path:
    """Single-process epoch0 publication and three-panel export for two architectures."""

    output_root = Path(output_root).resolve()
    data = RelationDataModule(
        h5ad_path=h5ad_path,
        schedule_root=schedule_root,
        cache_root=cache_root,
        num_workers=num_workers,
    )
    data.store.validate_all_caches()
    disease_rows = data.store.validate_disease_rows(disease_rows_path)
    main = build_relation_module(
        arm_name="main_skip", data_binding=data.checkpoint_binding, seed=42
    )
    pma = build_relation_module(
        arm_name="pma_skip", data_binding=data.checkpoint_binding, seed=42
    )
    if main.shared_initialization_sha256 != pma.shared_initialization_sha256:
        raise RuntimeError("main and PMA shared components differ at step0")
    source_manifest = validate_relation_source_manifest(
        main.relation_contract.get("source_manifest")
    )
    if pma.relation_contract.get("source_manifest") != source_manifest:
        raise RuntimeError("main and PMA initialization used different source manifests")
    artifacts = {
        "main": output_root / "epoch0/main.ckpt",
        "pma": output_root / "epoch0/pma.ckpt",
    }
    _write_or_verify_epoch0(artifacts["main"], main)
    _write_or_verify_epoch0(artifacts["pma"], pma)

    val_rows = data.store.split("val")[0]
    test_rows = data.store.split("test")[0]
    panels = {
        "relation_val": (val_rows, None),
        "relation_test": (test_rows, None),
        "disease": (disease_rows, disease_rows_path),
    }
    outputs: dict[str, dict[str, str]] = {}
    for architecture, checkpoint in artifacts.items():
        outputs[architecture] = {}
        for panel, (rows, source) in panels.items():
            destination = output_root / "embeddings/epoch0" / architecture / f"{panel}.npz"
            extract_relation_embeddings(
                checkpoint_path=checkpoint,
                h5ad_path=h5ad_path,
                row_ids=rows,
                row_source_path=source,
                output_path=destination,
                device=extraction_device,
                num_workers=num_workers,
            )
            outputs[architecture][panel] = sha256_file(destination)
    manifest = {
        "schema_version": 1,
        "seed": 42,
        "source_manifest": source_manifest,
        "shared_initialization_sha256": main.shared_initialization_sha256,
        "data_binding": data.checkpoint_binding,
        "disease_rows": {
            "path": str(Path(disease_rows_path).resolve()),
            "file_sha256": sha256_file(disease_rows_path),
            "array_sha256": data.store.split_manifest["array_sha256"]["u_eval_rows"],
            "count": int(disease_rows.size),
        },
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for name, path in artifacts.items()
        },
        "embedding_npz_sha256": outputs,
    }
    manifest_path = output_root / "epoch0/manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _cuda_preflight(device_index: int) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; relation training may not silently fall back to CPU")
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"device_index={device_index} but only {torch.cuda.device_count()} CUDA devices are visible"
        )
    device = torch.device("cuda", device_index)
    probe = torch.ones(8, device=device)
    if float((probe * probe).sum().item()) != 8.0:
        raise RuntimeError("CUDA arithmetic preflight failed")
    properties = torch.cuda.get_device_properties(device)
    return {
        "visible_device_count": torch.cuda.device_count(),
        "device_index": device_index,
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _student_output_preflight(
    data: RelationDataModule,
    module: RelationPretrainingModule,
) -> dict[str, Any]:
    """Prove the workflow consumes normalized ``output.z``, never ``z_raw``."""

    data.setup("fit")
    epoch = data.store.epoch_dataset("train", 0, data.datasets["train"])
    batch = epoch[0]
    if set(batch["teacher_distances"]) != {"protein", "unifrac"}:
        raise RuntimeError("workflow preflight did not load both exact teacher keys")
    was_training = module.training
    module.eval()
    with torch.no_grad():
        output = module.model(
            torch.as_tensor(batch["genus_ids"], dtype=torch.long),
            torch.as_tensor(batch["rclr"], dtype=torch.float32),
            torch.as_tensor(batch["padding_mask"], dtype=torch.bool),
        )
        assert_final_z(output.z)
    module.train(was_training)
    return {
        "source": "RelationModelOutput.z",
        "batch_size": int(output.z.shape[0]),
        "embedding_dim": int(output.z.shape[1]),
        "dtype": str(output.z.dtype),
        "finite": bool(torch.isfinite(output.z).all()),
        "row_norm_min": float(output.z.norm(dim=-1).min()),
        "row_norm_max": float(output.z.norm(dim=-1).max()),
        "teacher_keys": sorted(batch["teacher_distances"]),
    }


def run_relation_pretraining(config: RelationRunConfig) -> Path:
    """Validate all immutable inputs, then train exactly one single-GPU arm."""

    h5ad_path = config.h5ad_path.resolve()
    schedule_root = config.schedule_root.resolve()
    cache_root = config.cache_root.resolve()
    output_root = config.output_root.resolve()
    smoke_authorization = validate_smoke_launch_authorization(
        config.smoke_dir,
        h5ad_path=h5ad_path,
        schedule_root=schedule_root,
        cache_root=cache_root,
    )
    arm_dir = output_root / config.arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    data = RelationDataModule(
        h5ad_path=h5ad_path,
        schedule_root=schedule_root,
        cache_root=cache_root,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    # Validate every schedule/cache before allocating the long-running trainer.
    published_cache_records = data.store.validate_all_caches()
    disease_rows = data.store.validate_disease_rows(config.disease_rows_path)
    module = build_relation_module(
        arm_name=config.arm_name,
        data_binding=data.checkpoint_binding,
        seed=config.seed,
    )
    source_manifest = validate_relation_source_manifest(
        module.relation_contract.get("source_manifest")
    )
    initialization_manifest_path = output_root / "epoch0/manifest.json"
    if not initialization_manifest_path.exists():
        raise RuntimeError(
            "epoch0 preparation is missing; run prepare_relation_initialization once before arm jobs"
        )
    initialization_manifest = json.loads(
        initialization_manifest_path.read_text(encoding="utf-8")
    )
    if initialization_manifest.get("source_manifest") != source_manifest:
        raise RuntimeError("epoch0 initialization source manifest differs from live training code")
    architecture = module.model_config.decoder_kind
    artifact_record = initialization_manifest.get("artifacts", {}).get(architecture)
    if not isinstance(artifact_record, dict):
        raise RuntimeError(f"epoch0 manifest lacks the {architecture!r} architecture artifact")
    epoch0_artifact = Path(artifact_record["path"]).resolve()
    if sha256_file(epoch0_artifact) != artifact_record.get("sha256"):
        raise RuntimeError("epoch0 architecture artifact hash drifted")
    loaded_epoch0 = _load_embedding_checkpoint(epoch0_artifact)
    if (
        loaded_epoch0.data_binding != data.checkpoint_binding
        or full_initialization_sha256(loaded_epoch0.model)
        != module.full_initialization_sha256
        or initialization_manifest.get("shared_initialization_sha256")
        != module.shared_initialization_sha256
    ):
        raise RuntimeError("arm initialization does not match the single-process epoch0 publication")
    incompatible = module.model.load_state_dict(loaded_epoch0.model.state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("arm failed to load the published architecture-specific epoch0 state")
    student_output_preflight = _student_output_preflight(data, module)

    resume_path: Optional[str] = None
    if config.resume_checkpoint is not None:
        resume = config.resume_checkpoint.resolve()
        restored = load_relation_checkpoint(
            resume,
            expected_data_binding=data.checkpoint_binding,
            map_location="cpu",
        )
        if restored.relation_contract != module.relation_contract:
            raise RuntimeError("resume checkpoint arm/model/runtime contract drifted")
        data.store.verify_cache_provenance(restored.consumed_teacher_caches)
        resume_path = str(resume)

    cuda = _cuda_preflight(config.device_index)
    run_manifest = {
        "schema_version": 1,
        "arm_name": config.arm_name,
        "seed": config.seed,
        "source_manifest": source_manifest,
        "relation_contract": module.relation_contract,
        "epoch0_initialization": {
            "manifest_path": str(initialization_manifest_path),
            "manifest_sha256": sha256_file(initialization_manifest_path),
            "architecture_artifact": str(epoch0_artifact),
            "architecture_artifact_sha256": sha256_file(epoch0_artifact),
        },
        "shared_initialization_sha256": module.shared_initialization_sha256,
        "full_initialization_sha256": module.full_initialization_sha256,
        "inputs": {
            "h5ad_path": str(h5ad_path),
            "schedule_root": str(schedule_root),
            "cache_root": str(cache_root),
        },
        "smoke_launch_authorization": smoke_authorization,
        "published_teacher_caches": published_cache_records,
        "resume_checkpoint": resume_path,
        "cuda_preflight": cuda,
        "student_output_preflight": student_output_preflight,
        "disease_rows": {
            "path": str(config.disease_rows_path.resolve()),
            "file_sha256": sha256_file(config.disease_rows_path),
            "array_sha256": sha256_array(disease_rows),
            "count": int(disease_rows.size),
        },
    }
    manifest_path = arm_dir / "run_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_immutable = {key: value for key, value in existing.items() if key != "resume_checkpoint"}
        requested_immutable = {
            key: value for key, value in run_manifest.items() if key != "resume_checkpoint"
        }
        if existing_immutable != requested_immutable:
            raise RuntimeError("existing arm run_manifest differs; use a new output root")
    else:
        _atomic_json(manifest_path, run_manifest)
    if resume_path is not None:
        resume_sha = sha256_file(resume_path)
        resume_record = {
            "schema_version": 1,
            "arm_name": config.arm_name,
            "checkpoint": {"path": resume_path, "sha256": resume_sha},
            "run_manifest_sha256": sha256_file(manifest_path),
            "data_binding": data.checkpoint_binding,
            "source_manifest_sha256": source_manifest["manifest_sha256"],
        }
        resume_record_path = arm_dir / "resume" / f"{resume_sha}.json"
        if resume_record_path.exists():
            if json.loads(resume_record_path.read_text(encoding="utf-8")) != resume_record:
                raise RuntimeError("existing resume record differs from the strict resume request")
        else:
            _atomic_json(resume_record_path, resume_record)

    if (
        validate_smoke_launch_authorization(
            config.smoke_dir,
            h5ad_path=h5ad_path,
            schedule_root=schedule_root,
            cache_root=cache_root,
        )
        != smoke_authorization
    ):
        raise RuntimeError("smoke authorization changed during arm preflight")

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
    immutable_checkpoint = bind_immutable_epoch_endpoint(
        checkpoint_dir=arm_dir / "checkpoints",
        endpoint_manifest_path=arm_dir / "endpoints/arm_completion.json",
        run_manifest_path=manifest_path,
        expected_relation_contract=module.relation_contract,
        expected_data_binding=data.checkpoint_binding,
        arm_name=config.arm_name,
        endpoint_name="arm_completion",
        expected_epoch=module.optimization_config.max_epochs - 1,
        require_amp_scaler=True,
    )
    validate_matched_arm_completions(output_root)
    val_rows = data.store.split("val")[0]
    test_rows = data.store.split("test")[0]
    module.to("cpu")
    torch.cuda.empty_cache()
    for panel, rows, source in (
        ("relation_val", val_rows, None),
        ("relation_test", test_rows, None),
        ("disease", disease_rows, config.disease_rows_path),
    ):
        extract_relation_embeddings(
            checkpoint_path=immutable_checkpoint,
            h5ad_path=h5ad_path,
            row_ids=rows,
            row_source_path=source,
            output_path=output_root / "embeddings/final" / config.arm_name / f"{panel}.npz",
            device=f"cuda:{config.device_index}",
            num_workers=config.num_workers,
        )
    return immutable_checkpoint
