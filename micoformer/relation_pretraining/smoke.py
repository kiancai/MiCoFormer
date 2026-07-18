"""Real-CUDA launch gate for the frozen relation-only B32 pilot.

This module is deliberately separate from the production training workflow.  It
does not save checkpoints or embeddings.  A successful run publishes one
hash-bound manifest and a ``.complete`` sentinel; every failure publishes a
failed manifest without the sentinel and stops before production training.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import tempfile
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .data import RelationDataModule, sha256_array, sha256_file
from .module import (
    RelationPretrainingModule,
    capture_relation_rng_state,
)
from .workflow import (
    SMOKE_REQUIRED_PASS_FIELDS,
    SMOKE_SOURCE_PATHS,
    build_relation_module,
)


SCHEMA_VERSION = 1
ARM_ORDER = ("main_skip", "main_radius", "pma_skip")
MATCHED_REAL_STEPS = 50
SYNTHETIC_BATCH_SIZE = 32
SYNTHETIC_SEQUENCE_LENGTH = 512
PEAK_RESERVED_LIMIT_BYTES = 32 * 1024**3
AGGREGATE_HOST_RSS_LIMIT_BYTES = 45 * 1024**3
GIB = 1024**3


class RelationSmokeError(RuntimeError):
    """The B32 CUDA gate failed and production training must not start."""


class MatchedTrajectoryError(RelationSmokeError):
    """The three arms did not preserve the frozen matched-step contract."""


class PeakMemoryGateError(RelationSmokeError):
    """A B32 arm reached or exceeded the 32-GiB reserved-memory gate."""


class ResumeDeterminismError(RelationSmokeError):
    """Interrupted CUDA continuation differs from the uninterrupted trajectory."""


@dataclass(frozen=True)
class RelationSmokeConfig:
    h5ad_path: Path
    schedule_root: Path
    cache_root: Path
    output_dir: Path
    device_index: int = 0

    def __post_init__(self) -> None:
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prepare_output_directory(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.exists():
        existing = list(resolved.iterdir())
        if existing:
            raise RelationSmokeError(
                f"smoke output directory must be fresh and empty: {resolved}"
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def publish_smoke_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    passed: bool,
) -> Path:
    """Atomically publish a smoke result and, only on success, ``.complete``."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "smoke_manifest.json"
    complete_path = output_dir / ".complete"
    if complete_path.exists():
        complete_path.unlink()
    _atomic_json(manifest_path, manifest)
    if passed:
        if manifest.get("status") != "passed":
            raise ValueError("a passed smoke publication requires status='passed'")
        result = manifest.get("result")
        if not isinstance(result, dict) or any(
            result.get(name) is not True for name in SMOKE_REQUIRED_PASS_FIELDS
        ):
            raise ValueError(
                "a passed smoke publication requires every B32/50-step/three-arm/"
                "matched/resume/RSS/GPU-memory gate to pass"
            )
        complete = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "manifest_sha256": sha256_file(manifest_path),
            "completed_at": manifest.get("completed_at"),
            **{name: True for name in SMOKE_REQUIRED_PASS_FIELDS},
        }
        _atomic_json(complete_path, complete)
    return manifest_path


def require_real_cuda(device_index: int) -> tuple[torch.device, dict[str, Any]]:
    """Select one visible CUDA device; never permit a CPU fallback."""

    if not torch.cuda.is_available():
        raise RelationSmokeError(
            "CUDA is unavailable; the relation launch gate forbids CPU fallback"
        )
    visible = int(torch.cuda.device_count())
    if device_index < 0 or device_index >= visible:
        raise RelationSmokeError(
            f"device_index={device_index}, but only {visible} CUDA devices are visible"
        )
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    probe = torch.arange(8, device=device, dtype=torch.float32)
    if float(probe.square().sum().item()) != 140.0:
        raise RelationSmokeError("CUDA arithmetic preflight failed")
    properties = torch.cuda.get_device_properties(device)
    return device, {
        "visible_device_count": visible,
        "device_index": int(device_index),
        "device_name": properties.name,
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "amp_dtype": "float16",
        "grad_scaler": "torch.amp.GradScaler('cuda')",
        "cpu_fallback_allowed": False,
    }


def enable_production_determinism() -> dict[str, Any]:
    """Mirror ``Trainer(deterministic=True)`` before any model CUDA work."""

    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Lightning sets this exact value for deterministic CUDA BLAS execution.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return {
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "production_reference": "Lightning Trainer(deterministic=True)",
    }


def _current_rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if statm.exists():
        fields = statm.read_text(encoding="ascii").split()
        if len(fields) >= 2:
            return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    # Linux is the production platform.  This fallback keeps pure-logic tests
    # portable while remaining conservative on platforms that report KiB.
    import resource

    maximum = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return maximum * 1024


class _HostRssSampler:
    def __init__(self, interval_seconds: float = 0.02) -> None:
        self.interval_seconds = float(interval_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_bytes = _current_rss_bytes()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())

    def __enter__(self) -> "_HostRssSampler":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.peak_bytes = max(self.peak_bytes, _current_rss_bytes())
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class _LogCollector:
    """Capture the counter values that the production module sends to Lightning."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}

    def reset(self) -> None:
        self.values.clear()

    def __call__(self, name: str, value: Any, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        if isinstance(value, Tensor):
            if value.numel() != 1:
                return
            scalar = float(value.detach().double().cpu().item())
        else:
            scalar = float(value)
        if math.isfinite(scalar):
            self.values[str(name)] = scalar


def _integer_metric(values: Mapping[str, float], name: str) -> int:
    if name not in values:
        raise RelationSmokeError(f"production step did not emit required counter {name!r}")
    value = values[name]
    rounded = int(round(value))
    if not math.isclose(value, float(rounded), rel_tol=0.0, abs_tol=1e-6):
        raise RelationSmokeError(f"counter {name!r} is not integer-valued: {value}")
    return rounded


def _validate_and_extract_counters(values: Mapping[str, float]) -> dict[str, int]:
    prefix = "train/"
    names = {
        "mining_teacher_anchor_total": "mining/all/teacher_anchor_total",
        "mining_status_total": "mining/all/status_total",
        "mining_valid_relation": "mining/all/valid_relation",
        "mining_skipped_relation": "mining/all/skipped_relation",
        "objective_teacher_anchor_total": "objective/all/teacher_anchor_total",
        "objective_valid_anchor": "objective/all/valid_anchor",
        "objective_skipped_anchor": "objective/all/skipped_anchor",
        "objective_active_hinge": "objective/all/active_hinge",
        "objective_inactive_hinge": "objective/all/inactive_hinge",
    }
    counters = {
        output_name: _integer_metric(values, prefix + logged_name)
        for output_name, logged_name in names.items()
    }
    if counters["mining_status_total"] != counters["mining_teacher_anchor_total"]:
        raise RelationSmokeError("mining status counter conservation failed")
    if (
        counters["mining_valid_relation"] + counters["mining_skipped_relation"]
        != counters["mining_teacher_anchor_total"]
    ):
        raise RelationSmokeError("mining valid/skip counter conservation failed")
    if (
        counters["objective_valid_anchor"] + counters["objective_skipped_anchor"]
        != counters["objective_teacher_anchor_total"]
    ):
        raise RelationSmokeError("objective valid/skip counter conservation failed")
    if (
        counters["objective_active_hinge"] + counters["objective_inactive_hinge"]
        != counters["objective_valid_anchor"]
    ):
        raise RelationSmokeError("objective active/inactive counter conservation failed")
    return counters


def build_synthetic_b32_l512_batch() -> dict[str, Any]:
    """Create a deterministic, full-support B32x512 batch for CUDA stress."""

    batch_size = SYNTHETIC_BATCH_SIZE
    length = SYNTHETIC_SEQUENCE_LENGTH
    positions = np.arange(length, dtype=np.int64)
    genus_ids = np.stack(
        [((positions + 97 * row) % 8_114) + 2 for row in range(batch_size)]
    ).astype(np.int64, copy=False)
    phase = np.arange(length, dtype=np.float64)[None, :] / 17.0
    row_phase = np.arange(batch_size, dtype=np.float64)[:, None] / 11.0
    rclr64 = np.sin(phase + row_phase) + 0.25 * np.cos(phase * 0.37 - row_phase)
    rclr64 -= rclr64.mean(axis=1, keepdims=True)
    rclr = rclr64.astype(np.float32)

    coordinate = np.arange(batch_size, dtype=np.float64)
    protein = np.abs(coordinate[:, None] - coordinate[None, :]) / (batch_size - 1)
    permuted = ((np.arange(batch_size, dtype=np.int64) * 7) % batch_size).astype(
        np.float64
    )
    unifrac = np.abs(permuted[:, None] - permuted[None, :]) / (batch_size - 1)
    validity = ~np.eye(batch_size, dtype=bool)
    return {
        "genus_ids": genus_ids,
        "rclr": rclr,
        "padding_mask": np.zeros((batch_size, length), dtype=bool),
        "row_ids": np.arange(10_000_000, 10_000_000 + batch_size, dtype=np.int64),
        "project_ids": np.arange(batch_size, dtype=np.int64),
        "site_ids": (np.arange(batch_size, dtype=np.int64) % 12),
        "teacher_distances": {
            "protein": protein.astype(np.float64, copy=False),
            "unifrac": unifrac.astype(np.float64, copy=False),
        },
        "teacher_validity": {
            "protein": validity.copy(),
            "unifrac": validity.copy(),
        },
        "protein_valid_mass": np.ones(batch_size, dtype=np.float64),
        "protein_borrowed_mass": np.zeros(batch_size, dtype=np.float64),
        "protein_endpoint_valid": np.ones(batch_size, dtype=bool),
        "schedule_kind": "synthetic_worst_case",
        "schedule_index": np.int64(0),
        "schedule_batch_index": np.int64(0),
        "cache_manifest_sha256": "a" * 64,
        "cache_sha256": "b" * 64,
        "schedule_file_sha256": "c" * 64,
    }


def _configure_smoke_optimizer(
    module: RelationPretrainingModule,
    *,
    estimated_total_steps: int,
) -> tuple[torch.optim.Optimizer, Any]:
    if estimated_total_steps <= 0:
        raise ValueError("estimated_total_steps must be positive")
    module._trainer = SimpleNamespace(estimated_stepping_batches=estimated_total_steps)
    try:
        configured = module.configure_optimizers()
    finally:
        module._trainer = None
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    return optimizer, scheduler


def _all_finite(tensors: Sequence[Tensor]) -> bool:
    for tensor in tensors:
        if tensor.numel() and not bool(torch.isfinite(tensor).all().item()):
            return False
    return True


def _optimizer_state_tensors(optimizer: torch.optim.Optimizer) -> list[Tensor]:
    tensors: list[Tensor] = []
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor):
                tensors.append(value)
    return tensors


def _run_amp_step(
    *,
    module: RelationPretrainingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    batch: Mapping[str, Any],
    collector: _LogCollector,
    scheduled_index: int,
    optimizer_step_count: int,
    scheduler_step_count: int,
) -> tuple[dict[str, Any], int, int]:
    runtime_before = module.relation_runtime_counts
    if runtime_before["optimizer_steps"] != optimizer_step_count:
        raise RelationSmokeError(
            "smoke optimizer counter differs from production runtime buffer before step"
        )
    if runtime_before["scheduler_steps"] != scheduler_step_count:
        raise RelationSmokeError(
            "smoke scheduler counter differs from production runtime buffer before step"
        )
    module._relation_scheduled_batch_count.add_(1)
    collector.reset()
    lr_before = [float(group["lr"]) for group in optimizer.param_groups]
    scale_before = float(scaler.get_scale())
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        result = module._shared_step(batch, "train")
    if not torch.isfinite(result.loss):
        raise RelationSmokeError(f"nonfinite loss at scheduled step {scheduled_index}")
    counters = _validate_and_extract_counters(collector.values)

    if result.has_relation_update:
        optimizer.zero_grad()
        scaler.scale(result.loss).backward()
        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad
            for parameter in module.model.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RelationSmokeError(
                f"relation update at scheduled step {scheduled_index} produced no gradients"
            )
        if not _all_finite(gradients):
            raise RelationSmokeError(
                f"nonfinite unscaled gradient at scheduled step {scheduled_index}"
            )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        module._relation_scheduler_step_count.add_(1)
        module._relation_optimizer_step_count.add_(1)
        optimizer_step_count += 1
        scheduler_step_count += 1
    else:
        module._relation_skipped_batch_count.add_(1)
    module._assert_runtime_counter_invariants()

    lr_after = [float(group["lr"]) for group in optimizer.param_groups]
    if not all(math.isfinite(value) and value >= 0.0 for value in lr_after):
        raise RelationSmokeError(f"invalid learning rate at scheduled step {scheduled_index}")
    if not _all_finite(list(module.model.parameters())):
        raise RelationSmokeError(
            f"nonfinite model parameter after scheduled step {scheduled_index}"
        )
    row_ids = np.asarray(batch["row_ids"], dtype=np.int64)
    record = {
        "scheduled_index": int(scheduled_index),
        "schedule_batch_index": int(batch["schedule_batch_index"]),
        "row_ids_sha256": sha256_array(row_ids),
        "batch_size": int(row_ids.size),
        "dynamic_sequence_length": int(np.asarray(batch["genus_ids"]).shape[1]),
        "has_relation_update": bool(result.has_relation_update),
        "optimizer_step_count": int(optimizer_step_count),
        "scheduler_step_count": int(scheduler_step_count),
        "lr_before": lr_before,
        "lr_after": lr_after,
        "grad_scaler_scale_before": scale_before,
        "grad_scaler_scale_after": float(scaler.get_scale()),
        "loss": float(result.loss.detach().float().cpu().item()),
        "counters": counters,
        "runtime_counters": module.relation_runtime_counts,
    }
    return record, optimizer_step_count, scheduler_step_count


def _new_smoke_module(
    *,
    arm_name: str,
    data_binding: Mapping[str, Any],
    device: torch.device,
) -> tuple[RelationPretrainingModule, _LogCollector]:
    module = build_relation_module(
        arm_name=arm_name,
        data_binding=data_binding,
        seed=42,
    )
    collector = _LogCollector()
    module.log = collector  # type: ignore[method-assign]
    module.to(device).train()
    torch.cuda.manual_seed_all(42)
    return module, collector


def _release_cuda_objects(*objects: Any) -> None:
    for value in objects:
        del value
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _clone_to_cpu(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _clone_to_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_clone_to_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_clone_to_cpu(child) for child in value)
    return value


def _update_nested_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray\0")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.view(np.uint8))
        return
    if isinstance(value, dict):
        digest.update(b"dict\0")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_nested_digest(digest, key)
            _update_nested_digest(digest, value[key])
        return
    if isinstance(value, list):
        digest.update(b"list\0")
        for child in value:
            _update_nested_digest(digest, child)
        return
    if isinstance(value, tuple):
        digest.update(b"tuple\0")
        for child in value:
            _update_nested_digest(digest, child)
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(b"\0")
    digest.update(repr(value).encode("utf-8"))
    digest.update(b"\0")


def _nested_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _update_nested_digest(digest, value)
    return digest.hexdigest()


def _assert_nested_exact(left: Any, right: Any, *, path: str) -> None:
    if type(left) is not type(right):
        raise ResumeDeterminismError(
            f"resume mismatch at {path}: type {type(left).__name__} != {type(right).__name__}"
        )
    if isinstance(left, Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            raise ResumeDeterminismError(f"resume tensor mismatch at {path}")
        return
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise ResumeDeterminismError(f"resume ndarray mismatch at {path}")
        return
    if isinstance(left, dict):
        if set(left) != set(right):
            raise ResumeDeterminismError(f"resume mapping keys mismatch at {path}")
        for key in sorted(left, key=lambda item: repr(item)):
            _assert_nested_exact(left[key], right[key], path=f"{path}.{key}")
        return
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ResumeDeterminismError(f"resume sequence length mismatch at {path}")
        for index, (left_child, right_child) in enumerate(zip(left, right)):
            _assert_nested_exact(
                left_child,
                right_child,
                path=f"{path}[{index}]",
            )
        return
    if left != right:
        raise ResumeDeterminismError(
            f"resume scalar mismatch at {path}: {left!r} != {right!r}"
        )


def _resume_snapshot(
    *,
    module: RelationPretrainingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
    second_step_record: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "second_step_record": _clone_to_cpu(dict(second_step_record)),
        "model_state": _clone_to_cpu(module.model.state_dict()),
        "optimizer_state": _clone_to_cpu(optimizer.state_dict()),
        "scheduler_state": _clone_to_cpu(scheduler.state_dict()),
        "grad_scaler_state": _clone_to_cpu(scaler.state_dict()),
        "runtime_counters": dict(module.relation_runtime_counts),
        "consumed_teacher_caches": _clone_to_cpu(module.consumed_teacher_caches),
        "rng_after_second_step": _clone_to_cpu(capture_relation_rng_state()),
    }


def _compare_resume_snapshots(
    continuous: Mapping[str, Any],
    resumed: Mapping[str, Any],
) -> dict[str, Any]:
    compared = (
        "second_step_record",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "grad_scaler_state",
        "runtime_counters",
        "consumed_teacher_caches",
        "rng_after_second_step",
    )
    hashes: dict[str, str] = {}
    for name in compared:
        _assert_nested_exact(continuous[name], resumed[name], path=name)
        left_hash = _nested_sha256(continuous[name])
        right_hash = _nested_sha256(resumed[name])
        if left_hash != right_hash:
            raise ResumeDeterminismError(
                f"resume canonical hash mismatch at {name} despite exact comparison"
            )
        hashes[name] = left_hash
    return {
        "passed": True,
        "comparison": "bitwise/exact; no tolerance relaxation",
        "component_sha256": hashes,
        "second_step_loss": float(continuous["second_step_record"]["loss"]),
        "runtime_counters": continuous["runtime_counters"],
        "checkpoint_boundary": (
            "production RelationPretrainingModule.on_save_checkpoint and on_load_checkpoint; "
            "strict module state reload; AdamW/LambdaLR/MixedPrecision GradScaler state reload; "
            "production on_train_start RNG restoration"
        ),
    }


def _make_interrupted_checkpoint(
    *,
    module: RelationPretrainingModule,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    checkpoint: dict[str, Any] = {
        "epoch": 0,
        "global_step": module.relation_optimizer_step_count,
        "state_dict": module.state_dict(),
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()],
        # This is the exact key written by Lightning's MixedPrecision plugin.
        "MixedPrecision": scaler.state_dict(),
    }
    module.on_save_checkpoint(checkpoint)
    return checkpoint


def _restore_interrupted_runtime(
    checkpoint: Mapping[str, Any],
    *,
    arm_name: str,
    data_binding: Mapping[str, Any],
    device: torch.device,
    estimated_total_steps: int,
) -> tuple[
    RelationPretrainingModule,
    torch.optim.Optimizer,
    Any,
    torch.amp.GradScaler,
    _LogCollector,
]:
    module = build_relation_module(
        arm_name=arm_name,
        data_binding=data_binding,
        seed=42,
    )
    # Mirror Lightning's production metadata hook rather than bypassing it with
    # a weights-only reconstruction.
    module.on_load_checkpoint(dict(checkpoint))
    incompatible = module.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ResumeDeterminismError("strict interrupted module reload was incompatible")
    module.to(device).train()
    collector = _LogCollector()
    module.log = collector  # type: ignore[method-assign]
    optimizer, scheduler = _configure_smoke_optimizer(
        module,
        estimated_total_steps=estimated_total_steps,
    )
    optimizer.load_state_dict(checkpoint["optimizer_states"][0])
    scheduler.load_state_dict(checkpoint["lr_schedulers"][0])
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    scaler.load_state_dict(checkpoint["MixedPrecision"])
    module.on_train_start()
    return module, optimizer, scheduler, scaler, collector


def _run_resume_determinism_gate(
    *,
    arm_name: str,
    data_binding: Mapping[str, Any],
    device: torch.device,
    first_batch: Mapping[str, Any],
    second_batch: Mapping[str, Any],
    estimated_total_steps: int,
    scratch_root: Path,
) -> dict[str, Any]:
    _release_cuda_objects()
    module, collector = _new_smoke_module(
        arm_name=arm_name,
        data_binding=data_binding,
        device=device,
    )
    optimizer, scheduler = _configure_smoke_optimizer(
        module,
        estimated_total_steps=estimated_total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    torch.cuda.reset_peak_memory_stats(device)
    first_record, optimizer_steps, scheduler_steps = _run_amp_step(
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        batch=first_batch,
        collector=collector,
        scheduled_index=0,
        optimizer_step_count=0,
        scheduler_step_count=0,
    )
    if not first_record["has_relation_update"]:
        raise ResumeDeterminismError(
            f"{arm_name} first resume-smoke batch did not exercise optimizer/scaler state"
        )
    checkpoint = _make_interrupted_checkpoint(
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix=f"resume_{arm_name}_") as directory:
        checkpoint_path = Path(directory) / "interrupted.ckpt"
        torch.save(checkpoint, checkpoint_path)
        checkpoint_sha256 = sha256_file(checkpoint_path)
        del checkpoint

        continuous_record, optimizer_steps, scheduler_steps = _run_amp_step(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            batch=second_batch,
            collector=collector,
            scheduled_index=1,
            optimizer_step_count=optimizer_steps,
            scheduler_step_count=scheduler_steps,
        )
        continuous = _resume_snapshot(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            second_step_record=continuous_record,
        )
        del module, optimizer, scheduler, scaler, collector
        _release_cuda_objects()

        restored_checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        module, optimizer, scheduler, scaler, collector = _restore_interrupted_runtime(
            restored_checkpoint,
            arm_name=arm_name,
            data_binding=data_binding,
            device=device,
            estimated_total_steps=estimated_total_steps,
        )
        restored_counts = module.relation_runtime_counts
        resumed_record, resumed_optimizer_steps, resumed_scheduler_steps = _run_amp_step(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            batch=second_batch,
            collector=collector,
            scheduled_index=1,
            optimizer_step_count=restored_counts["optimizer_steps"],
            scheduler_step_count=restored_counts["scheduler_steps"],
        )
        resumed = _resume_snapshot(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            second_step_record=resumed_record,
        )
        comparison = _compare_resume_snapshots(continuous, resumed)
        if optimizer_steps != resumed_optimizer_steps or scheduler_steps != resumed_scheduler_steps:
            raise ResumeDeterminismError(
                "continuous and resumed optimizer/scheduler counts differ after step two"
            )
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if peak_reserved >= PEAK_RESERVED_LIMIT_BYTES:
            raise PeakMemoryGateError(
                f"resume {arm_name} peak reserved {peak_reserved / GIB:.3f} GiB "
                "is not below 32 GiB"
            )
        del restored_checkpoint, module, optimizer, scheduler, scaler, collector
        _release_cuda_objects()
    if any(scratch_root.glob(f"resume_{arm_name}_*")):
        raise ResumeDeterminismError("temporary interrupted checkpoint directory was not removed")
    return {
        **comparison,
        "arm_name": arm_name,
        "first_step": first_record,
        "temporary_checkpoint_sha256": checkpoint_sha256,
        "temporary_checkpoint_removed": True,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": peak_reserved / GIB,
        "peak_reserved_limit_bytes": PEAK_RESERVED_LIMIT_BYTES,
        "peak_reserved_strictly_below_limit": True,
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _run_synthetic_gate(
    *,
    arm_name: str,
    data_binding: Mapping[str, Any],
    device: torch.device,
    estimated_total_steps: int,
) -> dict[str, Any]:
    _release_cuda_objects()
    module, collector = _new_smoke_module(
        arm_name=arm_name,
        data_binding=data_binding,
        device=device,
    )
    optimizer, scheduler = _configure_smoke_optimizer(
        module,
        estimated_total_steps=estimated_total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    batch = build_synthetic_b32_l512_batch()
    torch.cuda.reset_peak_memory_stats(device)
    record, optimizer_steps, scheduler_steps = _run_amp_step(
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        batch=batch,
        collector=collector,
        scheduled_index=0,
        optimizer_step_count=0,
        scheduler_step_count=0,
    )
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    optimizer_state_finite = _all_finite(_optimizer_state_tensors(optimizer))
    if not optimizer_state_finite:
        raise RelationSmokeError(f"synthetic {arm_name} optimizer state is nonfinite")
    if optimizer_steps != 1 or scheduler_steps != 1:
        raise RelationSmokeError(
            f"synthetic {arm_name} did not exercise one optimizer/scheduler step"
        )
    if peak_reserved >= PEAK_RESERVED_LIMIT_BYTES:
        raise PeakMemoryGateError(
            f"synthetic {arm_name} peak reserved {peak_reserved / GIB:.3f} GiB "
            f"is not below 32 GiB"
        )
    result = {
        "shape": [SYNTHETIC_BATCH_SIZE, SYNTHETIC_SEQUENCE_LENGTH],
        "step": record,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": peak_reserved / GIB,
        "peak_reserved_limit_bytes": PEAK_RESERVED_LIMIT_BYTES,
        "peak_reserved_strictly_below_limit": True,
        "optimizer_state_finite": True,
    }
    del module, optimizer, scheduler, scaler, batch, collector
    _release_cuda_objects()
    return result


def _run_real_gate(
    *,
    arm_name: str,
    data_binding: Mapping[str, Any],
    device: torch.device,
    epoch_dataset: Any,
    expected_row_hashes: Sequence[str],
    estimated_total_steps: int,
) -> dict[str, Any]:
    module, collector = _new_smoke_module(
        arm_name=arm_name,
        data_binding=data_binding,
        device=device,
    )
    initialization = {
        "shared_initialization_sha256": module.shared_initialization_sha256,
        "full_initialization_sha256": module.full_initialization_sha256,
    }
    optimizer, scheduler = _configure_smoke_optimizer(
        module,
        estimated_total_steps=estimated_total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    optimizer_steps = 0
    scheduler_steps = 0
    records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    for scheduled_index in range(MATCHED_REAL_STEPS):
        batch = epoch_dataset[scheduled_index]
        actual_hash = sha256_array(np.asarray(batch["row_ids"], dtype=np.int64))
        if actual_hash != expected_row_hashes[scheduled_index]:
            raise MatchedTrajectoryError(
                f"{arm_name} batch {scheduled_index} differs from the frozen matched sequence"
            )
        record, optimizer_steps, scheduler_steps = _run_amp_step(
            module=module,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            batch=batch,
            collector=collector,
            scheduled_index=scheduled_index,
            optimizer_step_count=optimizer_steps,
            scheduler_step_count=scheduler_steps,
        )
        records.append(record)
    torch.cuda.synchronize(device)
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if peak_reserved >= PEAK_RESERVED_LIMIT_BYTES:
        raise PeakMemoryGateError(
            f"real {arm_name} peak reserved {peak_reserved / GIB:.3f} GiB "
            f"is not below 32 GiB"
        )
    if not _all_finite(_optimizer_state_tensors(optimizer)):
        raise RelationSmokeError(f"real {arm_name} optimizer state is nonfinite")
    result = {
        "initialization": initialization,
        "scheduled_steps": MATCHED_REAL_STEPS,
        "optimizer_steps": int(optimizer_steps),
        "scheduler_steps": int(scheduler_steps),
        "skipped_steps": int(MATCHED_REAL_STEPS - optimizer_steps),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_reserved_gib": peak_reserved / GIB,
        "peak_reserved_limit_bytes": PEAK_RESERVED_LIMIT_BYTES,
        "peak_reserved_strictly_below_limit": True,
        "optimizer_state_finite": True,
        "steps": records,
    }
    del module, optimizer, scheduler, scaler, collector
    _release_cuda_objects()
    return result


def validate_matched_trajectories(
    arm_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Fail if arm treatment changes schedule/update-count/LR bookkeeping."""

    if set(arm_results) != set(ARM_ORDER):
        raise MatchedTrajectoryError(
            f"matched smoke requires exactly {list(ARM_ORDER)}, got {sorted(arm_results)}"
        )
    reference = arm_results[ARM_ORDER[0]]
    reference_steps = reference.get("steps")
    if not isinstance(reference_steps, list) or len(reference_steps) != MATCHED_REAL_STEPS:
        raise MatchedTrajectoryError("reference arm does not contain exactly 50 real steps")
    mismatches: list[str] = []
    for arm_name in ARM_ORDER[1:]:
        candidate_steps = arm_results[arm_name].get("steps")
        if not isinstance(candidate_steps, list) or len(candidate_steps) != len(reference_steps):
            mismatches.append(f"{arm_name}: scheduled-step count differs")
            continue
        for index, (left, right) in enumerate(zip(reference_steps, candidate_steps)):
            for key in (
                "scheduled_index",
                "schedule_batch_index",
                "row_ids_sha256",
                "batch_size",
                "dynamic_sequence_length",
                "has_relation_update",
                "optimizer_step_count",
                "scheduler_step_count",
            ):
                if left.get(key) != right.get(key):
                    mismatches.append(
                        f"{arm_name}: step {index} field {key} differs "
                        f"({left.get(key)!r} != {right.get(key)!r})"
                    )
            for key in ("lr_before", "lr_after"):
                left_lr = left.get(key)
                right_lr = right.get(key)
                if not isinstance(left_lr, list) or not isinstance(right_lr, list):
                    mismatches.append(f"{arm_name}: step {index} missing {key}")
                    continue
                if len(left_lr) != len(right_lr) or any(
                    not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-15)
                    for a, b in zip(left_lr, right_lr)
                ):
                    mismatches.append(f"{arm_name}: step {index} {key} differs")
    if mismatches:
        preview = "; ".join(mismatches[:12])
        raise MatchedTrajectoryError(
            "three-arm optimizer-step/LR trajectory is not matched; "
            "the smoke gate will not change the scientific contract: " + preview
        )
    return {
        "passed": True,
        "reference_arm": ARM_ORDER[0],
        "scheduled_steps": MATCHED_REAL_STEPS,
        "optimizer_steps": int(reference["optimizer_steps"]),
        "skipped_steps": int(reference["skipped_steps"]),
        "fields_checked": [
            "schedule membership/order",
            "physical/dynamic shape",
            "has_relation_update",
            "optimizer-step trajectory",
            "scheduler-step trajectory",
            "LR before/after trajectory",
        ],
    }


def aggregate_host_rss_gate(arm_peak_rss_bytes: Mapping[str, int]) -> dict[str, Any]:
    if set(arm_peak_rss_bytes) != set(ARM_ORDER):
        raise RelationSmokeError("host RSS gate requires one isolated peak for every arm")
    normalized = {name: int(arm_peak_rss_bytes[name]) for name in ARM_ORDER}
    if any(value <= 0 for value in normalized.values()):
        raise RelationSmokeError("host RSS measurements must be positive")
    aggregate = int(sum(normalized.values()))
    if aggregate >= AGGREGATE_HOST_RSS_LIMIT_BYTES:
        raise RelationSmokeError(
            f"conservative aggregate host RSS {aggregate / GIB:.3f} GiB "
            f"is not below 45 GiB"
        )
    return {
        "per_single_arm_peak_bytes": normalized,
        "conservative_aggregate_bytes": aggregate,
        "conservative_aggregate_gib": aggregate / GIB,
        "aggregate_limit_bytes": AGGREGATE_HOST_RSS_LIMIT_BYTES,
        "aggregate_strictly_below_limit": True,
        "method": (
            "three arms run sequentially on one CUDA device; each arm is sampled independently "
            "inside the same process and the three peak RSS values are summed conservatively"
        ),
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    repo = Path(__file__).resolve().parents[2]
    result: dict[str, dict[str, str]] = {}
    for relative in SMOKE_SOURCE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise RelationSmokeError(f"required smoke source is missing: {path}")
        result[relative] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _schedule_row_hashes(epoch_dataset: Any) -> list[str]:
    if len(epoch_dataset) < MATCHED_REAL_STEPS:
        raise RelationSmokeError(
            f"train epoch 0 has only {len(epoch_dataset)} batches; 50 are required"
        )
    hashes: list[str] = []
    for index in range(MATCHED_REAL_STEPS):
        start, stop = (
            int(value) for value in epoch_dataset.batch_offsets[index : index + 2]
        )
        rows = np.asarray(epoch_dataset.scheduled_rows[start:stop], dtype=np.int64)
        if rows.shape != (32,):
            raise RelationSmokeError(
                f"train epoch0 batch {index} is not physical B32: {rows.shape}"
            )
        hashes.append(sha256_array(rows))
    return hashes


def run_relation_cuda_smoke(config: RelationSmokeConfig) -> Path:
    """Run the independent B32 gate and publish only auditable smoke artifacts."""

    output_dir = _prepare_output_directory(config.output_dir)
    manifest_path = output_dir / "smoke_manifest.json"
    started_at = _utc_now()
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "gate": "relation_b32_real_cuda_launch_gate",
        "status": "running",
        "started_at": started_at,
        "contract": {
            "arms": list(ARM_ORDER),
            "synthetic_shape": [SYNTHETIC_BATCH_SIZE, SYNTHETIC_SEQUENCE_LENGTH],
            "real_schedule": "train_epoch_000 fixed first 50 batches",
            "physical_batch_size": 32,
            "peak_reserved_strict_limit_bytes": PEAK_RESERVED_LIMIT_BYTES,
            "aggregate_host_rss_strict_limit_bytes": AGGREGATE_HOST_RSS_LIMIT_BYTES,
            "cuda_required": True,
            "cpu_fallback_allowed": False,
            "b16_implemented_or_triggered": False,
            "production_checkpoint_or_embedding_write": False,
        },
        "inputs": {
            "h5ad_path": str(config.h5ad_path.resolve()),
            "schedule_root": str(config.schedule_root.resolve()),
            "cache_root": str(config.cache_root.resolve()),
            "output_dir": str(output_dir),
        },
    }
    try:
        source_hashes = _source_hashes()
        base["source_assets"] = source_hashes
        device, cuda = require_real_cuda(config.device_index)
        base["cuda"] = cuda
        base["determinism"] = enable_production_determinism()
        data_sampler = _HostRssSampler()
        with data_sampler:
            data = RelationDataModule(
                h5ad_path=config.h5ad_path,
                schedule_root=config.schedule_root,
                cache_root=config.cache_root,
                num_workers=0,
                pin_memory=False,
            )
            published_caches = data.store.validate_all_caches()
            data.setup("fit")
            epoch_dataset = data.store.epoch_dataset(
                "train", 0, data.datasets["train"]
            )
            expected_row_hashes = _schedule_row_hashes(epoch_dataset)
            resume_first_batch = epoch_dataset[0]
            resume_second_batch = epoch_dataset[1]
        estimated_total_steps = len(epoch_dataset) * 10
        base["data_binding"] = data.checkpoint_binding
        base["published_teacher_caches"] = published_caches
        base["data_preflight_peak_host_rss_bytes"] = int(data_sampler.peak_bytes)
        base["matched_schedule"] = {
            "schedule_kind": "train",
            "schedule_index": 0,
            "steps": MATCHED_REAL_STEPS,
            "batch_row_id_sha256": expected_row_hashes,
        }

        arms: dict[str, dict[str, Any]] = {}
        base["arms"] = arms
        arm_peak_rss: dict[str, int] = {}
        for arm_name in ARM_ORDER:
            with _HostRssSampler() as arm_sampler:
                synthetic = _run_synthetic_gate(
                    arm_name=arm_name,
                    data_binding=data.checkpoint_binding,
                    device=device,
                    estimated_total_steps=estimated_total_steps,
                )
                real = _run_real_gate(
                    arm_name=arm_name,
                    data_binding=data.checkpoint_binding,
                    device=device,
                    epoch_dataset=epoch_dataset,
                    expected_row_hashes=expected_row_hashes,
                    estimated_total_steps=estimated_total_steps,
                )
                resume = _run_resume_determinism_gate(
                    arm_name=arm_name,
                    data_binding=data.checkpoint_binding,
                    device=device,
                    first_batch=resume_first_batch,
                    second_batch=resume_second_batch,
                    estimated_total_steps=estimated_total_steps,
                    scratch_root=output_dir,
                )
            arm_peak_rss[arm_name] = int(arm_sampler.peak_bytes)
            arms[arm_name] = {
                "synthetic_worst_case": synthetic,
                "real_first_50": real,
                "interrupted_resume": resume,
                "single_process_peak_host_rss_bytes": int(arm_sampler.peak_bytes),
            }

        real_results = {name: arms[name]["real_first_50"] for name in ARM_ORDER}
        shared_hashes = {
            result["initialization"]["shared_initialization_sha256"]
            for result in real_results.values()
        }
        if len(shared_hashes) != 1:
            raise MatchedTrajectoryError(
                "three arms do not share identical initialized input/encoder components"
            )
        if (
            real_results["main_skip"]["initialization"]["full_initialization_sha256"]
            != real_results["main_radius"]["initialization"]["full_initialization_sha256"]
        ):
            raise MatchedTrajectoryError(
                "main_skip and main_radius full initialization hashes differ"
            )
        matched = validate_matched_trajectories(real_results)
        host_rss = aggregate_host_rss_gate(arm_peak_rss)

        completed_at = _utc_now()
        manifest = {
            **base,
            "status": "passed",
            "completed_at": completed_at,
            "matched_trajectory_gate": matched,
            "host_rss_gate": host_rss,
            "result": {
                "b32_launch_gate_passed": True,
                "real_50_step_gate_passed": True,
                "three_arm_gate_passed": True,
                "matched_trajectory_gate_passed": True,
                "resume_determinism_gate_passed": all(
                    arms[name]["interrupted_resume"]["passed"] for name in ARM_ORDER
                ),
                "resume_exact_gate_passed": all(
                    arms[name]["interrupted_resume"]["comparison"]
                    == "bitwise/exact; no tolerance relaxation"
                    for name in ARM_ORDER
                ),
                "host_rss_gate_passed": True,
                "gpu_memory_gate_passed": all(
                    phase["peak_reserved_strictly_below_limit"]
                    for name in ARM_ORDER
                    for phase in (
                        arms[name]["synthetic_worst_case"],
                        arms[name]["real_first_50"],
                        arms[name]["interrupted_resume"],
                    )
                ),
                "production_training_authorized_by_this_gate": True,
                "b16_implemented_or_triggered": False,
            },
        }
        return publish_smoke_manifest(output_dir, manifest, passed=True)
    except Exception as exc:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        completed_at = _utc_now()
        is_capacity_failure = isinstance(
            exc,
            (PeakMemoryGateError, torch.cuda.OutOfMemoryError),
        )
        failure = {
            **base,
            "status": "failed",
            "completed_at": completed_at,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "result": {
                "b32_launch_gate_passed": False,
                "real_50_step_gate_passed": False,
                "three_arm_gate_passed": False,
                "matched_trajectory_gate_passed": False,
                "resume_determinism_gate_passed": False,
                "resume_exact_gate_passed": False,
                "host_rss_gate_passed": False,
                "gpu_memory_gate_passed": False,
                "production_training_authorized_by_this_gate": False,
                "matched_contract_was_not_modified": True,
                "b16_implemented_or_triggered": False,
                "capacity_failure": bool(is_capacity_failure),
                "next_action": (
                    "stop; create and validate a separate B16 schedule/cache/runtime before any B16 run"
                    if is_capacity_failure
                    else "stop and diagnose this contract or numerical failure; do not launch training"
                ),
                "b16_requires_separate_schedule_cache": True,
            },
        }
        publish_smoke_manifest(output_dir, failure, passed=False)
        raise RelationSmokeError(
            f"B32 CUDA smoke failed; see {manifest_path}: {exc}"
        ) from exc


__all__ = [
    "AGGREGATE_HOST_RSS_LIMIT_BYTES",
    "ARM_ORDER",
    "MATCHED_REAL_STEPS",
    "PEAK_RESERVED_LIMIT_BYTES",
    "MatchedTrajectoryError",
    "RelationSmokeConfig",
    "RelationSmokeError",
    "aggregate_host_rss_gate",
    "build_synthetic_b32_l512_batch",
    "publish_smoke_manifest",
    "require_real_cuda",
    "run_relation_cuda_smoke",
    "validate_matched_trajectories",
]
