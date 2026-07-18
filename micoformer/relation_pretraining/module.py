"""Lightning runtime for the frozen relation-only pilot.

Teacher geometry is selector-only state.  It is consumed as hash-bound
``float64`` batch data and is never registered as a model buffer or serialized
inside a checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .losses import RelationLossOutput, relation_triplet_loss
from .mining import MiningConfig, RelationMiningOutput, mine_relations
from .model import RelationModelConfig, RelationModelOutput, RelationOnlyModel


ARM_SPECS: dict[str, tuple[str, str]] = {
    "main_skip": ("main", "skip"),
    "main_radius": ("main", "closest_radius_inside"),
    "pma_skip": ("pma", "skip"),
}

RELATION_SOURCE_PATHS = (
    "micoformer/relation_pretraining/model.py",
    "micoformer/relation_pretraining/mining.py",
    "micoformer/relation_pretraining/losses.py",
    "micoformer/relation_pretraining/data.py",
    "micoformer/relation_pretraining/module.py",
    "micoformer/relation_pretraining/workflow.py",
    "micoformer/relation_pretraining/extract.py",
    "scripts/2.train_relation.py",
)

RESUME_DETERMINISM_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "checkpointed_rng": [
        "python_random",
        "numpy_legacy_global",
        "torch_cpu",
        "torch_cuda_all_visible_devices",
    ],
    "optimizer_scheduler": "Lightning checkpoint plus strict CPU next-step equivalence test",
    "dropout": "strict CPU next-step equivalence test with dropout enabled",
    "amp_scaler": {
        "owner": "Lightning 2.5 precision-plugin checkpoint state",
        "cpu_unit_coverage": False,
        "required_verification": "real CUDA interrupted-resume smoke before long training",
    },
    "data_order": "pre-materialized schedule/cache; no stochastic sample transform",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_relation_source_manifest() -> dict[str, Any]:
    """Hash every source that defines the frozen relation training semantics."""

    repository = Path(__file__).resolve().parents[2]
    files: dict[str, str] = {}
    for relative in RELATION_SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise RuntimeError(f"required relation source is missing: {path}")
        files[relative] = _file_sha256(path)
    body: dict[str, Any] = {
        "schema_version": 1,
        "repository_root": str(repository),
        "hash_contract": "sha256 exact file bytes; canonical manifest JSON excludes manifest_sha256",
        "files": files,
    }
    body["manifest_sha256"] = _canonical_json_sha256(body)
    return body


def validate_relation_source_manifest(manifest: Any) -> dict[str, Any]:
    """Require a checkpoint/run source manifest to equal the live source tree."""

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("relation source manifest is missing or malformed")
    recorded_hash = manifest.get("manifest_sha256")
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if recorded_hash != _canonical_json_sha256(without_hash):
        raise RuntimeError("relation source manifest self-hash is invalid")
    current = build_relation_source_manifest()
    if manifest != current:
        recorded_files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        drifted = sorted(
            path
            for path in set(recorded_files) | set(current["files"])
            if recorded_files.get(path) != current["files"].get(path)
        )
        raise RuntimeError(
            "relation source manifest differs from live code; "
            f"drifted_sources={drifted or ['manifest metadata']}"
        )
    return current


def capture_relation_rng_state() -> dict[str, Any]:
    """Capture every global RNG stream used by the single-process training path."""

    cuda_available = bool(torch.cuda.is_available())
    cuda_states = (
        [state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()]
        if cuda_available
        else []
    )
    state = {
        "schema_version": 1,
        "python_random": random.getstate(),
        "numpy_legacy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda": {
            "captured": cuda_available,
            "visible_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "states": cuda_states,
        },
    }
    return _validate_relation_rng_state(state)


def _validate_relation_rng_state(state: Any) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != {
        "schema_version",
        "python_random",
        "numpy_legacy_global",
        "torch_cpu",
        "torch_cuda",
    }:
        raise RuntimeError("relation checkpoint RNG state is missing or malformed")
    if state.get("schema_version") != 1:
        raise RuntimeError("unsupported relation checkpoint RNG schema")
    python_state = state["python_random"]
    numpy_state = state["numpy_legacy_global"]
    torch_cpu = state["torch_cpu"]
    cuda = state["torch_cuda"]
    if not isinstance(python_state, tuple):
        raise RuntimeError("checkpoint Python RNG state is malformed")
    if (
        not isinstance(numpy_state, tuple)
        or len(numpy_state) != 5
        or not isinstance(numpy_state[1], np.ndarray)
    ):
        raise RuntimeError("checkpoint NumPy RNG state is malformed")
    if not isinstance(torch_cpu, Tensor) or torch_cpu.dtype != torch.uint8 or torch_cpu.ndim != 1:
        raise RuntimeError("checkpoint torch CPU RNG state is malformed")
    if not isinstance(cuda, dict) or set(cuda) != {
        "captured",
        "visible_device_count",
        "states",
    }:
        raise RuntimeError("checkpoint CUDA RNG state is malformed")
    captured = cuda["captured"]
    count = cuda["visible_device_count"]
    states = cuda["states"]
    if not isinstance(captured, bool) or type(count) is not int or count < 0:
        raise RuntimeError("checkpoint CUDA RNG topology is malformed")
    if not isinstance(states, list) or len(states) != (count if captured else 0):
        raise RuntimeError("checkpoint CUDA RNG state count is malformed")
    if captured != (count > 0):
        raise RuntimeError("checkpoint CUDA RNG availability/count is inconsistent")
    for item in states:
        if not isinstance(item, Tensor) or item.dtype != torch.uint8 or item.ndim != 1:
            raise RuntimeError("checkpoint CUDA RNG tensor is malformed")
    return state


def restore_relation_rng_state(state: Any) -> None:
    """Restore all RNG streams, failing if the CUDA topology changed."""

    state = _validate_relation_rng_state(state)
    cuda = state["torch_cuda"]
    current_cuda_available = bool(torch.cuda.is_available())
    if current_cuda_available != cuda["captured"]:
        raise RuntimeError(
            "CUDA availability changed across resume: "
            f"checkpoint={cuda['captured']} current={current_cuda_available}"
        )
    if current_cuda_available:
        current_count = int(torch.cuda.device_count())
        if current_count != cuda["visible_device_count"]:
            raise RuntimeError(
                "CUDA visible-device count changed across resume: "
                f"checkpoint={cuda['visible_device_count']} current={current_count}"
            )
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_legacy_global"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if cuda["captured"]:
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda["states"]])


@dataclass(frozen=True)
class RelationOptimizationConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.02
    margin: float = 0.10
    max_epochs: int = 10
    physical_batch_size: int = 32
    accumulate_grad_batches: int = 1

    def __post_init__(self) -> None:
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        if self.max_epochs != 10:
            raise ValueError("the frozen v1 pilot requires exactly 10 maximum epochs")
        if self.physical_batch_size != 32:
            raise ValueError("the published v1 schedule/cache contract requires physical B=32")
        if self.accumulate_grad_batches != 1:
            raise ValueError("gradient accumulation may not enlarge the relation candidate pool")


def _state_dict_sha256(
    state_dict: Mapping[str, Tensor],
    *,
    exclude_prefixes: tuple[str, ...] = (),
) -> str:
    """Hash state by canonical key, dtype, shape and exact tensor bytes."""

    digest = hashlib.sha256()
    included = 0
    for name in sorted(state_dict):
        if name.startswith(exclude_prefixes):
            continue
        value = state_dict[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(torch.tensor(value.shape, dtype=torch.int64).numpy().tobytes())
        digest.update(value.numpy().tobytes(order="C"))
        included += 1
    if included == 0:
        raise RuntimeError("cannot hash an empty state_dict selection")
    return digest.hexdigest()


def shared_initialization_sha256(model: RelationOnlyModel) -> str:
    """Hash only components shared by the main and PMA decoder arms."""

    return _state_dict_sha256(model.state_dict(), exclude_prefixes=("decoder.",))


def full_initialization_sha256(model: RelationOnlyModel) -> str:
    return _state_dict_sha256(model.state_dict())


def assert_final_z(z: Tensor) -> None:
    """Fail if mining could receive anything but finite normalized final ``z``."""

    if z.ndim != 2 or z.dtype != torch.float32:
        raise RuntimeError("RelationModelOutput.z must be float32 [B, D] before mining")
    if not torch.isfinite(z).all():
        raise RuntimeError("RelationModelOutput.z contains NaN or infinity")
    norms = z.norm(dim=-1)
    if not torch.allclose(
        norms,
        torch.ones_like(norms),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("RelationModelOutput.z rows are not L2-normalized")


@dataclass
class _RelationStep:
    loss: Tensor
    has_relation_update: bool


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TypeError("relation contract values must be finite and JSON serializable") from exc


class RelationPretrainingModule(L.LightningModule):
    """One relation-only student arm with strict checkpoint provenance."""

    def __init__(
        self,
        *,
        arm_name: str,
        model_config: RelationModelConfig | Mapping[str, Any] | None = None,
        data_binding: Mapping[str, Any],
        mining_config: MiningConfig | Mapping[str, Any] | None = None,
        optimization_config: RelationOptimizationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if arm_name not in ARM_SPECS:
            raise ValueError(f"arm_name must be one of {sorted(ARM_SPECS)}, got {arm_name!r}")
        decoder_kind, no_next_mode = ARM_SPECS[arm_name]

        if model_config is None:
            model_config = RelationModelConfig(decoder_kind=decoder_kind)
        elif isinstance(model_config, Mapping):
            model_config = RelationModelConfig(**dict(model_config))
        if model_config.decoder_kind != decoder_kind:
            raise ValueError(
                f"arm {arm_name!r} requires decoder_kind={decoder_kind!r}, "
                f"got {model_config.decoder_kind!r}"
            )

        if mining_config is None:
            mining_config = MiningConfig(no_next_mode=no_next_mode)
        elif isinstance(mining_config, Mapping):
            mining_config = MiningConfig(**dict(mining_config))
        if mining_config.no_next_mode != no_next_mode:
            raise ValueError(
                f"arm {arm_name!r} requires no_next_mode={no_next_mode!r}, "
                f"got {mining_config.no_next_mode!r}"
            )

        if optimization_config is None:
            optimization_config = RelationOptimizationConfig()
        elif isinstance(optimization_config, Mapping):
            optimization_config = RelationOptimizationConfig(**dict(optimization_config))

        self.arm_name = arm_name
        self.model_config = model_config
        self.mining_config = mining_config
        self.optimization_config = optimization_config
        self.data_binding = _json_copy(data_binding)
        self.model = RelationOnlyModel(model_config)
        self.automatic_optimization = False
        self.register_buffer(
            "_relation_optimizer_step_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_relation_scheduled_batch_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_relation_skipped_batch_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_relation_scheduler_step_count",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )
        self.shared_initialization_sha256 = shared_initialization_sha256(self.model)
        self.full_initialization_sha256 = full_initialization_sha256(self.model)
        self._consumed_teacher_caches: set[tuple[str, str, str]] = set()
        self._pending_resume_rng_state: Optional[dict[str, Any]] = None
        self._restored_runtime_counters: Optional[dict[str, Any]] = None
        source_manifest = build_relation_source_manifest()

        self.relation_contract: dict[str, Any] = {
            "schema_version": 1,
            "arm_name": self.arm_name,
            "model_config": asdict(self.model_config),
            "mining_config": asdict(self.mining_config),
            "optimization_config": asdict(self.optimization_config),
            "data_binding": copy.deepcopy(self.data_binding),
            "shared_initialization_sha256": self.shared_initialization_sha256,
            "full_initialization_sha256": self.full_initialization_sha256,
            "source_manifest": source_manifest,
            "resume_determinism": copy.deepcopy(RESUME_DETERMINISM_CONTRACT),
        }
        self.save_hyperparameters(copy.deepcopy(self.relation_contract))

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> RelationModelOutput:
        return self.model(genus_ids, rclr, key_padding_mask)

    @property
    def consumed_teacher_caches(self) -> list[dict[str, str]]:
        return [
            {
                "cache_manifest_sha256": manifest,
                "cache_sha256": cache,
                "schedule_file_sha256": schedule,
            }
            for manifest, cache, schedule in sorted(self._consumed_teacher_caches)
        ]

    @property
    def relation_optimizer_step_count(self) -> int:
        return int(self._relation_optimizer_step_count.item())

    @property
    def relation_runtime_counts(self) -> dict[str, int]:
        return {
            "scheduled_batches": int(self._relation_scheduled_batch_count.item()),
            "optimizer_steps": int(self._relation_optimizer_step_count.item()),
            "skipped_batches": int(self._relation_skipped_batch_count.item()),
            "scheduler_steps": int(self._relation_scheduler_step_count.item()),
        }

    def _assert_runtime_counter_invariants(self) -> None:
        counts = self.relation_runtime_counts
        if counts["scheduled_batches"] != (
            counts["optimizer_steps"] + counts["skipped_batches"]
        ):
            raise RuntimeError(
                "relation runtime counters violate scheduled=optimizer+skipped: "
                f"{counts}"
            )
        if counts["scheduler_steps"] != counts["optimizer_steps"]:
            raise RuntimeError(
                "relation runtime counters violate scheduler=optimizer: "
                f"{counts}"
            )

    @staticmethod
    def _checkpoint_current_lr(checkpoint: Mapping[str, Any]) -> Optional[float]:
        optimizer_states = checkpoint.get("optimizer_states")
        if not isinstance(optimizer_states, list) or len(optimizer_states) != 1:
            return None
        groups = optimizer_states[0].get("param_groups")
        if not isinstance(groups, list) or not groups:
            return None
        values = [float(group["lr"]) for group in groups if isinstance(group, dict) and "lr" in group]
        if len(values) != len(groups) or any(not math.isfinite(value) for value in values):
            raise RuntimeError("checkpoint optimizer groups contain an invalid current LR")
        if any(value != values[0] for value in values[1:]):
            raise RuntimeError("checkpoint optimizer groups disagree on current LR")
        return values[0]

    def _record_cache_provenance(self, batch: Mapping[str, Any]) -> None:
        values: list[str] = []
        for name in (
            "cache_manifest_sha256",
            "cache_sha256",
            "schedule_file_sha256",
        ):
            value = batch.get(name)
            if not isinstance(value, str) or len(value) != 64:
                raise RuntimeError(f"batch {name} must be one exact SHA256 string")
            try:
                int(value, 16)
            except ValueError as exc:
                raise RuntimeError(f"batch {name} is not hexadecimal") from exc
            values.append(value.lower())
        self._consumed_teacher_caches.add(tuple(values))

    def _prepare_teacher_inputs(
        self, batch: Mapping[str, Any], device: torch.device
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        raw_distances = batch.get("teacher_distances")
        raw_validity = batch.get("teacher_validity")
        expected = {"protein", "unifrac"}
        if not isinstance(raw_distances, Mapping) or set(raw_distances) != expected:
            raise RuntimeError("batch must contain exactly the two frozen teacher distances")
        if not isinstance(raw_validity, Mapping) or set(raw_validity) != expected:
            raise RuntimeError("batch must contain exactly the two frozen teacher validity masks")
        distances: dict[str, Tensor] = {}
        validity: dict[str, Tensor] = {}
        for name in sorted(expected):
            distance = torch.as_tensor(raw_distances[name])
            if distance.dtype != torch.float64:
                raise TypeError(
                    f"teacher {name!r} must remain float64 from cache through mining, "
                    f"got {distance.dtype}"
                )
            mask = torch.as_tensor(raw_validity[name])
            if mask.dtype != torch.bool:
                raise TypeError(f"teacher {name!r} validity must use bool")
            distances[name] = distance.to(device=device, dtype=torch.float64, non_blocking=True)
            validity[name] = mask.to(device=device, dtype=torch.bool, non_blocking=True)
        return distances, validity

    def _log_scalar(
        self,
        stage: str,
        name: str,
        value: Tensor | float | int,
        *,
        batch_size: int,
        reduction: str = "mean",
    ) -> None:
        if not isinstance(value, Tensor):
            value = torch.tensor(float(value), device=self.device, dtype=torch.float32)
        self.log(
            f"{stage}/{name}",
            value,
            on_step=stage == "train",
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size,
            reduce_fx=reduction,
            sync_dist=False,
        )

    @staticmethod
    def _embedding_health(z: Tensor) -> dict[str, Tensor]:
        detached = z.detach().float()
        batch_size = int(detached.shape[0])
        norms = detached.norm(dim=-1)
        centered = detached - detached.mean(dim=0, keepdim=True)
        singular = torch.linalg.svdvals(centered)
        spectrum = singular.square()
        spectrum_sum = spectrum.sum()
        if bool(spectrum_sum > 0):
            probability = spectrum / spectrum_sum
            positive = probability > 0
            effective_rank = torch.exp(
                -(probability[positive] * probability[positive].log()).sum()
            )
        else:
            effective_rank = spectrum_sum
        if batch_size > 1:
            squared = (detached[:, None, :] - detached[None, :, :]).square().sum(dim=-1)
            off_diagonal = ~torch.eye(batch_size, dtype=torch.bool, device=detached.device)
            pair_mean = squared[off_diagonal].mean()
        else:
            pair_mean = detached.sum() * 0.0
        dimension_std = detached.std(dim=0, unbiased=False)
        return {
            "z_norm_mean": norms.mean(),
            "z_norm_min": norms.min(),
            "z_norm_max": norms.max(),
            "dimension_std_mean": dimension_std.mean(),
            "dimension_std_max": dimension_std.max(),
            "effective_rank": effective_rank,
            "pair_squared_l2_mean": pair_mean,
        }

    def _log_relation_diagnostics(
        self,
        *,
        stage: str,
        z: Tensor,
        z_raw: Tensor,
        batch: Mapping[str, Any],
        mining: RelationMiningOutput,
        loss_output: RelationLossOutput,
    ) -> None:
        batch_size = int(z.shape[0])
        self._log_scalar(stage, "loss", loss_output.loss.detach(), batch_size=batch_size)
        for name, value in self._embedding_health(z).items():
            self._log_scalar(stage, f"health/{name}", value, batch_size=batch_size)
        self._log_scalar(
            stage,
            "health/z_raw_norm_mean",
            z_raw.detach().float().norm(dim=-1).mean(),
            batch_size=batch_size,
        )
        self._log_scalar(
            stage,
            "teacher/protein_valid_mass_mean",
            torch.as_tensor(batch["protein_valid_mass"], device=z.device).double().mean(),
            batch_size=batch_size,
        )
        self._log_scalar(
            stage,
            "teacher/protein_borrowed_mass_mean",
            torch.as_tensor(batch["protein_borrowed_mass"], device=z.device).double().mean(),
            batch_size=batch_size,
        )

        for counter_name, count in mining.counters().items():
            self._log_scalar(
                stage,
                f"mining/{counter_name}",
                count,
                batch_size=1,
                reduction="sum",
            )
        for counter_name, count in loss_output.counters.items():
            self._log_scalar(
                stage,
                f"objective/{counter_name}",
                count,
                batch_size=1,
                reduction="sum",
            )

        sites = torch.as_tensor(batch["site_ids"], device=z.device)
        for teacher_name, stats in loss_output.teacher_stats.items():
            self._log_scalar(
                stage, f"teacher/{teacher_name}/loss", stats.loss.detach(), batch_size=batch_size
            )
            if stats.valid_count:
                denominator = float(stats.valid_count)
                self._log_scalar(
                    stage,
                    f"teacher/{teacher_name}/positive_squared_l2",
                    stats.positive_squared_distance_sum.detach() / denominator,
                    batch_size=batch_size,
                )
                self._log_scalar(
                    stage,
                    f"teacher/{teacher_name}/negative_squared_l2",
                    stats.negative_squared_distance_sum.detach() / denominator,
                    batch_size=batch_size,
                )
                self._log_scalar(
                    stage,
                    f"teacher/{teacher_name}/active_hinge_fraction",
                    stats.active_count / denominator,
                    batch_size=batch_size,
                )
                self._log_scalar(
                    stage,
                    f"teacher/{teacher_name}/teacher_order_recovered_fraction",
                    stats.teacher_order_recovered_count / denominator,
                    batch_size=batch_size,
                )
                result = mining.teachers[teacher_name]
                valid = result.valid_mask
                anchors = torch.nonzero(valid, as_tuple=False).flatten()
                positives = result.positive_index[anchors]
                negatives = result.negative_index[anchors]
                self._log_scalar(
                    stage,
                    f"teacher/{teacher_name}/positive_same_site_fraction",
                    (sites[anchors] == sites[positives]).float().mean(),
                    batch_size=batch_size,
                )
                self._log_scalar(
                    stage,
                    f"teacher/{teacher_name}/negative_same_site_fraction",
                    (sites[anchors] == sites[negatives]).float().mean(),
                    batch_size=batch_size,
                )

        teacher_names = sorted(mining.teachers)
        if len(teacher_names) == 2:
            left = mining.teachers[teacher_names[0]].positive_index
            right = mining.teachers[teacher_names[1]].positive_index
            both = (left >= 0) & (right >= 0)
            if bool(both.any()):
                conflict = (left[both] != right[both]).float().mean()
                self._log_scalar(
                    stage,
                    "teacher/cross_teacher_positive_disagreement_fraction",
                    conflict,
                    batch_size=batch_size,
                )

    def _log_teacher_z_gradients(
        self,
        *,
        stage: str,
        z: Tensor,
        loss_output: RelationLossOutput,
    ) -> None:
        if stage != "train" or not torch.is_grad_enabled():
            return
        gradients: dict[str, Tensor] = {}
        for name, stats in loss_output.teacher_stats.items():
            gradient = torch.autograd.grad(
                stats.loss,
                z,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if gradient is None:
                gradient = torch.zeros_like(z)
            gradients[name] = gradient.detach().float()
            self._log_scalar(
                stage,
                f"teacher/{name}/z_gradient_norm",
                gradients[name].norm(),
                batch_size=int(z.shape[0]),
            )
        names = sorted(gradients)
        if len(names) == 2:
            left = gradients[names[0]].flatten()
            right = gradients[names[1]].flatten()
            denominator = left.norm() * right.norm()
            cosine = (
                torch.dot(left, right) / denominator
                if bool(denominator > 0)
                else torch.zeros((), device=z.device)
            )
            self._log_scalar(
                stage,
                "teacher/cross_teacher_z_gradient_cosine",
                cosine,
                batch_size=int(z.shape[0]),
            )

    def _shared_step(self, batch: Mapping[str, Any], stage: str) -> _RelationStep:
        self._record_cache_provenance(batch)
        output = self.model(
            torch.as_tensor(batch["genus_ids"], device=self.device, dtype=torch.long),
            torch.as_tensor(batch["rclr"], device=self.device, dtype=torch.float32),
            torch.as_tensor(batch["padding_mask"], device=self.device, dtype=torch.bool),
        )
        z = output.z.float()
        assert_final_z(z)
        batch_size = int(z.shape[0])
        if batch_size < 2:
            loss = z.sum() * 0.0
            self._log_scalar(stage, "loss", loss.detach(), batch_size=batch_size)
            self._log_scalar(
                stage, "health/singleton_batch", 1, batch_size=1, reduction="sum"
            )
            return _RelationStep(loss=loss, has_relation_update=False)

        teacher_distances, teacher_validity = self._prepare_teacher_inputs(batch, z.device)
        # The encoder/decoder may be under mixed precision, but student geometry,
        # mining and loss reduction are explicitly outside autocast.
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = z.float()
            mining = mine_relations(
                z,
                teacher_distances,
                torch.as_tensor(batch["row_ids"], device=z.device, dtype=torch.long),
                torch.as_tensor(batch["project_ids"], device=z.device, dtype=torch.long),
                teacher_validity=teacher_validity,
                config=self.mining_config,
            )
            loss_output = relation_triplet_loss(
                z,
                mining,
                margin=self.optimization_config.margin,
            )
        self._log_relation_diagnostics(
            stage=stage,
            z=z,
            z_raw=output.z_raw,
            batch=batch,
            mining=mining,
            loss_output=loss_output,
        )
        self._log_teacher_z_gradients(
            stage=stage,
            z=z,
            loss_output=loss_output,
        )
        return _RelationStep(
            loss=loss_output.loss,
            has_relation_update=loss_output.has_relation_update,
        )

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        self._relation_scheduled_batch_count.add_(1)
        result = self._shared_step(batch, "train")
        if not result.has_relation_update:
            # AdamW applies decoupled weight decay even to a graph-connected
            # zero.  Therefore none-present batches skip backward, optimizer
            # and scheduler as one indivisible operation.
            self._log_scalar(
                "train", "optimizer/skipped_step", 1, batch_size=1, reduction="sum"
            )
            self._log_scalar(
                "train",
                "optimizer/completed_step_count",
                self.relation_optimizer_step_count,
                batch_size=1,
            )
            self._relation_skipped_batch_count.add_(1)
            self._assert_runtime_counter_invariants()
            return result.loss.detach()

        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        self.manual_backward(result.loss)
        optimizer.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
            self._relation_scheduler_step_count.add_(1)
        self._relation_optimizer_step_count.add_(1)
        self._assert_runtime_counter_invariants()
        self._log_scalar(
            "train", "optimizer/completed_step", 1, batch_size=1, reduction="sum"
        )
        self._log_scalar(
            "train",
            "optimizer/completed_step_count",
            self.relation_optimizer_step_count,
            batch_size=1,
        )
        return result.loss.detach()

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "val").loss

    def test_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        return self._shared_step(batch, "test").loss

    def configure_optimizers(self) -> dict[str, Any]:
        normalization_parameter_ids: set[int] = set()
        for submodule in self.model.modules():
            if isinstance(submodule, nn.LayerNorm):
                normalization_parameter_ids.update(
                    id(parameter) for parameter in submodule.parameters(recurse=False)
                )
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("bias") or id(parameter) in normalization_parameter_ids:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        if not decay or not no_decay:
            raise RuntimeError("optimizer parameter grouping unexpectedly produced an empty group")
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.optimization_config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.optimization_config.learning_rate,
        )
        trainer = getattr(self, "_trainer", None)
        total_steps = max(1, int(getattr(trainer, "estimated_stepping_batches", 1)))
        warmup_steps = max(1, int(math.ceil(total_steps * self.optimization_config.warmup_fraction)))

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = min(
                1.0,
                float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)),
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "warmup_cosine",
            },
        }

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self._assert_runtime_counter_invariants()
        checkpoint["relation_contract"] = copy.deepcopy(self.relation_contract)
        checkpoint["consumed_teacher_caches"] = self.consumed_teacher_caches
        checkpoint["relation_rng_state"] = capture_relation_rng_state()
        checkpoint["relation_runtime_counters"] = {
            **self.relation_runtime_counts,
            "current_lr": self._checkpoint_current_lr(checkpoint),
        }

    def _restore_checkpoint_metadata(
        self,
        checkpoint: dict[str, Any],
        *,
        restore_rng_state: bool,
    ) -> None:
        contract = checkpoint.get("relation_contract")
        if contract != self.relation_contract:
            raise RuntimeError("relation checkpoint contract does not match the current run contract")
        validate_relation_source_manifest(contract.get("source_manifest"))
        if contract.get("resume_determinism") != RESUME_DETERMINISM_CONTRACT:
            raise RuntimeError("relation checkpoint resume-determinism contract drifted")
        provenance = checkpoint.get("consumed_teacher_caches")
        if not isinstance(provenance, list):
            raise RuntimeError("relation checkpoint lacks consumed teacher-cache provenance")
        restored: set[tuple[str, str, str]] = set()
        for record in provenance:
            if not isinstance(record, dict) or set(record) != {
                "cache_manifest_sha256",
                "cache_sha256",
                "schedule_file_sha256",
            }:
                raise RuntimeError("relation checkpoint teacher-cache provenance is malformed")
            values = tuple(record[key] for key in (
                "cache_manifest_sha256",
                "cache_sha256",
                "schedule_file_sha256",
            ))
            if any(not isinstance(value, str) or len(value) != 64 for value in values):
                raise RuntimeError("relation checkpoint contains an invalid cache SHA256")
            restored.add(values)
        self._consumed_teacher_caches = restored
        rng_state = _validate_relation_rng_state(checkpoint.get("relation_rng_state"))
        counters = checkpoint.get("relation_runtime_counters")
        if not isinstance(counters, dict) or set(counters) != {
            "scheduled_batches",
            "optimizer_steps",
            "skipped_batches",
            "scheduler_steps",
            "current_lr",
        }:
            raise RuntimeError("relation checkpoint runtime counters are missing or malformed")
        for name in (
            "scheduled_batches",
            "optimizer_steps",
            "skipped_batches",
            "scheduler_steps",
        ):
            if type(counters[name]) is not int or counters[name] < 0:
                raise RuntimeError(f"relation checkpoint counter {name} is invalid")
        current_lr = counters["current_lr"]
        if current_lr is not None and (
            not isinstance(current_lr, (int, float))
            or isinstance(current_lr, bool)
            or not math.isfinite(float(current_lr))
        ):
            raise RuntimeError("relation checkpoint current LR is invalid")
        if counters["scheduled_batches"] != (
            counters["optimizer_steps"] + counters["skipped_batches"]
        ) or counters["scheduler_steps"] != counters["optimizer_steps"]:
            raise RuntimeError("relation checkpoint runtime counter invariants failed")
        self._restored_runtime_counters = copy.deepcopy(counters)
        if restore_rng_state:
            self._pending_resume_rng_state = copy.deepcopy(rng_state)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        # Lightning calls this on the true resume path.  Static checkpoint
        # inspection uses ``load_relation_checkpoint(..., restore_rng_state=False)``
        # so embedding export cannot perturb the caller's random streams.
        self._restore_checkpoint_metadata(checkpoint, restore_rng_state=True)

    def _validate_restored_runtime_state(self) -> None:
        if self._restored_runtime_counters is None:
            return
        observed = self.relation_runtime_counts
        expected = {
            key: int(self._restored_runtime_counters[key])
            for key in observed
        }
        if observed != expected:
            raise RuntimeError(
                "restored relation runtime buffers differ from checkpoint counters: "
                f"observed={observed} expected={expected}"
            )
        self._assert_runtime_counter_invariants()

    def on_train_start(self) -> None:
        self._validate_restored_runtime_state()
        if self._pending_resume_rng_state is not None:
            restore_relation_rng_state(self._pending_resume_rng_state)
            self._pending_resume_rng_state = None


def load_relation_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_data_binding: Optional[Mapping[str, Any]] = None,
    map_location: str | torch.device = "cpu",
    restore_rng_state: bool = False,
) -> RelationPretrainingModule:
    """Strictly reconstruct a relation module without deserializing teacher arrays."""

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("relation checkpoint root must be a mapping")
    contract = checkpoint.get("relation_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise RuntimeError("checkpoint is not a schema-v1 relation checkpoint")
    validate_relation_source_manifest(contract.get("source_manifest"))
    if contract.get("resume_determinism") != RESUME_DETERMINISM_CONTRACT:
        raise RuntimeError("relation checkpoint resume-determinism contract drifted")
    data_binding = contract.get("data_binding")
    if not isinstance(data_binding, dict):
        raise RuntimeError("relation checkpoint data binding is missing")
    if expected_data_binding is not None and data_binding != _json_copy(expected_data_binding):
        raise RuntimeError("relation checkpoint data binding does not match the expected data binding")

    caller_rng_state = capture_relation_rng_state() if not restore_rng_state else None
    try:
        module = RelationPretrainingModule(
            arm_name=contract["arm_name"],
            model_config=contract["model_config"],
            data_binding=data_binding,
            mining_config=contract["mining_config"],
            optimization_config=contract["optimization_config"],
        )
    finally:
        if caller_rng_state is not None:
            restore_relation_rng_state(caller_rng_state)
    # Initialization hashes describe the original step-0 stream, not the
    # throwaway constructor state used while restoring trained parameters.
    module.shared_initialization_sha256 = contract["shared_initialization_sha256"]
    module.full_initialization_sha256 = contract["full_initialization_sha256"]
    module.relation_contract = copy.deepcopy(contract)
    module._restore_checkpoint_metadata(
        checkpoint,
        restore_rng_state=restore_rng_state,
    )
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("relation checkpoint state_dict is missing")
    incompatible = module.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict relation checkpoint reload reported incompatible keys")
    module._validate_restored_runtime_state()
    if restore_rng_state:
        if module._pending_resume_rng_state is None:
            raise RuntimeError("relation checkpoint RNG state was not staged for restoration")
        restore_relation_rng_state(module._pending_resume_rng_state)
        module._pending_resume_rng_state = None
    return module
