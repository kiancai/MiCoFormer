"""Lightning runtime for the full-size F1/F2 architecture gate."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from micoformer.relation_pretraining.mining import MiningConfig
from micoformer.relation_pretraining.module import (
    RESUME_DETERMINISM_CONTRACT,
    _json_copy,
    _state_dict_sha256,
    _validate_relation_rng_state,
    capture_relation_rng_state,
    restore_relation_rng_state,
)

from .losses import MultiHeadRelationLossOutput, multi_head_relation_triplet_loss
from .mining import MultiHeadRelationMiningOutput, mine_relations_by_teacher
from .model import (
    FULLSCALE_RELATION_ARMS,
    FullscaleRelationArm,
    FullscaleRelationModel,
    FullscaleRelationModelConfig,
    FullscaleRelationModelOutput,
)


FULLSCALE_SOURCE_PATHS = (
    "micoformer/fullscale_relation_pretraining/__init__.py",
    "micoformer/fullscale_relation_pretraining/model.py",
    "micoformer/fullscale_relation_pretraining/mining.py",
    "micoformer/fullscale_relation_pretraining/losses.py",
    "micoformer/fullscale_relation_pretraining/data.py",
    "micoformer/fullscale_relation_pretraining/fast_data.py",
    "micoformer/fullscale_relation_pretraining/fast_teacher.py",
    "micoformer/fullscale_relation_pretraining/module.py",
)
COMMON_STATE_PREFIXES = (
    "input_stem.",
    "encoder.",
    "final_token_norm.",
    "mlm_head.",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def build_fullscale_source_manifest() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]
    files: dict[str, str] = {}
    for relative in FULLSCALE_SOURCE_PATHS:
        path = repository / relative
        if not path.is_file():
            raise RuntimeError(f"required fullscale relation source is missing: {path}")
        files[relative] = _file_sha256(path)
    # The new selector and objective intentionally reuse exact historical
    # helper functions; bind those dependencies by byte hash as well.
    dependencies = {
        relative: _file_sha256(repository / relative)
        for relative in (
            "micoformer/relation_pretraining/mining.py",
            "micoformer/relation_pretraining/losses.py",
            "micoformer/relation_pretraining/module.py",
            "micoformer/models/attn_bias.py",
            "micoformer/models/heads.py",
        )
    }
    body: dict[str, Any] = {
        "schema_version": 1,
        "repository_root": str(repository),
        "files": files,
        "dependency_files": dependencies,
    }
    body["manifest_sha256"] = _canonical_sha256(body)
    return body


def validate_fullscale_source_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("fullscale source manifest is missing or malformed")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != _canonical_sha256(body):
        raise RuntimeError("fullscale source manifest self-hash is invalid")
    current = build_fullscale_source_manifest()
    if manifest != current:
        raise RuntimeError("fullscale source manifest differs from the live source tree")
    return current


@dataclass(frozen=True)
class FullscaleRelationOptimizationConfig:
    total_optimizer_steps: int
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_fraction: float = 0.02
    margin: float = 0.10
    physical_batch_size: int = 32
    mlm_every_relation_steps: int = 4
    mlm_mask_probability: float = 0.15
    mlm_huber_beta: float = 1.0
    max_relation_epochs: int = 10
    gradient_cosine_interval: int = 1

    def __post_init__(self) -> None:
        if self.total_optimizer_steps <= 0:
            raise ValueError("total_optimizer_steps must be positive and explicitly frozen")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0,1)")
        if not math.isfinite(self.margin) or self.margin < 0:
            raise ValueError("margin must be finite and non-negative")
        if self.physical_batch_size not in {32, 64}:
            raise ValueError("physical_batch_size must remain inside the frozen B32/B64 audit")
        if self.mlm_every_relation_steps != 4:
            raise ValueError("the user-approved relation:MLM cadence is exactly 4:1")
        if self.mlm_mask_probability != 0.15:
            raise ValueError("the user-approved MLM mask probability is exactly 0.15")
        if not math.isfinite(self.mlm_huber_beta) or self.mlm_huber_beta <= 0:
            raise ValueError("mlm_huber_beta must be positive and finite")
        if self.max_relation_epochs not in {1, 2, 10}:
            raise ValueError("relation epochs must be one of the frozen fast/formal values {1,2,10}")
        if self.gradient_cosine_interval < 0:
            raise ValueError("gradient_cosine_interval must be non-negative")


@dataclass
class _RelationStep:
    loss: Tensor
    output: FullscaleRelationModelOutput
    mining: MultiHeadRelationMiningOutput | None
    loss_output: MultiHeadRelationLossOutput | None
    has_update: bool


def common_initialization_sha256(model: FullscaleRelationModel) -> str:
    selected = {
        name: tensor
        for name, tensor in model.state_dict().items()
        if name.startswith(COMMON_STATE_PREFIXES)
    }
    if not selected:
        raise RuntimeError("failed to isolate common fullscale model state")
    return _state_dict_sha256(selected)


def full_initialization_sha256(model: FullscaleRelationModel) -> str:
    return _state_dict_sha256(model.state_dict())


def assert_unit_embedding(z: Tensor, label: str) -> None:
    if z.ndim != 2 or z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
        raise RuntimeError(f"{label} must be finite float32 [B,D]")
    if not torch.allclose(z.norm(dim=-1), torch.ones(z.shape[0], device=z.device), atol=1e-6, rtol=1e-5):
        raise RuntimeError(f"{label} rows must be L2-normalized")


class FullscaleRelationPretrainingModule(L.LightningModule):
    """Manual two-step runtime: four relation steps then one independent MLM step."""

    def __init__(
        self,
        *,
        arm: FullscaleRelationArm,
        data_binding: Mapping[str, Any],
        optimization_config: FullscaleRelationOptimizationConfig | Mapping[str, Any],
        model_config: FullscaleRelationModelConfig | Mapping[str, Any] | None = None,
        mining_config: MiningConfig | Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if arm not in FULLSCALE_RELATION_ARMS:
            raise ValueError(f"arm must be one of {FULLSCALE_RELATION_ARMS}")
        if model_config is None:
            model_config = FullscaleRelationModelConfig(arm=arm)
        elif isinstance(model_config, Mapping):
            model_config = FullscaleRelationModelConfig(**dict(model_config))
        if model_config.arm != arm:
            raise ValueError("model_config.arm must match module arm")
        if isinstance(optimization_config, Mapping):
            optimization_config = FullscaleRelationOptimizationConfig(**dict(optimization_config))
        if mining_config is None:
            mining_config = MiningConfig(no_next_mode="skip")
        elif isinstance(mining_config, Mapping):
            mining_config = MiningConfig(**dict(mining_config))
        if mining_config.no_next_mode != "skip":
            raise ValueError("the frozen fullscale selector requires no_next_mode='skip'")

        self.arm = arm
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.mining_config = mining_config
        self.data_binding = _json_copy(data_binding)
        self.model = FullscaleRelationModel(model_config)
        self.automatic_optimization = False
        for name in (
            "relation_scheduled_batches",
            "relation_optimizer_steps",
            "relation_skipped_batches",
            "mlm_optimizer_steps",
            "scheduler_steps",
        ):
            self.register_buffer(f"_{name}", torch.zeros((), dtype=torch.long), persistent=True)

        self.shared_initialization_sha256 = common_initialization_sha256(self.model)
        self.full_initialization_sha256 = full_initialization_sha256(self.model)
        self.source_manifest = build_fullscale_source_manifest()
        self.fullscale_contract = {
            "schema_version": 1,
            "arm": arm,
            "model_config": asdict(model_config),
            "optimization_config": asdict(optimization_config),
            "mining_config": asdict(mining_config),
            "data_binding": copy.deepcopy(self.data_binding),
            "shared_initialization_sha256": self.shared_initialization_sha256,
            "full_initialization_sha256": self.full_initialization_sha256,
            "sample_interfaces": ["h_raw", "h_unit"],
            "teacher_heads": "shared" if arm == "f1_shared" else "protein_and_unifrac",
            "source_manifest": self.source_manifest,
            "resume_determinism": copy.deepcopy(RESUME_DETERMINISM_CONTRACT),
        }
        self.save_hyperparameters(copy.deepcopy(self.fullscale_contract))

    @property
    def runtime_counts(self) -> dict[str, int]:
        return {
            "relation_scheduled_batches": int(self._relation_scheduled_batches.item()),
            "relation_optimizer_steps": int(self._relation_optimizer_steps.item()),
            "relation_skipped_batches": int(self._relation_skipped_batches.item()),
            "mlm_optimizer_steps": int(self._mlm_optimizer_steps.item()),
            "scheduler_steps": int(self._scheduler_steps.item()),
        }

    def _assert_runtime_counts(self) -> None:
        counts = self.runtime_counts
        if counts["relation_scheduled_batches"] != (
            counts["relation_optimizer_steps"] + counts["relation_skipped_batches"]
        ):
            raise RuntimeError(f"relation runtime conservation failed: {counts}")
        total = counts["relation_optimizer_steps"] + counts["mlm_optimizer_steps"]
        if counts["scheduler_steps"] != total:
            raise RuntimeError(f"scheduler/runtime conservation failed: {counts}")
        if counts["mlm_optimizer_steps"] > counts["relation_optimizer_steps"] // 4:
            raise RuntimeError(f"MLM cadence conservation failed: {counts}")

    def forward(
        self,
        genus_ids: Tensor,
        rclr: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> FullscaleRelationModelOutput:
        return self.model.forward_relation(genus_ids, rclr, key_padding_mask)

    def _prepare_teacher_inputs(
        self, batch: Mapping[str, Any], device: torch.device
    ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        raw_distances = batch.get("teacher_distances")
        raw_validity = batch.get("teacher_validity")
        if not isinstance(raw_distances, Mapping) or set(raw_distances) != {"protein", "unifrac"}:
            raise RuntimeError("batch teacher_distances must contain exactly protein and unifrac")
        if not isinstance(raw_validity, Mapping) or set(raw_validity) != set(raw_distances):
            raise RuntimeError("batch teacher_validity must match teacher_distances")
        distances = {
            name: torch.as_tensor(value, device=device, dtype=torch.float64)
            for name, value in raw_distances.items()
        }
        validity = {
            name: torch.as_tensor(value, device=device, dtype=torch.bool)
            for name, value in raw_validity.items()
        }
        return distances, validity

    @staticmethod
    def _embedding_health(z: Tensor) -> dict[str, Tensor]:
        z = z.detach().float()
        centered = z - z.mean(dim=0, keepdim=True)
        std = centered.std(dim=0, unbiased=False)
        singular = torch.linalg.svdvals(centered) if z.shape[0] > 1 else torch.zeros(1, device=z.device)
        energy = singular.square()
        probability = energy / energy.sum().clamp_min(1e-12)
        effective_rank = torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum())
        return {
            "dimension_std_mean": std.mean(),
            "dimension_std_max": std.max(),
            "effective_rank": effective_rank,
        }

    def _relation_forward(self, batch: Mapping[str, Any], stage: str) -> _RelationStep:
        genus = torch.as_tensor(batch["genus_ids"], device=self.device, dtype=torch.long)
        rclr = torch.as_tensor(batch["rclr"], device=self.device, dtype=torch.float32)
        padding = torch.as_tensor(batch["padding_mask"], device=self.device, dtype=torch.bool)
        output = self.model.forward_relation(genus, rclr, padding)
        for name, z in output.teacher_z.items():
            assert_unit_embedding(z, f"teacher_z/{name}")
        if genus.shape[0] < 2:
            zero = output.h_raw.sum() * 0.0
            return _RelationStep(zero, output, None, None, False)
        distances, validity = self._prepare_teacher_inputs(batch, self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            teacher_z = {name: z.float() for name, z in output.teacher_z.items()}
            mining = mine_relations_by_teacher(
                teacher_z,
                distances,
                torch.as_tensor(batch["row_ids"], device=self.device, dtype=torch.long),
                torch.as_tensor(batch["project_ids"], device=self.device, dtype=torch.long),
                teacher_validity=validity,
                config=self.mining_config,
            )
            loss_output = multi_head_relation_triplet_loss(
                teacher_z,
                mining,
                margin=self.optimization_config.margin,
            )
        batch_size = int(genus.shape[0])
        self.log(f"{stage}/relation_loss", loss_output.loss.detach(), batch_size=batch_size)
        for name, value in self._embedding_health(output.h_unit).items():
            self.log(f"{stage}/h_unit/{name}", value, batch_size=batch_size)
        for name, stats in loss_output.teacher_stats.items():
            self.log(f"{stage}/teacher/{name}/loss", stats.loss.detach(), batch_size=batch_size)
            if stats.valid_count:
                self.log(
                    f"{stage}/teacher/{name}/active_hinge_fraction",
                    stats.active_count / stats.valid_count,
                    batch_size=batch_size,
                )
        gradient_interval = self.optimization_config.gradient_cosine_interval
        relation_index = self.runtime_counts["relation_scheduled_batches"]
        if (
            stage == "train"
            and torch.is_grad_enabled()
            and gradient_interval > 0
            and relation_index % gradient_interval == 0
        ):
            gradients: dict[str, Tensor] = {}
            for name, teacher_loss in loss_output.teacher_losses.items():
                gradient = torch.autograd.grad(
                    teacher_loss,
                    output.h_raw,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                gradients[name] = (
                    torch.zeros_like(output.h_raw) if gradient is None else gradient
                ).detach().float()
            if len(gradients) == 2:
                names = sorted(gradients)
                left, right = (gradients[name].flatten() for name in names)
                denominator = left.norm() * right.norm()
                cosine = torch.dot(left, right) / denominator if bool(denominator > 0) else left.sum() * 0.0
                self.log("train/teacher/backbone_gradient_cosine", cosine, batch_size=batch_size)
        return _RelationStep(
            loss_output.loss,
            output,
            mining,
            loss_output,
            loss_output.has_relation_update,
        )

    def _mlm_forward(self, batch: Mapping[str, Any], stage: str) -> Tensor:
        genus = torch.as_tensor(batch["genus_ids"], device=self.device, dtype=torch.long)
        rclr = torch.as_tensor(batch["rclr"], device=self.device, dtype=torch.float32)
        padding = torch.as_tensor(batch["padding_mask"], device=self.device, dtype=torch.bool)
        abundance_mask = torch.as_tensor(
            batch["abundance_mask"], device=self.device, dtype=torch.bool
        )
        output = self.model.forward_mlm(genus, rclr, abundance_mask, padding)
        if output.mlm_prediction is None:
            raise RuntimeError("MLM forward did not return predictions")
        loss = F.smooth_l1_loss(
            output.mlm_prediction[abundance_mask].float(),
            rclr[abundance_mask].float(),
            beta=self.optimization_config.mlm_huber_beta,
            reduction="mean",
        )
        mae = (output.mlm_prediction[abundance_mask].float() - rclr[abundance_mask]).abs().mean()
        self.log(f"{stage}/mlm_loss", loss.detach(), batch_size=int(genus.shape[0]))
        self.log(f"{stage}/mlm_mae", mae.detach(), batch_size=int(genus.shape[0]))
        return loss

    def _optimizer_scheduler_step(self, loss: Tensor) -> None:
        optimizer = self.optimizers(use_pl_optimizer=True)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()
        self._scheduler_steps.add_(1)

    def training_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        relation_batch = batch.get("relation", batch)
        if not isinstance(relation_batch, Mapping):
            raise RuntimeError("training relation batch must be a mapping")
        self._relation_scheduled_batches.add_(1)
        result = self._relation_forward(relation_batch, "train")
        if result.has_update:
            self._optimizer_scheduler_step(result.loss)
            self._relation_optimizer_steps.add_(1)
        else:
            self._relation_skipped_batches.add_(1)

        if (
            result.has_update
            and self.runtime_counts["relation_optimizer_steps"]
            % self.optimization_config.mlm_every_relation_steps
            == 0
        ):
            mlm_batch = batch.get("mlm")
            if not isinstance(mlm_batch, Mapping):
                raise RuntimeError("every fourth successful relation step requires an independent MLM batch")
            mlm_loss = self._mlm_forward(mlm_batch, "train")
            self._optimizer_scheduler_step(mlm_loss)
            self._mlm_optimizer_steps.add_(1)
        self._assert_runtime_counts()
        return result.loss.detach()

    def validation_step(self, batch: Mapping[str, Any], batch_idx: int) -> Tensor:
        del batch_idx
        relation_batch = batch.get("relation", batch)
        if not isinstance(relation_batch, Mapping):
            raise RuntimeError("validation relation batch must be a mapping")
        return self._relation_forward(relation_batch, "val").loss

    def configure_optimizers(self) -> dict[str, Any]:
        normalization_ids: set[int] = set()
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                normalization_ids.update(id(p) for p in module.parameters(recurse=False))
        decay: list[Tensor] = []
        no_decay: list[Tensor] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.endswith("bias") or id(parameter) in normalization_ids:
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        if not decay or not no_decay:
            raise RuntimeError("optimizer parameter grouping produced an empty group")
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.optimization_config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.optimization_config.learning_rate,
        )
        total_steps = self.optimization_config.total_optimizer_steps
        warmup_steps = max(1, math.ceil(total_steps * self.optimization_config.warmup_fraction))

        def lr_multiplier(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
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
        self._assert_runtime_counts()
        validate_fullscale_source_manifest(self.source_manifest)
        checkpoint["fullscale_relation_contract"] = copy.deepcopy(self.fullscale_contract)
        checkpoint["fullscale_relation_rng_state"] = capture_relation_rng_state()
        checkpoint["fullscale_relation_runtime_counts"] = self.runtime_counts

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        contract = checkpoint.get("fullscale_relation_contract")
        if not isinstance(contract, dict) or contract.get("schema_version") != 1:
            raise RuntimeError("fullscale relation checkpoint contract is missing")
        validate_fullscale_source_manifest(contract.get("source_manifest"))
        if contract != self.fullscale_contract:
            raise RuntimeError("fullscale relation checkpoint contract differs from module contract")
        rng_state = checkpoint.get("fullscale_relation_rng_state")
        _validate_relation_rng_state(rng_state)
        counts = checkpoint.get("fullscale_relation_runtime_counts")
        if not isinstance(counts, dict) or set(counts) != set(self.runtime_counts):
            raise RuntimeError("fullscale relation checkpoint runtime counts are missing")


def load_fullscale_relation_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_arm: FullscaleRelationArm | None = None,
    expected_data_binding: Mapping[str, Any] | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng_state: bool = False,
) -> FullscaleRelationPretrainingModule:
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("fullscale relation checkpoint root must be a mapping")
    contract = checkpoint.get("fullscale_relation_contract")
    if not isinstance(contract, dict) or contract.get("schema_version") != 1:
        raise RuntimeError("fullscale relation checkpoint contract is missing")
    validate_fullscale_source_manifest(contract.get("source_manifest"))
    arm = contract.get("arm")
    if arm not in FULLSCALE_RELATION_ARMS:
        raise RuntimeError("checkpoint contains an unknown fullscale relation arm")
    if expected_arm is not None and arm != expected_arm:
        raise RuntimeError(f"checkpoint arm mismatch: expected={expected_arm} observed={arm}")
    data_binding = contract.get("data_binding")
    if not isinstance(data_binding, dict):
        raise RuntimeError("checkpoint data binding is missing")
    if expected_data_binding is not None and data_binding != _json_copy(expected_data_binding):
        raise RuntimeError("checkpoint data binding mismatch")

    caller_rng = capture_relation_rng_state() if not restore_rng_state else None
    try:
        module = FullscaleRelationPretrainingModule(
            arm=arm,
            data_binding=data_binding,
            model_config=contract["model_config"],
            optimization_config=contract["optimization_config"],
            mining_config=contract["mining_config"],
        )
    finally:
        if caller_rng is not None:
            restore_relation_rng_state(caller_rng)
    module.shared_initialization_sha256 = contract["shared_initialization_sha256"]
    module.full_initialization_sha256 = contract["full_initialization_sha256"]
    module.source_manifest = copy.deepcopy(contract["source_manifest"])
    module.fullscale_contract = copy.deepcopy(contract)
    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("fullscale relation checkpoint state_dict is missing")
    incompatible = module.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("strict fullscale checkpoint load reported incompatible keys")
    recorded_counts = checkpoint.get("fullscale_relation_runtime_counts")
    if recorded_counts != module.runtime_counts:
        raise RuntimeError("checkpoint runtime metadata disagrees with restored persistent counters")
    rng_state = checkpoint.get("fullscale_relation_rng_state")
    _validate_relation_rng_state(rng_state)
    if restore_rng_state:
        restore_relation_rng_state(rng_state)
    module._assert_runtime_counts()
    return module
