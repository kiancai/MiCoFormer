#!/usr/bin/env python3
"""Single-GPU fixed-PP MPC continuation with optional UniFrac shaping."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from micoformer.mpc_fixed_pretraining.model import MPCFixedModelConfig, MPCFixedPretrainingModel

from .data import PilotCollator, PilotRowDataset
from .losses import (
    LOSS_PARAMETERIZATION_ID,
    LossWeights,
    SoftNeighborhoodConfig,
    soft_neighborhood_loss,
    weighted_loss_components,
)
from .teacher import UniFracTeacher


CHECKPOINT_KIND = "mpc_model_milestone"
CHECKPOINT_STEP = 264_450
CHECKPOINT_MANIFEST_SHA256 = "618e4534cd79125ccd2dd9041c3f4a655a9dc5a5fa3951d31288c9627a6be121"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PilotTrainConfig:
    corpus: Path
    schedule_npz: Path
    prior_assets: Path
    checkpoint: Path
    incidence: Path
    branch_lengths: Path
    output_root: Path
    arm: str = "control"
    steps: int = 32
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 10
    mlm_weight: float = 1.0
    query_continuous_weight: float = 0.534399128791011
    query_presence_weight: float = 0.24296011504612833
    unifrac_weight: float = 0.05
    teacher_temperature: float = 0.02
    student_temperature: float = 0.10
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.arm not in {"control", "random", "biome_study"}:
            raise ValueError("arm must be control, random, or biome_study")
        if self.steps <= 0 or self.batch_size <= 1:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        if self.warmup_steps < 0 or self.grad_clip <= 0:
            raise ValueError("invalid warmup or gradient clip")
        LossWeights(
            mlm=self.mlm_weight,
            query_continuous=self.query_continuous_weight,
            query_presence=self.query_presence_weight,
            unifrac=self.unifrac_weight,
        )


def load_fixed_checkpoint(config: PilotTrainConfig) -> tuple[MPCFixedPretrainingModel, np.ndarray]:
    checkpoint_hash = sha256_file(config.checkpoint)
    if checkpoint_hash != "1dcd2030bc4eda6289cde3f938dd2641e77fd3a7007d5e9bae664b7ad8fdee18":
        raise RuntimeError("fixed-PP checkpoint hash differs from the contract")
    payload = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("checkpoint_kind") != CHECKPOINT_KIND
        or int(payload.get("global_step", -1)) != CHECKPOINT_STEP
        or payload.get("manifest_sha256") != CHECKPOINT_MANIFEST_SHA256
    ):
        raise RuntimeError("fixed-PP checkpoint identity differs from the contract")
    with np.load(config.prior_assets, allow_pickle=False) as archive:
        table = torch.from_numpy(np.asarray(archive["pp_table"], dtype=np.float32))
        observed = np.asarray(archive["observed_genus_ids"], dtype=np.int64)
    model = MPCFixedPretrainingModel(
        table,
        config=MPCFixedModelConfig(**payload["model_config"]),
    )
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("fixed-PP checkpoint failed strict restore")
    return model, observed


def choose_schedule(config: PilotTrainConfig) -> np.ndarray:
    with np.load(config.schedule_npz, allow_pickle=False) as archive:
        schedule = np.asarray(archive["schedule_rows"], dtype=np.int64)
        train_rows = np.asarray(archive["train_rows"], dtype=np.int64)
    if schedule.ndim != 2 or schedule.shape[1] != config.batch_size:
        raise RuntimeError("cached schedule shape does not match configured batch size")
    if config.arm == "random":
        required = schedule.size
        if train_rows.size < required:
            raise RuntimeError("random arm needs at least as many rows as the cached schedule")
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, 0x52414E]))
        rows = train_rows[rng.permutation(train_rows.size)[:required]]
        schedule = rows.reshape(schedule.shape)
    if config.arm == "control":
        # Control uses the same label-organised schedule as the main UniFrac arm.
        pass
    return schedule


def make_optimizer(model: torch.nn.Module, config: PilotTrainConfig) -> AdamW:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    return AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)


def set_learning_rate(optimizer: AdamW, config: PilotTrainConfig, step: int) -> float:
    if config.warmup_steps and step < config.warmup_steps:
        scale = float(step + 1) / float(config.warmup_steps)
    else:
        progress = 0.0 if config.steps <= config.warmup_steps else (
            step - config.warmup_steps
        ) / (config.steps - config.warmup_steps)
        scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))))
    lr = config.learning_rate * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def run(config: PilotTrainConfig) -> dict[str, Any]:
    output_dir = config.output_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    summary_path = output_dir / "summary.json"
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    model, observed = load_fixed_checkpoint(config)
    model = model.to(device)
    teacher = UniFracTeacher(
        incidence_path=config.incidence,
        branch_lengths_path=config.branch_lengths,
        expected_n_genus=8_114,
    ).to(device).eval()
    schedule = choose_schedule(config)
    dataset = PilotRowDataset(config.corpus, schedule.reshape(-1))
    collator = PilotCollator(observed_genus_ids=observed, seed=config.seed, epoch=0)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        drop_last=True,
    )
    optimizer = make_optimizer(model, config)
    relation_config = SoftNeighborhoodConfig(
        teacher_temperature=config.teacher_temperature,
        student_temperature=config.student_temperature,
    )
    model.train()
    loss_weights = LossWeights(
        mlm=config.mlm_weight,
        query_continuous=config.query_continuous_weight,
        query_presence=config.query_presence_weight,
        unifrac=config.unifrac_weight,
    )
    started = time.time()
    rows: list[dict[str, Any]] = []
    for step, batch in enumerate(loader):
        if step >= config.steps:
            break
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = device.type == "cuda"
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            model_output = model(
                genus_ids=batch["genus_ids"],
                rclr=batch["rclr"],
                padding_mask=batch["padding_mask"],
                query_mask=batch["query_mask"],
                mlm_mask=batch["mlm_mask"],
                negative_genus_ids=batch["negative_genus_ids"],
            )
            if config.arm == "control" or config.unifrac_weight == 0:
                unifrac_loss = torch.zeros((), device=device)
                teacher_mean = torch.zeros((), device=device)
            else:
                with torch.no_grad():
                    teacher_distance = teacher(batch["full_composition"])
                    mask = ~torch.eye(config.batch_size, dtype=torch.bool, device=device)
                    teacher_mean = teacher_distance[mask].mean()
                _, h_raw = model.unmasked_representations(
                    batch["genus_ids"], batch["rclr"], batch["padding_mask"]
                )
                unifrac_loss = soft_neighborhood_loss(h_raw, teacher_distance, relation_config)
            raw_losses, weighted_losses, total = weighted_loss_components(
                mlm=model_output.mlm_loss,
                query_continuous=model_output.continuous_loss,
                query_presence=model_output.presence_loss,
                unifrac=unifrac_loss,
                weights=loss_weights,
            )
        if not bool(torch.isfinite(total)):
            raise RuntimeError(f"non-finite loss at step {step}")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        lr = set_learning_rate(optimizer, config, step)
        optimizer.step()
        record = {
            "step": step,
            "lr": lr,
            "loss_parameterization_id": LOSS_PARAMETERIZATION_ID,
            "loss_weights": loss_weights.as_dict(),
            "loss_components": {
                name: float(value.detach().cpu())
                for name, value in raw_losses.items()
            },
            "weighted_components": {
                name: float(value.detach().cpu())
                for name, value in weighted_losses.items()
            },
            "total_loss": float(total.detach().cpu()),
            "teacher_mean": float(teacher_mean.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu()),
        }
        rows.append(record)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(json.dumps(record, sort_keys=True), flush=True)
    checkpoint_path = output_dir / f"{config.arm}_step_{len(rows):04d}.pt"
    torch.save(
        {
            "checkpoint_kind": "mpc_fixedpp_unifrac_pilot",
            "loss_parameterization_id": LOSS_PARAMETERIZATION_ID,
            "loss_weights": loss_weights.as_dict(),
            "arm": config.arm,
            "steps": len(rows),
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "config": {**asdict(config), "corpus": str(config.corpus), "schedule_npz": str(config.schedule_npz), "prior_assets": str(config.prior_assets), "checkpoint": str(config.checkpoint), "incidence": str(config.incidence), "branch_lengths": str(config.branch_lengths), "output_root": str(config.output_root)},
        },
        checkpoint_path,
    )
    summary = {
        "status": "PASS" if rows else "EMPTY",
        "arm": config.arm,
        "steps": len(rows),
        "seconds": time.time() - started,
        "last": rows[-1] if rows else None,
        "checkpoint": str(checkpoint_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> PilotTrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--schedule-npz", type=Path, required=True)
    parser.add_argument("--prior-assets", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--incidence", type=Path, required=True)
    parser.add_argument("--branch-lengths", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--arm", choices=("control", "random", "biome_study"), default="control")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--mlm-weight", type=float, default=1.0)
    parser.add_argument("--query-continuous-weight", type=float, default=0.534399128791011)
    parser.add_argument("--query-presence-weight", type=float, default=0.24296011504612833)
    parser.add_argument("--unifrac-weight", type=float, default=0.05)
    parser.add_argument("--teacher-temperature", type=float, default=0.02)
    parser.add_argument("--student-temperature", type=float, default=0.10)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    return PilotTrainConfig(**vars(args))


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
