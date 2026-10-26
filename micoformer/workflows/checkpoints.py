"""Save model snapshots and restore training state.
保存模型快照并恢复训练状态。

Training checkpoints include optimizer, scheduler, and per-rank RNG state; load_model restores only the model.
训练 checkpoint 包含优化器、调度器和各进程随机状态；load_model 仅恢复模型。
"""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from micoformer.config import (
    RunConfig,
    ModelConfig,
    TrainConfig,
)
from micoformer.models import MiCoFormer
from micoformer.utils.io import (
    atomic_torch,
    atomic_json,
    sha256_file,
)


def load_model_config(checkpoint: Path) -> ModelConfig:
    """Read model settings from a training checkpoint or model snapshot.
    从训练 checkpoint 或模型快照读取模型配置。
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return ModelConfig.from_checkpoint_dict(payload["model_config"])


def initialize_weights(model: MiCoFormer, checkpoint: Path) -> None:
    """Load base weights for a new optimization stage, keeping corpus token identity.
    为新优化阶段加载基座权重和先验，确保语料中的 token 编号一致。
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model_state_dict"]
    if tuple(payload["vocabulary"]) != tuple(model.vocabulary):
        raise ValueError("initial checkpoint must match the corpus vocabulary order")
    model.load_state_dict(state, strict=True)
    model.initialization = {"checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint)}


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    """Capture this rank's random streams. / 记录当前进程的随机状态。"""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device)
        if device.type == "cuda"
        else None,
    }


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    """Restore this rank's random streams. / 恢复当前进程的随机状态。"""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # RNG APIs require CPU byte tensors even when weights were loaded onto a GPU.
    # 即使权重加载到了 GPU，随机状态接口仍要求 CPU 字节张量。
    torch.set_rng_state(
        state["torch_cpu"].detach().to(device="cpu", dtype=torch.uint8).contiguous()
    )
    if device.type == "cuda":
        torch.cuda.set_rng_state(
            state["torch_cuda"].detach().cpu().contiguous(), device
        )


def _save_training_checkpoint(
    *,
    config: RunConfig,
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_score: float,
    meaningful_anchor: float,
    stale_epochs: int,
    lambdas: Mapping[str, float],
    preparation: Mapping[str, Any],
    validation: Mapping[str, Any],
    is_absolute_best: bool,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Gather rank states and save on rank 0. / 汇集各进程状态，由主进程保存。"""
    # Every rank participates; rank 0 receives RNG states in rank order.
    # 所有进程参与，主进程按 rank 顺序收集各自的随机状态。
    local_rng = _capture_rng_state(device)
    gathered_rng: list[dict[str, Any]] | None = [None] * world if rank == 0 else None  # type: ignore[list-item]
    dist.gather_object(local_rng, gathered_rng, dst=0)
    if rank != 0:
        dist.barrier()
        return None
    payload = {
        "checkpoint_kind": "training_checkpoint",
        "data_preparation": dict(preparation),
        "config": config.to_dict(),
        "model_config": ddp.module.config.to_dict(),
        "vocabulary": ddp.module.vocabulary,
        # Save the unwrapped model, including its persistent prior-table buffer.
        # 保存 DDP 内部模型，包含持久化的固定先验表。
        "model_state_dict": ddp.module.state_dict(),
        "initialization": getattr(ddp.module, "initialization", None),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        # Epoch is zero-based and identifies a completed training/validation round.
        # epoch 从 0 开始，记录已经完成训练与验证的一轮。
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "meaningful_anchor": meaningful_anchor,
        "stale_epochs": stale_epochs,
        "lambda": dict(lambdas),
        "rng_by_rank": gathered_rng,
        "validation": dict(validation),
    }
    # Replace last each epoch; replace best only when validation reaches a new low.
    # 每轮覆盖 last，验证损失刷新最佳值时覆盖 best；两者均支持续训。
    last_path = config.output_root / "run/checkpoints/last.pt"
    atomic_torch(last_path, payload)
    result = {"last": str(last_path), "last_sha256": sha256_file(last_path)}
    if is_absolute_best:
        best_path = config.output_root / "run/checkpoints/best.pt"
        atomic_torch(best_path, payload)
        result.update({"best": str(best_path), "best_sha256": sha256_file(best_path)})
    # Keep model-only snapshots at epoch 1 and every 5 epochs; these are for inference.
    # 第 1 轮及每 5 轮保留模型快照，不含优化器等续训状态。
    if epoch == 0 or (epoch + 1) % 5 == 0:
        milestone_path = config.output_root / f"run/milestones/epoch_{epoch + 1:03d}.pt"
        atomic_torch(
            milestone_path,
            {
                "checkpoint_kind": "model_checkpoint",
                "data_preparation": dict(preparation),
                "epoch": epoch,
                "global_step": global_step,
                "lambda": dict(lambdas),
                "model_config": ddp.module.config.to_dict(),
                "vocabulary": ddp.module.vocabulary,
                "model_state_dict": ddp.module.state_dict(),
                "initialization": payload["initialization"],
                "validation": dict(validation),
            },
        )
        result.update(
            {
                "milestone": str(milestone_path),
                "milestone_sha256": sha256_file(milestone_path),
            }
        )
    # Record paths/hashes; atomic writes replace files only after writing completes.
    # 记录文件路径和哈希；原子写入完成后才替换目标文件。
    atomic_json(config.output_root / "results/checkpoint_index_latest.json", result)
    # Other ranks wait here until rank 0 finishes writing.
    # 其他进程等待主进程写入完成后再继续。
    dist.barrier()
    return result


def _load_resume(
    path: Path,
    *,
    config: RunConfig,
    preparation: Mapping[str, Any],
    lambdas: Mapping[str, float],
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[int, int, float, float, int]:
    """Restore training and return progress counters. / 恢复训练状态并返回进度。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    # Output location and CPU worker count may change without changing training.
    # 允许迁移输出目录、调整 CPU worker 数；其余训练配置仍须完全一致。
    saved_config = dict(payload["config"])
    current_config = config.to_dict()
    for settings in (saved_config, current_config):
        settings.pop("output_root")
        # Expand defaults so checkpoints saved before optional LoRA fields still resume.
        # 补齐新增可选字段的默认值，兼容接入 LoRA 前的普通预训练 checkpoint。
        settings.setdefault("initialize_from", None)
        settings["model"] = ModelConfig.from_checkpoint_dict(settings["model"]).to_dict()
        settings["training"] = asdict(TrainConfig(**settings["training"]))
        settings["training"].pop("workers_per_rank")
    if (
        payload["checkpoint_kind"] != "training_checkpoint"
        or payload["data_preparation"] != preparation
        or saved_config != current_config
        or payload["lambda"] != dict(lambdas)
        or len(payload["rng_by_rank"]) != world
    ):
        raise RuntimeError("resume requires a training checkpoint with matching data, config, loss weights, and world size")
    # Restore existing objects; fixed buffers are included in the model state.
    # 恢复已经创建的对象；模型状态包含固定 buffer。
    ddp.module.load_state_dict(payload["model_state_dict"], strict=True)
    ddp.module.initialization = payload.get("initialization")
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_rng_state(payload["rng_by_rank"][rank], device)
    # Continue at the next epoch, preserving optimizer steps and early-stopping history.
    # 从下一轮继续，保留累计更新次数及早停记录。
    return (
        int(payload["epoch"]) + 1,
        int(payload["global_step"]),
        float(payload["best_score"]),
        float(payload["meaningful_anchor"]),
        int(payload["stale_epochs"]),
    )


def load_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
    expected_checkpoint_sha256: str | None = None,
) -> MiCoFormer:
    """Load a self-contained model for inference. / 加载自包含模型供推理使用。"""
    # Check file identity only when the caller supplies an expected hash.
    # 只有调用方指定预期哈希时，才核对文件指纹。
    if (
        expected_checkpoint_sha256
        and sha256_file(checkpoint) != expected_checkpoint_sha256
    ):
        raise ValueError("checkpoint SHA256 mismatch")
    # Full training checkpoints and model snapshots share these model fields.
    # 完整训练 checkpoint 与模型快照共用以下模型字段。
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ModelConfig.from_checkpoint_dict(payload["model_config"])
    # Rebuild the model using the saved prior table. / 使用已保存的先验表建立模型。
    state = payload["model_state_dict"]
    model = MiCoFormer(state["prior_table"], config)
    model.load_state_dict(state, strict=True)
    model.vocabulary = tuple(payload["vocabulary"])
    model.initialization = payload.get("initialization")
    # Inference needs no training corpus; eval disables training-time dropout.
    # 加载模型无需训练语料库；eval 关闭训练时的 Dropout。
    model.to(torch.device(device)).eval()
    return model
