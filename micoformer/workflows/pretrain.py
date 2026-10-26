"""Run pretraining or domain adaptation with shared validation and checkpointing.
执行预训练或领域适配，共用验证与 checkpoint 保存。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from micoformer.config import RunConfig, TrainConfig, REFERENCE_LAMBDAS
from micoformer.data.genus import load_genus_data
from micoformer.data.unifrac import load_unifrac_assets
from micoformer.models import MiCoFormer
from micoformer.models.lora import LoRAProjection, add_lora
from micoformer.data.collate import batch_to_device
from micoformer.data.loaders import TrainingData

from micoformer.workflows.checkpoints import (
    _load_resume,
    _save_training_checkpoint,
    initialize_weights,
)
from micoformer.utils.io import (
    sha256_file,
    atomic_json,
    append_jsonl,
)
from .distributed import _init_distributed, _seed_all

from .validation import (
    _validate_epoch,
    LossStatistics,
    _weighted_loss,
)


def validate_preparation(path: Path, corpus: Path, train: Path, val: Path) -> dict:
    """Match corpus and splits to their recorded hashes. / 核对语料与划分的记录指纹。"""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    files = (
        (corpus, record["corpus"]["sha256"]),
        (train, record["splits"]["train"]["sha256"]),
        (val, record["splits"]["val"]["sha256"]),
    )
    for file, expected in files:
        if sha256_file(file) != expected:
            raise ValueError(f"SHA256 mismatch: {file}")
    print("Corpus and split SHA256 checks passed.", flush=True)
    return record


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    base_lr: float,
    minimum_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warm up, then decay to the minimum learning rate with a cosine curve.
    先预热，再沿余弦曲线衰减至最低学习率。
    """

    # Count optimizer updates, not samples or epochs.
    # 按参数更新次数计算，不按样本数或轮数。
    warmup_steps = int(round(total_steps * warmup_ratio))

    # LambdaLR expects a multiplier of the base learning rate.
    # LambdaLR 使用基础学习率的倍率。
    minimum_factor = minimum_lr / base_lr

    # Increase during warmup, then apply cosine decay.
    # 预热阶段逐步升高，之后进行余弦衰减。
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps

        progress = min(
            max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0
        )

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    # Initialize the learning rate; scheduler.step() advances it after each update.
    # 设置初始学习率；每次参数更新后由 scheduler.step() 推进。
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _make_optimizer(model: MiCoFormer, config: TrainConfig) -> torch.optim.AdamW:
    """Update the whole model, or LoRA A/B and optional prediction heads.
    普通预训练更新全模型；适配更新 LoRA A/B 及显式开启的预测头。
    """
    if model.config.lora_rank == 0:
        groups = [{"params": list(model.parameters()), "lr": config.learning_rate, "name": "model"}]
    else:
        projections = [module for module in model.modules() if isinstance(module, LoRAProjection)]
        groups = [
            {"params": [module.lora_A for module in projections], "lr": config.learning_rate, "name": "lora_A"},
            {"params": [module.lora_B for module in projections], "lr": config.learning_rate * config.lora_lr_ratio, "name": "lora_B"},
        ]
        if config.train_heads:
            heads = [model.mlm_head, model.query_head]
            for head in heads:
                head.requires_grad_(True)
            groups.append({"params": [p for head in heads for p in head.parameters()],
                           "lr": config.head_learning_rate, "name": "heads"})
    return torch.optim.AdamW(groups, lr=config.learning_rate, weight_decay=config.weight_decay)


def run_training(config: RunConfig, *, resume: Path | None = None) -> Path:
    """Train or resume a run and return its completion-record path.
    开始或恢复训练，返回正常完成记录的路径。
    """

    # Require explicit resume before reusing a run with a saved checkpoint.
    # 已有训练 checkpoint 时，必须显式续训。
    if resume is None and (config.output_root / "run/checkpoints/last.pt").exists():
        raise RuntimeError(
            "existing training checkpoint found; pass --resume explicitly"
        )
    # Preparation record: corpus/split hashes, saved in logs and checkpoints.
    # 准备记录：语料与划分指纹，将写入日志和 checkpoint。
    preparation = validate_preparation(
        config.data.preparation_manifest,
        config.data.corpus,
        config.data.train_rows,
        config.data.val_rows,
    )

    # GenusData: prior table [8116, D], vocabulary, and observed genus IDs.
    # 菌属数据：先验表 [8116, D]、名称表和已观测菌属编号。
    genus_data = load_genus_data(config.data.corpus)

    # Use the fixed loss coefficients for the selected identity type.
    # 使用所选身份输入模式对应的固定损失系数。
    lambdas = dict(REFERENCE_LAMBDAS[config.model.identity_type])
    unifrac_assets = None
    model_options = {}
    objective_notes = {}
    if config.training.unifrac_weight != 0:
        unifrac_assets = load_unifrac_assets(config.data.corpus)
        # Bind the tree to this run and to strict checkpoint resumption.
        # 树资产指纹写入本次运行，并参与严格的断点恢复核对。
        preparation["unifrac"] = unifrac_assets.identity
        lambdas["unifrac"] = config.training.unifrac_weight
        model_options = {
            "unifrac_lengths": unifrac_assets.branch_lengths,
            "teacher_temperature": config.training.unifrac_teacher_temperature,
            "student_temperature": config.training.unifrac_student_temperature,
        }
        objective_notes["unifrac"] = {
            "neighborhood": "rank-local batch; no cross-rank candidate gathering",
            "pool_size_per_rank": config.training.batch_size_per_rank,
            "singleton_policy": "zero effective targets; excluded from objective means",
        }

    # Select this process's GPU and initialize distributed communication.
    # 选择当前进程的 GPU，并初始化分布式通信。
    rank, world, _, device = _init_distributed()

    try:
        # Initialize identical model weights on all ranks.
        # 各进程使用相同种子建立模型初始权重。
        _seed_all(config.training.seed)

        # Load the original model strictly before adding new low-rank parameters.
        # 先严格加载原模型，再加入新的低秩参数；恢复任务直接由 resume 还原权重。
        model = MiCoFormer(genus_data.prior_table, replace(config.model, lora_rank=0), **model_options)

        # Keep genus names with the model for checkpoint export.
        # 将菌属名称保存在模型中，供 checkpoint 导出。
        model.vocabulary = genus_data.vocabulary
        if config.initialize_from is not None and resume is None:
            initialize_weights(model, config.initialize_from)
        if config.model.lora_rank > 0:
            add_lora(model, config.model)
        model.to(device)
        # Enable any optional head parameters before DDP collects trainable tensors.
        # 在 DDP 收集训练参数前处理预测头开关。
        optimizer = _make_optimizer(model, config.training)

        # Separate subsequent random streams by rank.
        # 后续随机过程按进程编号区分。
        _seed_all(config.training.seed + 10_000 * rank)

        # Build per-rank loaders, samplers, and shared epoch state.
        # 创建各进程的数据读取器、采样器与共享 epoch 状态。
        training_data = TrainingData.from_config(
            config, rank=rank, world=world, observed=genus_data.observed_genus_ids,
            unifrac_assets=unifrac_assets,
        )

        # DDP synchronizes gradients during backward propagation.
        # DDP 在反向传播期间同步各进程的梯度。
        ddp = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,

            # Keep fixed buffers local; reuse gradient buckets with a static graph.
            # 固定 buffer 不重复广播；使用梯度 bucket 视图与静态计算图。
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )

        # One update per batch; early stopping may shorten the planned schedule.
        # 每批更新一次参数；早停可能使实际更新次数少于计划值。
        total_steps = config.training.max_epochs * len(training_data.train_loader)
        scheduler = _make_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=config.training.warmup_ratio,
            base_lr=config.training.learning_rate,
            minimum_lr=config.training.minimum_lr,
        )

        # Track the next epoch, completed updates, best score, and early-stop state.
        # 记录下一轮编号、累计更新次数、最佳分数和早停状态。
        start_epoch = 0
        global_step = 0

        best_score = math.inf
        meaningful_anchor = math.inf
        stale_epochs = 0

        # Restore weights, optimizer, scheduler, progress, and per-rank random states.
        # 恢复权重、优化器、学习率进度、训练进度及各进程随机状态。
        if resume is not None:
            (
                start_epoch,
                global_step,
                best_score,
                meaningful_anchor,
                stale_epochs,
            ) = _load_resume(
                resume.resolve(),
                config=config,
                preparation=preparation,
                lambdas=lambdas,
                ddp=ddp,
                optimizer=optimizer,
                scheduler=scheduler,
                rank=rank,
                world=world,
                device=device,
            )

        # Only rank 0 writes the shared run record.
        # 仅主进程写入公共启动记录。
        if rank == 0:
            atomic_json(
                config.output_root / "audit/training_run.json",
                {
                    "status": "RUNNING",
                    "config": config.to_dict(),
                    "data_preparation": preparation,
                    "lambda": lambdas,
                    "start_epoch": start_epoch,
                    "start_global_step": global_step,
                    "resume": str(resume.resolve()) if resume is not None else None,
                    "world_size": world,
                    "batch_size_per_rank": config.training.batch_size_per_rank,

                    # Each sample produces separate MLM and Query encoder views.
                    # 每个样本分别产生 MLM 和 Query 两条编码分支。
                    "global_sample_batch": config.training.batch_size_per_rank * world,
                    "encoder_sample_views_per_update": config.training.batch_size_per_rank
                    * world
                    * 2,
                    "steps_per_epoch": len(training_data.train_loader),
                    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "objective_notes": objective_notes,
                    "initialization": getattr(model, "initialization", None),
                    "optimizer_groups": [
                        {"name": group.get("name", "model"), "initial_lr": group["initial_lr"],
                         "parameters": sum(p.numel() for p in group["params"])}
                        for group in optimizer.param_groups
                    ],
                },
            )

        # Append batch progress and epoch summaries as JSONL records.
        # 以 JSONL 逐条追加批次进度与整轮汇总。
        progress_path = config.output_root / "logs/progress.jsonl"
        epoch_path = config.output_root / "results/epochs.jsonl"

        final_reason = "MAX_EPOCHS"
        last_completed_epoch = start_epoch

        for epoch in range(start_epoch, config.training.max_epochs):
            # Enable training behavior and refresh sample order and corruption for this epoch.
            # 启用训练模式，并更新本轮样本顺序和任务扰动。
            ddp.train()

            training_data.set_train_epoch(epoch)

            # Measure training/validation time and peak reserved GPU memory.
            # 记录训练与验证耗时，以及 GPU 预留显存峰值。
            epoch_started = perf_counter()
            torch.cuda.reset_peak_memory_stats(device)

            train_statistics = LossStatistics(device)

            for batch_index, batch in enumerate(training_data.train_loader):
                # Move model inputs to this GPU; each batch performs one optimizer update.
                # 将模型输入搬到当前 GPU；每批进行一次参数更新。
                arguments = batch_to_device(batch, device)

                optimizer.zero_grad(set_to_none=True)

                # Use BF16 autocast for the two model branches.
                # 对模型两条前向分支使用 BF16 混合精度。
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = ddp(**arguments)

                    total = _weighted_loss(output, lambdas)

                # Stop on invalid loss before updating parameters.
                # 损失出现 NaN 或无穷大时，在更新参数前终止。
                if not bool(torch.isfinite(total)):
                    raise RuntimeError(
                        f"non-finite total loss at epoch={epoch}, batch={batch_index}"
                    )

                # Compute gradients; DDP averages them across ranks.
                # 计算梯度；DDP 在各进程间取平均。
                total.backward()

                # Reject nonfinite gradients and clip their norm; log the pre-clip norm.
                # 拒绝非有限梯度并裁剪范数；记录裁剪前的范数。
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    ddp.parameters(),
                    config.training.gradient_clip,
                    error_if_nonfinite=True,
                )

                # Update parameters, then set the learning rate for the next batch.
                # 先更新参数，再设置下一批的学习率。
                optimizer.step()
                scheduler.step()

                # Detach loss statistics from the computation graph before accumulating.
                # 将损失统计与计算图分离后累计。
                train_statistics.update(output, total)

                global_step += 1

                # Log rank-0 batch values, not global epoch averages.
                # 记录主进程当前批次的数值，不是跨卡整轮均值。
                if rank == 0 and (batch_index == 0 or (batch_index + 1) % 100 == 0):
                    append_jsonl(
                        progress_path,
                        {
                            "epoch": epoch + 1,
                            "batch": batch_index + 1,
                            "batches_per_epoch": len(training_data.train_loader),
                            "global_step": global_step,
                            "weighted_total_local": float(total.detach().cpu()),
                            "objectives_local": {
                                name: {
                                    "loss": float(result.loss.detach().cpu()),
                                    "count": result.count,
                                    "diagnostics": {
                                        key: float(value.detach().cpu())
                                        for key, value in result.diagnostics.items()
                                    },
                                } for name, result in output.items()
                            },
                            "gradient_norm_preclip_local": float(
                                gradient_norm.detach().cpu()
                            ),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                            "elapsed_seconds": perf_counter() - epoch_started,
                        },
                    )

            # Aggregate all ranks; average each loss over its prediction targets.
            # 汇总各进程统计，按各项预测目标数计算损失均值。
            train_statistics.synchronize()
            training_loss = train_statistics.summary(lambdas)

            # Validate with fixed corruption and no updates; also monitor unmasked representations.
            # 使用固定扰动验证，不更新参数；同时监测无遮盖表征。
            training_data.reset_validation()
            validation = _validate_epoch(
                ddp,
                training_data.val_loader,
                lambdas=lambdas,
                rank=rank,
                device=device,
            )

            # Report the largest peak reserved memory across GPUs, in GiB.
            # 记录各 GPU 中最大的预留显存峰值，单位 GiB。
            peak = torch.tensor(
                torch.cuda.max_memory_reserved(device) / 2**30, device=device
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            peak_reserved = float(peak.cpu())

            # Select the best checkpoint by weighted validation loss; lower is better.
            # 按加权验证损失选择最佳 checkpoint；越低越好。
            score = float(validation["loss"]["weighted_total"])

            is_absolute_best = score < best_score
            if is_absolute_best:
                best_score = score

            # Reset patience only after a sufficient relative improvement from the anchor.
            # 相对参考值改善达到阈值时，才重置早停等待计数。
            if not math.isfinite(meaningful_anchor) or score <= meaningful_anchor * (
                1.0 - config.training.relative_min_improvement
            ):
                meaningful_anchor = score
                stale_epochs = 0
                meaningful_improvement = True
            else:
                stale_epochs += 1
                meaningful_improvement = False
            last_completed_epoch = epoch + 1
            epoch_seconds = perf_counter() - epoch_started

            # Epoch time includes validation but excludes the checkpoint writes below.
            # 本轮耗时包含验证，不包含下方 checkpoint 写入。
            epoch_record = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "training_loss": training_loss,
                "validation": validation,
                "lambda": lambdas,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": epoch_seconds,
                "global_samples_per_second": len(training_data.train_rows) / epoch_seconds,
                "peak_reserved_gib": peak_reserved,
                "absolute_best": is_absolute_best,
                "best_score": best_score,
                "meaningful_improvement": meaningful_improvement,
                "meaningful_anchor": meaningful_anchor,
                "stale_epochs": stale_epochs,
            }
            if rank == 0:
                append_jsonl(epoch_path, epoch_record)
                print(json.dumps(epoch_record, sort_keys=True), flush=True)

            # All ranks supply random states; rank 0 writes last/best and periodic snapshots.
            # 各进程提供随机状态，主进程保存 last/best 及周期模型快照。
            checkpoint_record = _save_training_checkpoint(
                config=config,
                ddp=ddp,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                meaningful_anchor=meaningful_anchor,
                stale_epochs=stale_epochs,
                lambdas=lambdas,
                preparation=preparation,
                validation=validation,
                is_absolute_best=is_absolute_best,
                rank=rank,
                world=world,
                device=device,
            )
            if rank == 0:
                append_jsonl(
                    config.output_root / "results/checkpoints.jsonl",
                    {
                        "epoch": epoch + 1,
                        **(checkpoint_record or {}),
                    },
                )

            # Stop only after the minimum epoch count and the patience limit are both reached.
            # 同时达到最少训练轮数和等待上限时，才提前结束。
            if (
                config.training.early_stopping_patience is not None
                and epoch + 1 >= config.training.min_epochs
                and stale_epochs >= config.training.early_stopping_patience
            ):
                final_reason = "LABEL_FREE_SATURATION"
                break

        # Record normal completion after reaching the epoch limit or early stopping.
        # 跑满轮数或早停后记录正常完成状态。
        result = {
            "status": "TRAINING_COMPLETE",
            "reason": final_reason,
            "last_completed_epoch": last_completed_epoch,
            "global_step": global_step,
            "best_score": best_score,
            "meaningful_anchor": meaningful_anchor,
            "stale_epochs": stale_epochs,
        }
        if rank == 0:
            path = config.output_root / "results/training_complete.json"
            atomic_json(path, result)
            launch_path = config.output_root / "audit/training_run.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["status"] = "TRAINING_COMPLETE"
            launch["result"] = result
            atomic_json(launch_path, launch)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)

        # Wait for all ranks before returning the completion-record path.
        # 等待各进程完成后，返回完成记录路径。
        dist.barrier()
        return config.output_root / "results/training_complete.json"

    # Release data-loading workers, file handles, and the distributed process group.
    # 释放数据加载子进程、文件句柄和分布式通信组。
    finally:
        if "training_data" in locals():
            training_data.close()
        if dist.is_initialized():
            dist.destroy_process_group()
