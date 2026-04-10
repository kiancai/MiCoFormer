"""预训练 workflow：可复用的训练逻辑（不含 argparse）"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass

import anndata as ad
import lightning as L
import numpy as np
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info

from micoformer.datamodules.pretrain_datamodule import MiCoDataModule
from micoformer.models.pretrain_module import MiCoFormerModule
from micoformer.utils.train_utils import (
    choose_precision,
    validate_index_arrays,
    validate_pretrain_config,
)


TAG = "[train_pretrain]"


@dataclass
class PretrainRunConfig:
    # 0. 输入与切分参数
    h5ad_path: str

    # 1. 模型版本开关
    token_embedding_mode: str = "taxon_path"
    use_taxonomy_bias: bool = False

    # 2.1. 模型主体参数
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    # ff_dim 与 ff_ratio 互斥：指定其中一个，另一个保持 None
    ff_dim: int | None = None      # FeedForward 绝对维度
    ff_ratio: int | None = 4       # FeedForward 比例（dim_ff = d_model × ff_ratio），默认
    num_abundance_bins: int = 40

    # 2.2. 模型主体参数的协议参数
    abundance_mode: str = "abs_log_bins"
    min_abundance: float = 4e-6
    max_seq_len: int = 1024

    # 3.1. 预训练中的训练主体参数
    batch_size: int = 32
    mask_prob: float = 0.15
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.02

    # 3.2. 预训练中的协议参数
    lr_scheduler_type: str = "cosine"
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    lr_plateau_min_lr: float = 1e-6

    # 4. 预算与验证协议参数
    budget_mode: str = "epoch"
    max_epochs: int | None = None
    max_steps: int | None = None
    val_interval_epochs: int | None = None
    val_interval_steps: int | None = None
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0

    # 4.1. Early stopping（0=禁用，与 finetune 对称）
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0

    # 5. 运行与工程参数
    devices: int = 1
    precision: str = "auto"
    seed: int = 42
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    num_workers: int = 4
    log_dir: str = "tmp/logs"
    no_progress_bar: bool = False


# 执行一次完整的预训练流程，返回结果字典
def run_pretrain_once(
    config: PretrainRunConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    log_subdir: str = "pretrain_stage0",
) -> dict[str, object]:
    validate_pretrain_config(config)
    L.seed_everything(config.seed, workers=True)

    # 索引越界 sanity check：避免使用了和 h5ad 不匹配的 splits 文件
    n_obs = validate_index_arrays(
        config.h5ad_path,
        train=train_indices,
        val=val_indices,
    )

    # 计算有效的 dim_feedforward（ff_ratio 或 ff_dim 二选一）
    if config.ff_dim is not None:
        effective_ff_dim = config.ff_dim
    else:
        effective_ff_dim = config.d_model * config.ff_ratio  # type: ignore[operator]

    chosen_precision = choose_precision(config.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)} (n_obs={n_obs})")
    rank_zero_info(f"{TAG} Budget mode: {config.budget_mode}")
    if config.budget_mode == "epoch":
        rank_zero_info(
            f"{TAG} Max epochs: {config.max_epochs}, "
            f"val_interval_epochs={config.val_interval_epochs}"
        )
    else:
        rank_zero_info(
            f"{TAG} Max steps: {config.max_steps}, "
            f"val_interval_steps={config.val_interval_steps}"
        )
    rank_zero_info(
        f"{TAG} LR scheduler: {config.lr_scheduler_type}, "
        f"warmup_ratio={config.warmup_ratio}"
    )

    # 1. 初始化数据模块
    rank_zero_info(f"{TAG} Initializing DataModule...")
    dm = MiCoDataModule(
        h5ad_path=config.h5ad_path,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=None,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_seq_len=config.max_seq_len,
        mask_prob=config.mask_prob,
        num_abundance_bins=config.num_abundance_bins,
        min_abundance=config.min_abundance,
        abundance_mode=config.abundance_mode,
    )

    # 2. 初始化模型
    rank_zero_info(
        f"{TAG} Initializing Model with d_model={config.d_model}, "
        f"layers={config.num_layers}, ff_dim={effective_ff_dim}"
    )
    rank_zero_info(f"{TAG} Token embedding mode: {config.token_embedding_mode}")
    model = MiCoFormerModule(
        genus_vocab_size=dm.genus_vocab_size,
        total_abundance_bins=dm.total_abundance_bins,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=effective_ff_dim,
        dropout=config.dropout,
        pad_taxon_id=dm.special_ids["pad_taxon_id"],
        pad_bin_id=dm.special_ids["pad_bin_id"],
        token_embedding_mode=config.token_embedding_mode,
        rank_vocab_sizes=dm.rank_vocab_sizes,
        use_taxonomy_bias=config.use_taxonomy_bias,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler=config.lr_scheduler_type,
        plateau_factor=config.lr_plateau_factor,
        plateau_patience=config.lr_plateau_patience,
        plateau_min_lr=config.lr_plateau_min_lr,
    )

    # 3. 设置日志记录器与回调
    # 加 uuid 后缀避免同秒并行启动的时间戳碰撞
    run_version = time.strftime("run_%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:6]}"
    csv_logger = CSVLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)

    # 显式指定 dirpath，避免依赖 "第一个 logger 的 save_dir" 这种隐式行为
    ckpt_dir = os.path.join(config.log_dir, log_subdir, run_version, "checkpoints")
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
        filename="micoformer-epoch{epoch:02d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]
    if config.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        )

    # 4. 初始化 Trainer
    trainer_kwargs = dict(
        devices=config.devices,
        precision=chosen_precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        default_root_dir=config.log_dir,
    )
    if config.budget_mode == "epoch":
        trainer_kwargs["max_epochs"] = config.max_epochs
        trainer_kwargs["check_val_every_n_epoch"] = config.val_interval_epochs
    else:
        trainer_kwargs["max_epochs"] = -1
        trainer_kwargs["max_steps"] = config.max_steps
        trainer_kwargs["val_check_interval"] = config.val_interval_steps
    if config.no_progress_bar:
        trainer_kwargs["enable_progress_bar"] = False

    trainer = L.Trainer(**trainer_kwargs)

    # 5. 开始训练
    rank_zero_info(f"{TAG} Starting training...")
    trainer.fit(model, datamodule=dm)

    best_score = checkpoint_callback.best_model_score
    return {
        "best_model_path": checkpoint_callback.best_model_path,
        "best_score": best_score.item() if best_score is not None else None,
        "best_val_loss": best_score.item() if best_score is not None else None,  # 向后兼容别名
        "val_metrics": {},   # 预训练暂不做额外 validate，保留 key 供协议消费
        "test_metrics": None,
    }
