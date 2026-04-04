from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info

from micoformer.datamodules.pretrain_datamodule import MiCoDataModule
from micoformer.models.pretrain_module import MiCoFormerModule


TAG = "[train_pretrain]"


@dataclass(slots=True)
class PretrainRunConfig:
    h5ad_path: str
    token_embedding_mode: str
    use_taxonomy_bias: bool
    d_model: int
    nhead: int
    num_layers: int
    ff_dim: int
    num_abundance_bins: int
    abundance_mode: str
    min_abundance: float
    max_seq_len: int
    batch_size: int
    mask_prob: float
    dropout: float
    lr: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    lr_plateau_factor: float
    lr_plateau_patience: int
    lr_plateau_min_lr: float
    budget_mode: str
    max_epochs: int | None
    max_steps: int | None
    val_interval_epochs: int | None
    val_interval_steps: int | None
    limit_train_batches: float
    limit_val_batches: float
    devices: int
    precision: str
    seed: int
    accumulate_grad_batches: int
    gradient_clip_val: float
    num_workers: int
    log_dir: str
    no_progress_bar: bool


def validate_pretrain_config(config: PretrainRunConfig) -> None:
    # 检验训练配置的基础合法性
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must satisfy 0 <= warmup_ratio < 1.")
    if not 0.0 < config.lr_plateau_factor < 1.0:
        raise ValueError("--lr_plateau_factor must satisfy 0 < lr_plateau_factor < 1.")
    if config.lr_plateau_patience < 0:
        raise ValueError("--lr_plateau_patience must be >= 0.")
    if config.lr_plateau_min_lr < 0:
        raise ValueError("--lr_plateau_min_lr must be >= 0.")

    # 预算模式互斥，避免 epoch / step 协议混用
    if config.budget_mode == "epoch":
        if config.max_epochs is None or config.max_epochs <= 0:
            raise ValueError("--max_epochs must be provided and > 0 when budget_mode=epoch.")
        if config.val_interval_epochs is None or config.val_interval_epochs <= 0:
            raise ValueError("--val_interval_epochs must be provided and > 0 when budget_mode=epoch.")
        if config.max_steps is not None:
            raise ValueError("--max_steps is step-mode only. Do not pass it when budget_mode=epoch.")
        if config.val_interval_steps is not None:
            raise ValueError("--val_interval_steps is step-mode only. Do not pass it when budget_mode=epoch.")
        return

    if config.max_steps is None or config.max_steps <= 0:
        raise ValueError("--max_steps must be provided and > 0 when budget_mode=step.")
    if config.val_interval_steps is None or config.val_interval_steps <= 0:
        raise ValueError("--val_interval_steps must be provided and > 0 when budget_mode=step.")
    if config.max_epochs is not None:
        raise ValueError("--max_epochs is epoch-mode only. Do not pass it when budget_mode=step.")
    if config.val_interval_epochs is not None:
        raise ValueError("--val_interval_epochs is epoch-mode only. Do not pass it when budget_mode=step.")


def choose_precision(precision: str) -> str:
    # 自动根据设备选择精度，避免 CPU 下使用 16-mixed 报错
    if precision == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32"
    return precision


def load_indices(indices_path: str, split_name: str) -> np.ndarray:
    rank_zero_info(f"{TAG} Loading {split_name} indices from {indices_path} ...")
    return np.load(indices_path)


def run_pretrain_once(
    config: PretrainRunConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    *,
    log_subdir: str = "pretrain_stage0",
) -> dict[str, object]:
    validate_pretrain_config(config)
    L.seed_everything(config.seed, workers=True)

    chosen_precision = choose_precision(config.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)}")
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
        f"layers={config.num_layers}"
    )
    rank_zero_info(f"{TAG} Token embedding mode: {config.token_embedding_mode}")
    model = MiCoFormerModule(
        genus_vocab_size=dm.genus_vocab_size,
        total_abundance_bins=dm.total_abundance_bins,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.ff_dim,
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
    csv_logger = CSVLogger(save_dir=config.log_dir, name=log_subdir)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=log_subdir)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        filename="micoformer-{epoch:02d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. 初始化 Trainer
    trainer_kwargs = dict(
        devices=config.devices,
        precision=chosen_precision,
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        limit_train_batches=config.limit_train_batches,
        limit_val_batches=config.limit_val_batches,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback, lr_monitor],
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
        "log_subdir": log_subdir,
        "precision": chosen_precision,
        "best_model_path": checkpoint_callback.best_model_path,
        "best_val_loss": best_score.item() if best_score is not None else None,
    }

