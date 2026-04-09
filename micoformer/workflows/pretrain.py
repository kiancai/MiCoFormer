"""预训练 workflow：可复用的训练逻辑（不含 argparse）"""

from __future__ import annotations

import os
import time
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
    ff_dim: int = 1024
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

    # 5. 运行与工程参数
    devices: int = 1
    precision: str = "auto"
    seed: int = 42
    accumulate_grad_batches: int = 1
    gradient_clip_val: float = 1.0
    num_workers: int = 4
    log_dir: str = "tmp/logs"
    no_progress_bar: bool = False


def validate_pretrain_config(config: PretrainRunConfig) -> None:
    """检验预训练配置的合法性"""
    # 模型主体参数：基础正整性
    if config.d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {config.d_model}.")
    if config.nhead <= 0:
        raise ValueError(f"nhead must be > 0, got {config.nhead}.")
    if config.num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {config.num_layers}.")
    if config.ff_dim <= 0:
        raise ValueError(f"ff_dim must be > 0, got {config.ff_dim}.")
    if config.d_model % config.nhead != 0:
        raise ValueError(
            f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})."
        )
    if config.num_abundance_bins < 1:
        raise ValueError(
            f"num_abundance_bins must be >= 1, got {config.num_abundance_bins}."
        )
    if config.min_abundance <= 0:
        raise ValueError(
            f"min_abundance must be > 0 (used as log-bin lower bound), got {config.min_abundance}."
        )
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}.")

    # 训练协议参数
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must satisfy 0 <= warmup_ratio < 1.")

    if not 0.0 < config.lr_plateau_factor < 1.0:
        raise ValueError("lr_plateau_factor must satisfy 0 < lr_plateau_factor < 1.")
    if config.lr_plateau_patience < 0:
        raise ValueError("lr_plateau_patience must be >= 0.")
    if config.lr_plateau_min_lr < 0:
        raise ValueError("lr_plateau_min_lr must be >= 0.")

    # 预算模式互斥
    if config.budget_mode not in ("epoch", "step"):
        raise ValueError(
            f"budget_mode must be 'epoch' or 'step', got {config.budget_mode!r}."
        )

    # plateau scheduler 与 step 预算不兼容：plateau 按 epoch 触发，
    # step 模式下验证可能跨 epoch 触发，调度时机会错位
    if config.lr_scheduler_type == "plateau" and config.budget_mode == "step":
        raise ValueError(
            "lr_scheduler_type='plateau' is incompatible with budget_mode='step' "
            "(plateau steps per epoch, step-budget validates mid-epoch). "
            "Use budget_mode='epoch' with plateau, or use lr_scheduler_type='cosine' with step budget."
        )

    if config.budget_mode == "epoch":
        if config.max_epochs is None or config.max_epochs <= 0:
            raise ValueError("max_epochs must be provided and > 0 when budget_mode=epoch.")
        if config.val_interval_epochs is None or config.val_interval_epochs <= 0:
            raise ValueError("val_interval_epochs must be provided and > 0 when budget_mode=epoch.")
        if config.max_steps is not None:
            raise ValueError("max_steps is step-mode only. Do not pass it when budget_mode=epoch.")
        if config.val_interval_steps is not None:
            raise ValueError("val_interval_steps is step-mode only. Do not pass it when budget_mode=epoch.")
    else:
        if config.max_steps is None or config.max_steps <= 0:
            raise ValueError("max_steps must be provided and > 0 when budget_mode=step.")
        if config.val_interval_steps is None or config.val_interval_steps <= 0:
            raise ValueError("val_interval_steps must be provided and > 0 when budget_mode=step.")
        if config.max_epochs is not None:
            raise ValueError("max_epochs is epoch-mode only. Do not pass it when budget_mode=step.")
        if config.val_interval_epochs is not None:
            raise ValueError("val_interval_epochs is epoch-mode only. Do not pass it when budget_mode=step.")


def _choose_precision(precision: str) -> str:
    # 自动根据设备选择精度
    if precision == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32"
    return precision


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
    # （读元信息只用 backed 模式，不把 X 加载进内存）
    import anndata as ad
    _peek_adata = ad.read_h5ad(config.h5ad_path, backed="r")
    try:
        n_obs = int(_peek_adata.n_obs)
    finally:
        if getattr(_peek_adata, "file", None) is not None:
            _peek_adata.file.close()

    train_arr = np.asarray(train_indices)
    val_arr = np.asarray(val_indices)
    if train_arr.size == 0:
        raise ValueError("train_indices is empty.")
    if val_arr.size == 0:
        raise ValueError("val_indices is empty.")
    if int(train_arr.min()) < 0 or int(train_arr.max()) >= n_obs:
        raise ValueError(
            f"train_indices out of range [0, {n_obs}): "
            f"min={int(train_arr.min())}, max={int(train_arr.max())}. "
            f"Splits .npy probably comes from a different h5ad."
        )
    if int(val_arr.min()) < 0 or int(val_arr.max()) >= n_obs:
        raise ValueError(
            f"val_indices out of range [0, {n_obs}): "
            f"min={int(val_arr.min())}, max={int(val_arr.max())}. "
            f"Splits .npy probably comes from a different h5ad."
        )

    chosen_precision = _choose_precision(config.precision)
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
    # 显式锁定 version：让 CSVLogger 与 TensorBoardLogger 共享同一个 version 字符串，
    # 避免两者各自自增、出现 version_3 / version_5 错位
    run_version = time.strftime("run_%Y%m%d_%H%M%S")
    csv_logger = CSVLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)

    # 显式指定 dirpath，避免依赖 "第一个 logger 的 save_dir" 这种隐式行为
    ckpt_dir = os.path.join(config.log_dir, log_subdir, run_version, "checkpoints")
    # 注意：filename 中不能包含 monitor 的 'val/loss'，否则斜杠会被当作子目录创建
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
        "best_model_path": checkpoint_callback.best_model_path,
        "best_val_loss": best_score.item() if best_score is not None else None,
    }
