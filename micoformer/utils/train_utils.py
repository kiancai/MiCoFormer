"""共享训练工具函数与参数检验（供 workflows 层和 model 层复用）"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

# 微调合法的 pooling 模式（validate_finetune_config 使用）
_VALID_POOLING_MODES = frozenset({"sample", "mean_pool", "sample_and_mean"})


def choose_precision(precision: str) -> str:
    """自动根据设备选择精度：auto → '16-mixed'（GPU）或 '32'（CPU）"""
    if precision == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32"
    return precision


def validate_budget_and_lr_config(config: Any) -> None:
    """
    检验 budget 模式与 LR scheduler 的共享约束（PretrainRunConfig / FinetuneRunConfig 均适用）。

    检查项：
    - warmup_ratio 范围
    - plateau 参数合法性
    - plateau + step 不兼容
    - epoch/step 预算参数互斥完整性
    """
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


def validate_index_arrays(h5ad_path: str, **splits: "np.ndarray | None") -> int:
    """
    验证各 split 的索引数组不越界、不为空。以 backed='r' 只读模式打开 h5ad，仅读取样本数。

    参数：
        h5ad_path: h5ad 文件路径
        **splits: 命名关键字参数，如 train=train_arr, val=val_arr, test=test_arr
                  None 值自动跳过
    返回：
        n_obs: 数据集样本总数
    """
    import anndata as ad

    _peek = ad.read_h5ad(h5ad_path, backed="r")
    try:
        n_obs = int(_peek.n_obs)
    finally:
        if getattr(_peek, "file", None) is not None:
            _peek.file.close()

    for name, arr in splits.items():
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            raise ValueError(f"{name}_indices is empty.")
        if int(arr.min()) < 0 or int(arr.max()) >= n_obs:
            raise ValueError(
                f"{name}_indices out of range [0, {n_obs}): "
                f"min={int(arr.min())}, max={int(arr.max())}. "
                f"Splits .npy probably comes from a different h5ad."
            )

    return n_obs


def validate_no_split_overlap(**splits: "np.ndarray | None") -> None:
    """
    校验各 split 之间无重叠索引（防御性检查）。忽略 None 值的 split。
    """
    named_arrs = [(name, np.asarray(arr)) for name, arr in splits.items() if arr is not None]
    for i in range(len(named_arrs)):
        for j in range(i + 1, len(named_arrs)):
            name_a, arr_a = named_arrs[i]
            name_b, arr_b = named_arrs[j]
            overlap = np.intersect1d(arr_a, arr_b)
            if overlap.size > 0:
                raise ValueError(
                    f"Overlap detected between {name_a} and {name_b} splits: "
                    f"{overlap.size} shared indices (e.g. {overlap[:5]})."
                )


def build_lr_scheduler(
    optimizer,
    scheduler_type: str,
    warmup_ratio: float,
    total_steps: int,
    plateau_factor: float = 0.5,
    plateau_patience: int = 2,
    plateau_min_lr: float = 1e-6,
    plateau_mode: str = "min",
    plateau_monitor: str = "val/loss",
) -> dict:
    """
    构建 Lightning lr_scheduler 配置字典。

    参数：
        optimizer: PyTorch 优化器
        scheduler_type: "cosine" 或 "plateau"
        warmup_ratio: 线性预热占总步数的比例
        total_steps: 总训练步数（用于 cosine 模式）
        plateau_*: plateau 调度器参数
        plateau_mode: "min" 或 "max"（预训练用 min，下游用 max）
        plateau_monitor: plateau 监控的指标名称
    返回：
        Lightning configure_optimizers 所需的 lr_scheduler 配置字典
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau, SequentialLR

    if scheduler_type == "cosine":
        warmup_steps = int(float(warmup_ratio) * total_steps)
        if warmup_steps <= 0:
            # warmup_ratio=0：直接走 cosine，避免 LinearLR(total_iters=0) 的边界问题
            scheduler = CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps), eta_min=1e-6
            )
        else:
            decay_steps = max(1, total_steps - warmup_steps)
            warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
            scheduler = SequentialLR(
                optimizer, [warmup, cosine], milestones=[warmup_steps]
            )
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

    if scheduler_type == "plateau":
        # Lightning 原生支持：通过 monitor 字段自动将指标传给 scheduler
        return {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode=plateau_mode,
                factor=float(plateau_factor),
                patience=int(plateau_patience),
                min_lr=float(plateau_min_lr),
            ),
            "interval": "epoch",
            "monitor": plateau_monitor,
        }

    raise ValueError(f"Unknown scheduler_type: {scheduler_type!r}")


def validate_pretrain_config(config: Any) -> None:
    """检验预训练配置的合法性（PretrainRunConfig）"""
    # 模型主体参数：基础正整性
    if config.d_model <= 0:
        raise ValueError(f"d_model must be > 0, got {config.d_model}.")
    if config.nhead <= 0:
        raise ValueError(f"nhead must be > 0, got {config.nhead}.")
    if config.num_layers <= 0:
        raise ValueError(f"num_layers must be > 0, got {config.num_layers}.")
    if config.d_model % config.nhead != 0:
        raise ValueError(
            f"d_model ({config.d_model}) must be divisible by nhead ({config.nhead})."
        )

    # ff_dim / ff_ratio 互斥检验（恰好指定其中一个）
    if (config.ff_dim is None) == (config.ff_ratio is None):
        raise ValueError(
            "Exactly one of ff_dim / ff_ratio must be specified (the other must be None). "
            f"Got ff_dim={config.ff_dim}, ff_ratio={config.ff_ratio}."
        )
    if config.ff_dim is not None and config.ff_dim <= 0:
        raise ValueError(f"ff_dim must be > 0, got {config.ff_dim}.")
    if config.ff_ratio is not None and config.ff_ratio <= 0:
        raise ValueError(f"ff_ratio must be > 0, got {config.ff_ratio}.")

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
    if config.lr <= 0:
        raise ValueError(f"lr must be > 0, got {config.lr}.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1.")
    if not 0.0 < config.mask_prob < 1.0:
        raise ValueError("mask_prob must satisfy 0 < mask_prob < 1.")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {config.weight_decay}.")

    # Early stopping 参数
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0 (0 disables early stopping).")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")

    # 共享的 budget/scheduler 检验
    validate_budget_and_lr_config(config)


def validate_finetune_config(config: Any) -> None:
    """检验微调配置的合法性（FinetuneRunConfig）"""
    # 分类头参数：基础正整性
    if config.pooling_mode not in _VALID_POOLING_MODES:
        raise ValueError(
            f"pooling_mode must be one of {sorted(_VALID_POOLING_MODES)}, "
            f"got {config.pooling_mode!r}."
        )
    if config.head_hidden_dim < 0:
        raise ValueError(f"head_hidden_dim must be >= 0, got {config.head_hidden_dim}.")
    if not 0.0 <= config.head_dropout < 1.0:
        raise ValueError("head_dropout must satisfy 0 <= head_dropout < 1.")

    # 数据协议参数
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}.")
    if config.max_seq_len < 1:
        raise ValueError(f"max_seq_len must be >= 1, got {config.max_seq_len}.")

    # 训练协议参数
    if config.lr_head <= 0:
        raise ValueError(f"lr_head must be > 0, got {config.lr_head}.")
    if config.lr_encoder <= 0:
        raise ValueError(f"lr_encoder must be > 0, got {config.lr_encoder}.")
    if config.weight_decay < 0:
        raise ValueError(f"weight_decay must be >= 0, got {config.weight_decay}.")

    # Early stopping 协议参数
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0 (0 disables early stopping).")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")

    # 共享的 budget/scheduler 检验
    validate_budget_and_lr_config(config)
