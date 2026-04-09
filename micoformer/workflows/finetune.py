"""微调 workflow：可复用的训练逻辑（不含 argparse）"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info

from micoformer.datamodules.classification_datamodule import ClassificationDataModule
from micoformer.models.classification_module import MiCoFormerClassifier


TAG = "[train_finetune]"


_VALID_POOLING_MODES = {"sample", "mean_pool", "sample_and_mean"}


@dataclass
class FinetuneRunConfig:
    # 0. 输入与切分参数
    h5ad_path: str
    pretrained_ckpt: str

    # 1. 分类头 / pooling 参数
    pooling_mode: str = "mean_pool"
    head_hidden_dim: int = 0
    head_dropout: float = 0.1
    freeze_encoder: bool = False

    # 2.1. 数据协议参数
    batch_size: int = 32
    max_seq_len: int = 1024

    # 3.1. 微调训练主体参数
    lr_head: float = 1e-3
    lr_encoder: float = 1e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1

    # 3.2. 微调协议参数
    # lr_scheduler_type 决定学习率下降方式：
    # - cosine：warmup 后按 cosine 平滑衰减
    # - plateau：warmup 后根据监控指标（默认 val/{task}/f1_macro）是否停滞来自动降 LR
    lr_scheduler_type: str = "cosine"
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    lr_plateau_min_lr: float = 1e-6

    # 4. 预算与验证协议参数
    # budget_mode 决定"训练预算"的单位：
    # - epoch：更适合当前这种数据规模不算特别大的实验
    # - step：更适合超大数据集或只想固定 optimizer 更新次数的场景
    budget_mode: str = "epoch"
    max_epochs: int | None = None
    max_steps: int | None = None
    val_interval_epochs: int | None = None
    val_interval_steps: int | None = None
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0

    # 5. 运行与工程参数
    devices: int = 1
    precision: str = "auto"
    seed: int = 42
    num_workers: int = 4
    log_dir: str = "tmp/logs/finetune"
    no_progress_bar: bool = False


def validate_finetune_config(config: FinetuneRunConfig) -> None:
    """检验微调配置的合法性"""
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

    # Early stopping 协议参数
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0 (0 disables early stopping).")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")


def parse_label_values(label_values_str: str | None) -> dict[str, list[str]] | None:
    """解析 --label_values JSON 字符串为 {field: [values]} 映射"""
    if label_values_str is None:
        return None

    try:
        parsed = json.loads(label_values_str)
    except json.JSONDecodeError as exc:
        raise ValueError("label_values must be a valid JSON object.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("label_values must be a JSON object mapping field -> list[str].")

    normalized: dict[str, list[str]] = {}
    for field, values in parsed.items():
        if not isinstance(field, str):
            raise ValueError("label_values keys must be strings.")
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("label_values values must be lists of strings.")
        normalized[field] = values
    return normalized


def build_label_configs(
    label_fields: list[str],
    label_values_str: str | None = None,
) -> list[dict[str, object]]:
    """从标签字段列表和可选的 values 过滤构建 label_configs"""
    label_values_map = parse_label_values(label_values_str)

    label_configs: list[dict[str, object]] = []
    for field in label_fields:
        config: dict[str, object] = {"field": field}
        if label_values_map is not None and field in label_values_map:
            config["values"] = label_values_map[field]
        label_configs.append(config)
    return label_configs


def _choose_precision(precision: str) -> str:
    # 自动根据设备选择精度
    if precision == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32"
    return precision


def _extract_scalar_metrics(metrics: dict) -> dict[str, float | int]:
    extracted: dict[str, float | int] = {}
    for key, value in metrics.items():
        scalar_value = value.item() if hasattr(value, "item") else value
        if isinstance(scalar_value, (int, float)):
            extracted[key] = scalar_value
    return extracted


# 执行一次完整的微调流程，返回结果字典
def run_finetune_once(
    config: FinetuneRunConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None,
    label_configs: list[dict[str, object]],
    *,
    log_subdir: str = "finetune_stage0",
) -> dict[str, dict[str, float | int]]:
    validate_finetune_config(config)
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
    if test_indices is not None:
        test_arr = np.asarray(test_indices)
        if test_arr.size == 0:
            raise ValueError("test_indices is empty.")
        if int(test_arr.min()) < 0 or int(test_arr.max()) >= n_obs:
            raise ValueError(
                f"test_indices out of range [0, {n_obs}): "
                f"min={int(test_arr.min())}, max={int(test_arr.max())}. "
                f"Splits .npy probably comes from a different h5ad."
            )

    chosen_precision = _choose_precision(config.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(
        f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)}"
        + (f", Test: {len(test_indices)}" if test_indices is not None else "")
        + f" (n_obs={n_obs})"
    )
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
    dm = ClassificationDataModule(
        h5ad_path=config.h5ad_path,
        label_configs=label_configs,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_seq_len=config.max_seq_len,
    )

    # 2. 构建任务配置
    task_configs = [
        {"name": cfg["field"], "num_classes": cfg["num_classes"]}
        for cfg in dm.task_configs
    ]
    primary_task = task_configs[0]["name"]
    monitor_metric = f"val/{primary_task}/f1_macro"

    # 3. 初始化分类模型
    rank_zero_info(
        f"{TAG} Initializing classifier with pooling={config.pooling_mode}, "
        f"freeze_encoder={config.freeze_encoder}"
    )
    rank_zero_info(f"{TAG} Monitor metric: {monitor_metric}")
    model = MiCoFormerClassifier(
        pretrained_ckpt_path=config.pretrained_ckpt,
        task_configs=task_configs,
        pooling_mode=config.pooling_mode,
        head_hidden_dim=config.head_hidden_dim,
        head_dropout=config.head_dropout,
        freeze_encoder=config.freeze_encoder,
        lr_head=config.lr_head,
        lr_encoder=config.lr_encoder,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler=config.lr_scheduler_type,
        plateau_factor=config.lr_plateau_factor,
        plateau_patience=config.lr_plateau_patience,
        plateau_min_lr=config.lr_plateau_min_lr,
        monitor_metric=monitor_metric,
    )

    # 4. 设置日志记录器与回调
    # 显式锁定 version：让 CSVLogger 与 TensorBoardLogger 共享同一个 version 字符串，
    # 避免两者各自自增、出现 version_3 / version_5 错位
    run_version = time.strftime("run_%Y%m%d_%H%M%S")
    csv_logger = CSVLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=log_subdir, version=run_version)

    # 显式指定 dirpath，避免依赖 "第一个 logger 的 save_dir" 这种隐式行为
    ckpt_dir = os.path.join(config.log_dir, log_subdir, run_version, "checkpoints")
    # 注意：filename 中不能包含 monitor 的 'val/{task}/f1_macro'，否则斜杠会被当作子目录创建
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor=monitor_metric,
        mode="max",
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
                monitor=monitor_metric,
                mode="max",
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        )

    # 5. 初始化 Trainer
    trainer_kwargs = dict(
        devices=config.devices,
        precision=chosen_precision,
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

    # 6. 开始训练
    rank_zero_info(f"{TAG} Starting fine-tuning (log_subdir={log_subdir}) ...")
    trainer.fit(model, datamodule=dm)

    # 7. 收集结果
    results: dict[str, dict[str, float | int]] = {}
    if test_indices is not None:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        if test_results:
            results["test"] = _extract_scalar_metrics(test_results[0])

    if trainer.callback_metrics:
        results["val"] = _extract_scalar_metrics(trainer.callback_metrics)

    rank_zero_info(f"{TAG} Best checkpoint: {checkpoint_callback.best_model_path}")
    if checkpoint_callback.best_model_score is not None:
        rank_zero_info(
            f"{TAG} Best {monitor_metric}: {checkpoint_callback.best_model_score.item():.6f}"
        )
    return results
