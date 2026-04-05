"""微调 workflow：可复用的训练逻辑（不含 argparse）"""

from __future__ import annotations

import glob
import json
import os
import re
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


@dataclass
class FinetuneRunConfig:
    # 数据路径
    h5ad_path: str
    pretrained_ckpt: str

    # 分类模型参数
    pooling_mode: str = "mean_pool"
    head_hidden_dim: int = 0
    head_dropout: float = 0.1
    freeze_encoder: bool = False

    # 训练参数
    lr_head: float = 1e-3
    lr_encoder: float = 1e-5
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    max_steps: int = 10000
    max_epochs: int = 100

    # 验证与早停
    patience: int = 10
    val_check_interval: int | None = None
    gradient_clip_val: float = 1.0

    # 运行参数
    batch_size: int = 32
    num_workers: int = 4
    max_seq_len: int = 1024
    devices: int = 1
    precision: str = "auto"
    seed: int = 42
    log_dir: str = "tmp/logs/finetune"
    no_progress_bar: bool = False


def validate_finetune_config(config: FinetuneRunConfig) -> None:
    """检验微调配置的合法性"""
    if config.head_hidden_dim < 0:
        raise ValueError("head_hidden_dim must be >= 0.")
    if not 0.0 <= config.head_dropout < 1.0:
        raise ValueError("head_dropout must satisfy 0 <= head_dropout < 1.")
    if config.lr_head <= 0:
        raise ValueError("lr_head must be > 0.")
    if config.lr_encoder <= 0:
        raise ValueError("lr_encoder must be > 0.")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be >= 0.")
    if config.warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0.")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be > 0.")
    if config.max_epochs <= 0:
        raise ValueError("max_epochs must be > 0.")
    if config.patience < 0:
        raise ValueError("patience must be >= 0.")
    if config.val_check_interval is not None and config.val_check_interval <= 0:
        raise ValueError("val_check_interval must be > 0 when provided.")
    if config.gradient_clip_val < 0:
        raise ValueError("gradient_clip_val must be >= 0.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be > 0.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be >= 0.")
    if config.max_seq_len <= 0:
        raise ValueError("max_seq_len must be > 0.")
    if config.devices <= 0:
        raise ValueError("devices must be > 0.")


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
    if precision == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32"
    return precision


def _load_indices(indices_path: str, split_name: str) -> np.ndarray:
    rank_zero_info(f"{TAG} Loading {split_name} indices from {indices_path} ...")
    return np.load(indices_path)


def _extract_scalar_metrics(metrics: dict) -> dict[str, float | int]:
    extracted: dict[str, float | int] = {}
    for key, value in metrics.items():
        scalar_value = value.item() if hasattr(value, "item") else value
        if isinstance(scalar_value, (int, float)):
            extracted[key] = scalar_value
    return extracted


def run_finetune_once(
    config: FinetuneRunConfig,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None,
    label_configs: list[dict[str, object]],
    *,
    log_subdir: str = "finetune",
) -> dict[str, dict[str, float | int]]:
    """执行一次完整的微调流程，返回结果字典"""
    validate_finetune_config(config)
    L.seed_everything(config.seed, workers=True)

    chosen_precision = _choose_precision(config.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)}")
    if test_indices is not None:
        rank_zero_info(f"{TAG} Test: {len(test_indices)}")

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

    # 3. 初始化分类模型
    rank_zero_info(
        f"{TAG} Initializing classifier with pooling={config.pooling_mode}, "
        f"freeze_encoder={config.freeze_encoder}"
    )
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
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps,
    )

    # 4. 设置日志记录器与回调
    csv_logger = CSVLogger(save_dir=config.log_dir, name=log_subdir)
    tb_logger = TensorBoardLogger(save_dir=config.log_dir, name=log_subdir)

    primary_task = task_configs[0]["name"]
    monitor_metric = f"val/{primary_task}/f1_macro"
    rank_zero_info(f"{TAG} Monitor metric: {monitor_metric}")

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]
    if config.patience > 0:
        callbacks.append(
            EarlyStopping(monitor=monitor_metric, mode="max", patience=config.patience)
        )

    # 5. 初始化 Trainer
    rank_zero_info(
        f"{TAG} Training budget: max_epochs={config.max_epochs}, "
        f"max_steps={config.max_steps}"
    )
    if config.val_check_interval is not None:
        rank_zero_info(f"{TAG} Validation interval (steps): {config.val_check_interval}")

    trainer_kwargs = dict(
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        devices=config.devices,
        precision=chosen_precision,
        gradient_clip_val=config.gradient_clip_val,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        default_root_dir=config.log_dir,
    )
    if config.no_progress_bar:
        trainer_kwargs["enable_progress_bar"] = False
    if config.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = config.val_check_interval

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

    return results


def run_kfold(
    config: FinetuneRunConfig,
    kfold_dir: str,
    label_configs: list[dict[str, object]],
) -> list[dict[str, dict[str, float | int]]]:
    """自动检测 kfold_dir 中的 fold 文件并循环训练，返回各 fold 结果"""
    pattern = os.path.join(kfold_dir, "fold_*_train.npy")
    train_files = sorted(glob.glob(pattern))
    if not train_files:
        raise FileNotFoundError(f"No fold files found matching {pattern}")

    fold_indices = []
    for train_file in train_files:
        match = re.search(r"fold_(\d+)_train\.npy", os.path.basename(train_file))
        if match:
            fold_indices.append(int(match.group(1)))
    fold_indices.sort()

    rank_zero_info(f"{TAG} Detected {len(fold_indices)} folds: {fold_indices}")

    all_results = []
    for fold_i in fold_indices:
        train_path = os.path.join(kfold_dir, f"fold_{fold_i}_train.npy")
        val_path = os.path.join(kfold_dir, f"fold_{fold_i}_val.npy")

        rank_zero_info(f"{TAG} {'=' * 50}")
        rank_zero_info(f"{TAG} Fold {fold_i}")
        rank_zero_info(f"{TAG} {'=' * 50}")

        train_indices = _load_indices(train_path, f"fold_{fold_i} train")
        val_indices = _load_indices(val_path, f"fold_{fold_i} val")

        results = run_finetune_once(
            config,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=None,
            label_configs=label_configs,
            log_subdir=f"fold_{fold_i}",
        )
        all_results.append(results)

    # 汇总各 fold 的验证指标
    rank_zero_info(f"{TAG} {'=' * 50}")
    rank_zero_info(f"{TAG} K-fold Summary")
    rank_zero_info(f"{TAG} {'=' * 50}")

    val_keys: set[str] = set()
    for result in all_results:
        if "val" in result:
            val_keys.update(result["val"].keys())

    for key in sorted(val_keys):
        values = []
        for result in all_results:
            if "val" in result and key in result["val"]:
                values.append(result["val"][key])
        if values:
            mean = np.mean(values)
            std = np.std(values)
            rank_zero_info(f"{TAG}   {key}: {mean:.4f} +/- {std:.4f}")

    return all_results
