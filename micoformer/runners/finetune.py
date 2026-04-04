from __future__ import annotations

from dataclasses import dataclass
import glob
import json
import os
import re

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_info

from micoformer.datamodules.classification_datamodule import ClassificationDataModule
from micoformer.models.classification_module import MiCoFormerClassifier


TAG = "[train_finetune]"


@dataclass(slots=True)
class FinetuneRunConfig:
    h5ad_path: str
    label_fields: list[str]
    label_values: str | None
    pretrained_ckpt: str
    pooling_mode: str
    head_hidden_dim: int
    head_dropout: float
    freeze_encoder: bool
    lr_head: float
    lr_encoder: float
    weight_decay: float
    warmup_steps: int
    max_steps: int
    max_epochs: int
    patience: int
    val_check_interval: int | None
    gradient_clip_val: float
    batch_size: int
    num_workers: int
    max_seq_len: int
    devices: int
    precision: str
    seed: int
    log_dir: str
    no_progress_bar: bool


def validate_finetune_config(config: FinetuneRunConfig) -> None:
    # 检验标签任务配置
    if not config.label_fields:
        raise ValueError("--label_fields must contain at least one task field.")

    # 检验训练主体超参数
    if config.head_hidden_dim < 0:
        raise ValueError("--head_hidden_dim must be >= 0.")
    if not 0.0 <= config.head_dropout < 1.0:
        raise ValueError("--head_dropout must satisfy 0 <= head_dropout < 1.")
    if config.lr_head <= 0:
        raise ValueError("--lr_head must be > 0.")
    if config.lr_encoder <= 0:
        raise ValueError("--lr_encoder must be > 0.")
    if config.weight_decay < 0:
        raise ValueError("--weight_decay must be >= 0.")
    if config.warmup_steps < 0:
        raise ValueError("--warmup_steps must be >= 0.")
    if config.max_steps <= 0:
        raise ValueError("--max_steps must be > 0.")
    if config.max_epochs <= 0:
        raise ValueError("--max_epochs must be > 0.")

    # 检验验证与早停协议
    if config.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if config.val_check_interval is not None and config.val_check_interval <= 0:
        raise ValueError("--val_check_interval must be > 0 when provided.")
    if config.gradient_clip_val < 0:
        raise ValueError("--gradient_clip_val must be >= 0.")

    # 检验运行参数
    if config.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if config.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")
    if config.max_seq_len <= 0:
        raise ValueError("--max_seq_len must be > 0.")
    if config.devices <= 0:
        raise ValueError("--devices must be > 0.")


def choose_precision(precision: str) -> str:
    # 自动根据设备选择精度，避免 CPU 下使用 16-mixed 报错
    if precision == "auto":
        return "16-mixed" if torch.cuda.is_available() else "32"
    return precision


def load_indices(indices_path: str, split_name: str) -> np.ndarray:
    rank_zero_info(f"{TAG} Loading {split_name} indices from {indices_path} ...")
    return np.load(indices_path)


def parse_label_values(label_values_str: str | None) -> dict[str, list[str]] | None:
    if label_values_str is None:
        return None

    try:
        parsed = json.loads(label_values_str)
    except json.JSONDecodeError as exc:
        raise ValueError("--label_values must be a valid JSON object.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("--label_values must be a JSON object mapping field -> list[str].")

    normalized: dict[str, list[str]] = {}
    for field, values in parsed.items():
        if not isinstance(field, str):
            raise ValueError("--label_values keys must be strings.")
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError("--label_values values must be lists of strings.")
        normalized[field] = values
    return normalized


def build_label_configs(config: FinetuneRunConfig) -> list[dict[str, object]]:
    # 解析标签值过滤配置；未指定时表示从数据中自动抽取有效标签集合
    label_values_map = parse_label_values(config.label_values)
    label_configs: list[dict[str, object]] = []
    for field in config.label_fields:
        label_config: dict[str, object] = {"field": field}
        if label_values_map is not None and field in label_values_map:
            label_config["values"] = label_values_map[field]
        label_configs.append(label_config)
    return label_configs


def extract_scalar_metrics(metrics: dict) -> dict[str, float | int]:
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
    *,
    test_indices: np.ndarray | None = None,
    log_subdir: str = "finetune",
) -> dict[str, dict[str, float | int]]:
    validate_finetune_config(config)
    L.seed_everything(config.seed, workers=True)

    chosen_precision = choose_precision(config.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)}")
    if test_indices is not None:
        rank_zero_info(f"{TAG} Test: {len(test_indices)}")

    # 1. 构建标签配置
    label_configs = build_label_configs(config)
    rank_zero_info(f"{TAG} Label fields: {config.label_fields}")

    # 2. 初始化数据模块
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

    # 3. 构建任务配置
    task_configs = [
        {"name": task_cfg["field"], "num_classes": task_cfg["num_classes"]}
        for task_cfg in dm.task_configs
    ]

    # 4. 初始化分类模型
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

    # 5. 设置日志记录器与回调
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

    # 6. 初始化 Trainer
    rank_zero_info(
        f"{TAG} Training budget: max_epochs={config.max_epochs}, "
        f"max_steps={config.max_steps}"
    )
    if config.val_check_interval is not None:
        rank_zero_info(
            f"{TAG} Validation interval (steps): {config.val_check_interval}"
        )

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

    # 7. 开始训练
    rank_zero_info(f"{TAG} Starting fine-tuning (log_subdir={log_subdir}) ...")
    trainer.fit(model, datamodule=dm)

    # 8. 可选测试与结果汇总
    results: dict[str, dict[str, float | int]] = {}
    if test_indices is not None:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        if test_results:
            results["test"] = extract_scalar_metrics(test_results[0])

    if trainer.callback_metrics:
        results["val"] = extract_scalar_metrics(trainer.callback_metrics)

    return results


def collect_kfold_indices(kfold_dir: str) -> list[int]:
    # 自动检测 kfold_dir 中的 fold 文件并返回可用折号
    pattern = os.path.join(kfold_dir, "fold_*_train.npy")
    train_files = sorted(glob.glob(pattern))
    if not train_files:
        raise FileNotFoundError(f"No fold files found matching {pattern}")

    fold_indices = []
    for train_file in train_files:
        match = re.search(r"fold_(\d+)_train\.npy", os.path.basename(train_file))
        if match:
            fold_indices.append(int(match.group(1)))
    return sorted(fold_indices)


def summarize_kfold_results(
    all_results: list[dict[str, dict[str, float | int]]],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    val_keys = set()
    for result in all_results:
        if "val" in result:
            val_keys.update(result["val"].keys())

    for key in sorted(val_keys):
        values = []
        for result in all_results:
            if "val" in result and key in result["val"]:
                values.append(result["val"][key])
        if values:
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
    return summary

