from __future__ import annotations

import argparse
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


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Downstream Classification Fine-tuning")

    # 0.输入与切分参数
    p.add_argument("--h5ad_path", "--h5ad", dest="h5ad_path", type=str, required=True)
    p.add_argument("--train_indices_path", "--train_indices", dest="train_indices_path", type=str, default=None)
    p.add_argument("--val_indices_path", "--val_indices", dest="val_indices_path", type=str, default=None)
    p.add_argument("--test_indices_path", "--test_indices", dest="test_indices_path", type=str, default=None)
    p.add_argument("--kfold_dir", type=str, default=None)

    # 1.下游任务标签参数
    p.add_argument("--label_fields", type=str, nargs="+", required=True)
    p.add_argument("--label_values", type=str, default=None)

    # 2.预训练模型参数
    p.add_argument("--pretrained_ckpt", type=str, required=True)

    # 3.1.分类模型主体参数
    p.add_argument("--pooling_mode", type=str, default="mean_pool", choices=["sample", "mean_pool", "sample_and_mean"])
    p.add_argument("--head_hidden_dim", type=int, default=0)         # 0 表示线性分类头，>0 表示 MLP hidden dim
    p.add_argument("--head_dropout", type=float, default=0.1)        # 分类头 dropout
    p.add_argument("--freeze_encoder", action="store_true", default=False)

    # 3.2.微调中的训练主体参数
    p.add_argument("--lr_head", type=float, default=1e-3)            # 分类头学习率
    p.add_argument("--lr_encoder", type=float, default=1e-5)         # Encoder 学习率（仅在不冻结时生效）
    p.add_argument("--weight_decay", type=float, default=1e-2)       # 权重衰减 (L2 正则化)
    p.add_argument("--warmup_steps", type=int, default=200)          # 学习率 warmup 步数
    p.add_argument("--max_steps", type=int, default=10000)           # 最大 optimizer steps
    p.add_argument("--max_epochs", type=int, default=100)            # 最大训练轮数

    # 4.验证与早停协议参数
    p.add_argument("--patience", type=int, default=10)               # Early stopping patience，0 表示禁用
    p.add_argument("--val_check_interval", type=int, default=None)   # 每多少个 optimizer steps 做一次验证
    p.add_argument("--gradient_clip_val", type=float, default=1.0)   # 梯度裁剪阈值

    # 5.运行与工程参数
    p.add_argument("--batch_size", type=int, default=32)             # 每个 batch 的样本数
    p.add_argument("--num_workers", type=int, default=4)            # DataLoader 的 num_workers，默认为 4
    p.add_argument("--max_seq_len", type=int, default=1024)         # 每个样本保留的最大物种数 (截断长度)
    p.add_argument("--devices", type=int, default=1)                # 使用的 GPU/设备 数量
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)                  # 随机种子，用于可复现
    p.add_argument("--log_dir", type=str, default="tmp/logs/finetune")
    p.add_argument("--no_progress_bar", action="store_true", default=False)  # 关闭进度条（远程服务器/nohup 运行时避免刷屏）

    return p


def validate_args(args: argparse.Namespace) -> None:
    # 检验标签任务配置
    if not args.label_fields:
        raise ValueError("--label_fields must contain at least one task field.")

    # 检验训练主体超参数
    if args.head_hidden_dim < 0:
        raise ValueError("--head_hidden_dim must be >= 0.")
    if not 0.0 <= args.head_dropout < 1.0:
        raise ValueError("--head_dropout must satisfy 0 <= head_dropout < 1.")
    if args.lr_head <= 0:
        raise ValueError("--lr_head must be > 0.")
    if args.lr_encoder <= 0:
        raise ValueError("--lr_encoder must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay must be >= 0.")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be >= 0.")
    if args.max_steps <= 0:
        raise ValueError("--max_steps must be > 0.")
    if args.max_epochs <= 0:
        raise ValueError("--max_epochs must be > 0.")

    # 检验验证与早停协议
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")
    if args.val_check_interval is not None and args.val_check_interval <= 0:
        raise ValueError("--val_check_interval must be > 0 when provided.")
    if args.gradient_clip_val < 0:
        raise ValueError("--gradient_clip_val must be >= 0.")

    # 检验运行参数
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0.")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be >= 0.")
    if args.max_seq_len <= 0:
        raise ValueError("--max_seq_len must be > 0.")
    if args.devices <= 0:
        raise ValueError("--devices must be > 0.")

    # 为了避免切分来源混乱，单次模式与 k-fold 模式互斥：
    # - k-fold 模式只允许提供 kfold_dir
    # - 单次模式必须显式提供 train / val 索引
    using_kfold = args.kfold_dir is not None
    using_single = args.train_indices_path is not None or args.val_indices_path is not None

    if using_kfold:
        if using_single or args.test_indices_path is not None:
            raise ValueError(
                "--kfold_dir is incompatible with --train_indices_path/--val_indices_path/--test_indices_path."
            )
    else:
        if args.train_indices_path is None or args.val_indices_path is None:
            raise ValueError(
                "Single-run mode requires both --train_indices_path and --val_indices_path."
            )


def _parse_label_values(label_values_str: str | None) -> dict[str, list[str]] | None:
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


def _build_label_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    # 解析标签值过滤配置；未指定时表示从数据中自动抽取有效标签集合
    label_values_map = _parse_label_values(args.label_values)

    label_configs: list[dict[str, object]] = []
    for field in args.label_fields:
        config: dict[str, object] = {"field": field}
        if label_values_map is not None and field in label_values_map:
            config["values"] = label_values_map[field]
        label_configs.append(config)
    return label_configs


def _choose_precision(precision: str) -> str:
    # 自动根据设备选择精度，避免 CPU 下使用 16-mixed 报错
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


def run_single(
    args: argparse.Namespace,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None,
    log_subdir: str,
) -> dict[str, dict[str, float | int]]:
    L.seed_everything(args.seed, workers=True)

    chosen_precision = _choose_precision(args.precision)
    rank_zero_info(f"{TAG} Using precision={chosen_precision}")
    rank_zero_info(f"{TAG} Train: {len(train_indices)}, Val: {len(val_indices)}")
    if test_indices is not None:
        rank_zero_info(f"{TAG} Test: {len(test_indices)}")

    # 1. 构建标签配置
    label_configs = _build_label_configs(args)
    rank_zero_info(f"{TAG} Label fields: {args.label_fields}")

    # 2. 初始化数据模块
    rank_zero_info(f"{TAG} Initializing DataModule...")
    dm = ClassificationDataModule(
        h5ad_path=args.h5ad_path,
        label_configs=label_configs,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
    )

    # 3. 构建任务配置
    task_configs = [
        {"name": cfg["field"], "num_classes": cfg["num_classes"]}
        for cfg in dm.task_configs
    ]

    # 4. 初始化分类模型
    rank_zero_info(
        f"{TAG} Initializing classifier with pooling={args.pooling_mode}, "
        f"freeze_encoder={args.freeze_encoder}"
    )
    model = MiCoFormerClassifier(
        pretrained_ckpt_path=args.pretrained_ckpt,
        task_configs=task_configs,
        pooling_mode=args.pooling_mode,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        freeze_encoder=args.freeze_encoder,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    # 5. 设置日志记录器与回调（CSV 用于离线查看，TensorBoard 用于实时监控）
    csv_logger = CSVLogger(save_dir=args.log_dir, name=log_subdir)
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=log_subdir)

    # 默认用第一个任务的 f1_macro 作为 checkpoint / early stopping 监控指标
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
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(monitor=monitor_metric, mode="max", patience=args.patience)
        )

    # 6. 初始化 Lightning Trainer
    rank_zero_info(f"{TAG} Training budget: max_epochs={args.max_epochs}, max_steps={args.max_steps}")
    if args.val_check_interval is not None:
        rank_zero_info(f"{TAG} Validation interval (steps): {args.val_check_interval}")

    trainer_kwargs = dict(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        devices=args.devices,
        precision=chosen_precision,
        gradient_clip_val=args.gradient_clip_val,
        logger=[csv_logger, tb_logger],
        callbacks=callbacks,
        default_root_dir=args.log_dir,
    )
    if args.no_progress_bar:
        trainer_kwargs["enable_progress_bar"] = False
    if args.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = args.val_check_interval

    trainer = L.Trainer(**trainer_kwargs)

    # 7. 开始训练
    rank_zero_info(f"{TAG} Starting fine-tuning (log_subdir={log_subdir}) ...")
    trainer.fit(model, datamodule=dm)

    # 8. 可选测试与结果汇总
    results: dict[str, dict[str, float | int]] = {}
    if test_indices is not None:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        if test_results:
            results["test"] = _extract_scalar_metrics(test_results[0])

    if trainer.callback_metrics:
        results["val"] = _extract_scalar_metrics(trainer.callback_metrics)

    return results


def run_kfold(args: argparse.Namespace) -> None:
    # 自动检测 kfold_dir 中的 fold 文件并循环训练
    pattern = os.path.join(args.kfold_dir, "fold_*_train.npy")
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
        train_path = os.path.join(args.kfold_dir, f"fold_{fold_i}_train.npy")
        val_path = os.path.join(args.kfold_dir, f"fold_{fold_i}_val.npy")

        rank_zero_info(f"{TAG} {'=' * 50}")
        rank_zero_info(f"{TAG} Fold {fold_i}")
        rank_zero_info(f"{TAG} {'=' * 50}")

        train_indices = _load_indices(train_path, f"fold_{fold_i} train")
        val_indices = _load_indices(val_path, f"fold_{fold_i} val")

        results = run_single(
            args,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=None,
            log_subdir=f"fold_{fold_i}",
        )
        all_results.append(results)

    # 汇总各 fold 的验证指标
    rank_zero_info(f"{TAG} {'=' * 50}")
    rank_zero_info(f"{TAG} K-fold Summary")
    rank_zero_info(f"{TAG} {'=' * 50}")

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
            mean = np.mean(values)
            std = np.std(values)
            rank_zero_info(f"{TAG}   {key}: {mean:.4f} +/- {std:.4f}")


def _print_results(results: dict[str, dict[str, float | int]]) -> None:
    rank_zero_info(f"{TAG} Results:")
    for split, metrics in results.items():
        rank_zero_info(f"{TAG}   [{split}]")
        for key, value in sorted(metrics.items()):
            rank_zero_info(f"{TAG}     {key}: {value}")


def main() -> None:
    args = build_argparser().parse_args()
    validate_args(args)

    if args.kfold_dir is not None:
        # k-fold 模式：自动遍历 fold_x_train.npy / fold_x_val.npy
        run_kfold(args)
        return

    # 单次模式：显式加载 train / val / test 切分
    train_indices = _load_indices(args.train_indices_path, "train")
    val_indices = _load_indices(args.val_indices_path, "val")
    test_indices = None
    if args.test_indices_path is not None:
        test_indices = _load_indices(args.test_indices_path, "test")

    results = run_single(
        args,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        log_subdir="finetune",
    )
    _print_results(results)


if __name__ == "__main__":
    main()
