"""
scripts/3.train_finetune.py — MiCoFormer 下游分类微调入口脚本

支持两种模式：
  - 单次模式：显式提供 --train_indices_path 和 --val_indices_path
  - K-fold 模式：提供 --kfold_dir，自动检测 fold 文件

使用示例：

  # 单次微调
  python scripts/3.train_finetune.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --pretrained_ckpt outputs/pretrain/checkpoints/best.ckpt \
      --train_indices_path data/processed/splits/finetune/fold_0_train.npy \
      --val_indices_path data/processed/splits/finetune/fold_0_val.npy \
      --label_fields Phenotype

  # K-fold 微调
  python scripts/3.train_finetune.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --pretrained_ckpt outputs/pretrain/checkpoints/best.ckpt \
      --kfold_dir data/processed/splits/finetune/ \
      --label_fields Phenotype
"""

from __future__ import annotations

import argparse

import numpy as np
from lightning.pytorch.utilities import rank_zero_info

from micoformer.workflows.finetune import (
    FinetuneRunConfig,
    build_label_configs,
    run_finetune_once,
    run_kfold,
)


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
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="tmp/logs/finetune")
    p.add_argument("--no_progress_bar", action="store_true", default=False)

    return p


def _validate_args(args: argparse.Namespace) -> None:
    """检验切分来源互斥：kfold 模式与单次模式不能混用"""
    if not args.label_fields:
        raise ValueError("--label_fields must contain at least one task field.")

    using_kfold = args.kfold_dir is not None
    using_single = args.train_indices_path is not None or args.val_indices_path is not None

    if using_kfold:
        if using_single or args.test_indices_path is not None:
            raise ValueError(
                "--kfold_dir is incompatible with "
                "--train_indices_path/--val_indices_path/--test_indices_path."
            )
    else:
        if args.train_indices_path is None or args.val_indices_path is None:
            raise ValueError(
                "Single-run mode requires both --train_indices_path and --val_indices_path."
            )


def _args_to_config(args: argparse.Namespace) -> FinetuneRunConfig:
    return FinetuneRunConfig(
        h5ad_path=args.h5ad_path,
        pretrained_ckpt=args.pretrained_ckpt,
        pooling_mode=args.pooling_mode,
        head_hidden_dim=args.head_hidden_dim,
        head_dropout=args.head_dropout,
        freeze_encoder=args.freeze_encoder,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_check_interval=args.val_check_interval,
        gradient_clip_val=args.gradient_clip_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        log_dir=args.log_dir,
        no_progress_bar=args.no_progress_bar,
    )


def _load_indices(path: str, name: str) -> np.ndarray:
    rank_zero_info(f"{TAG} Loading {name} indices from {path} ...")
    return np.load(path)


def _print_results(results: dict[str, dict[str, float | int]]) -> None:
    rank_zero_info(f"{TAG} Results:")
    for split, metrics in results.items():
        rank_zero_info(f"{TAG}   [{split}]")
        for key, value in sorted(metrics.items()):
            rank_zero_info(f"{TAG}     {key}: {value}")


def main() -> None:
    args = build_argparser().parse_args()
    _validate_args(args)

    config = _args_to_config(args)
    label_configs = build_label_configs(args.label_fields, args.label_values)

    if args.kfold_dir is not None:
        run_kfold(config, args.kfold_dir, label_configs)
        return

    train_indices = _load_indices(args.train_indices_path, "train")
    val_indices = _load_indices(args.val_indices_path, "val")
    test_indices = None
    if args.test_indices_path is not None:
        test_indices = _load_indices(args.test_indices_path, "test")

    results = run_finetune_once(
        config,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        label_configs=label_configs,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
