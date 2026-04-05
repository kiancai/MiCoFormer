"""
scripts/4.run_finetune_protocol.py — MiCoFormer 微调协议编排脚本

在 scripts/3.train_finetune.py 的基础上提供更高层的协议编排：
  - kfold: 自动检测 fold 文件，循环训练并汇总
  - holdout: 显式 train/val 切分，可选 test
  - ood: 显式 train/val/test 切分，test 评估由协议统一负责

使用示例：

  # K-fold 协议
  python scripts/4.run_finetune_protocol.py \
      --protocol kfold \
      --kfold_dir data/processed/splits/finetune/kfold/ \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --pretrained_ckpt outputs/pretrain/checkpoints/best.ckpt \
      --label_fields Phenotype

  # OOD 协议
  python scripts/4.run_finetune_protocol.py \
      --protocol ood \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --pretrained_ckpt outputs/pretrain/checkpoints/best.ckpt \
      --train_indices_path splits/train.npy \
      --val_indices_path splits/val.npy \
      --test_indices_path splits/test.npy \
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


TAG = "[run_finetune_protocol]"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Fine-tuning Protocol Runner")

    p.add_argument("--protocol", type=str, required=True, choices=["kfold", "holdout", "ood"])

    # 数据与模型
    p.add_argument("--h5ad_path", "--h5ad", dest="h5ad_path", type=str, required=True)
    p.add_argument("--pretrained_ckpt", type=str, required=True)
    p.add_argument("--label_fields", type=str, nargs="+", required=True)
    p.add_argument("--label_values", type=str, default=None)

    # K-fold 专用
    p.add_argument("--kfold_dir", type=str, default=None)

    # Holdout / OOD 专用
    p.add_argument("--train_indices_path", type=str, default=None)
    p.add_argument("--val_indices_path", type=str, default=None)
    p.add_argument("--test_indices_path", type=str, default=None)

    # 分类模型参数
    p.add_argument("--pooling_mode", type=str, default="mean_pool", choices=["sample", "mean_pool", "sample_and_mean"])
    p.add_argument("--head_hidden_dim", type=int, default=0)
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--freeze_encoder", action="store_true", default=False)

    # 训练参数
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val_check_interval", type=int, default=None)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)

    # 运行参数
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="tmp/logs/finetune")
    p.add_argument("--no_progress_bar", action="store_true", default=False)

    return p


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


def main() -> None:
    args = build_argparser().parse_args()
    config = _args_to_config(args)
    label_configs = build_label_configs(args.label_fields, args.label_values)

    if args.protocol == "kfold":
        if args.kfold_dir is None:
            raise ValueError("--kfold_dir is required for kfold protocol.")
        run_kfold(config, args.kfold_dir, label_configs)

    elif args.protocol in ("holdout", "ood"):
        if args.train_indices_path is None or args.val_indices_path is None:
            raise ValueError(
                f"--train_indices_path and --val_indices_path are required "
                f"for {args.protocol} protocol."
            )
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

        rank_zero_info(f"{TAG} Results:")
        for split, metrics in results.items():
            rank_zero_info(f"{TAG}   [{split}]")
            for key, value in sorted(metrics.items()):
                rank_zero_info(f"{TAG}     {key}: {value}")


if __name__ == "__main__":
    main()
