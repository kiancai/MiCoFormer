from __future__ import annotations

import argparse

from lightning.pytorch.utilities import rank_zero_info

from micoformer.runners.finetune import (
    FinetuneRunConfig,
    load_indices,
    run_finetune_once,
)


TAG = "[train_finetune]"


def add_finetune_common_args(p: argparse.ArgumentParser, *, default_log_dir: str) -> argparse.ArgumentParser:
    # 0.输入参数
    p.add_argument("--h5ad_path", "--h5ad", dest="h5ad_path", type=str, required=True)

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
    p.add_argument("--num_workers", type=int, default=4)             # DataLoader 的 num_workers，默认为 4
    p.add_argument("--max_seq_len", type=int, default=1024)          # 每个样本保留的最大物种数 (截断长度)
    p.add_argument("--devices", type=int, default=1)                 # 使用的 GPU/设备 数量
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)                   # 随机种子，用于可复现
    p.add_argument("--log_dir", type=str, default=default_log_dir)
    p.add_argument("--no_progress_bar", action="store_true", default=False)  # 关闭进度条（远程服务器/nohup 运行时避免刷屏）
    return p


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Downstream Classification Fine-tuning")
    p.add_argument("--train_indices_path", "--train_indices", dest="train_indices_path", type=str, required=True)
    p.add_argument("--val_indices_path", "--val_indices", dest="val_indices_path", type=str, required=True)
    return add_finetune_common_args(p, default_log_dir="outputs/finetune")


def namespace_to_config(args: argparse.Namespace) -> FinetuneRunConfig:
    return FinetuneRunConfig(
        h5ad_path=args.h5ad_path,
        label_fields=args.label_fields,
        label_values=args.label_values,
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


def print_results(results: dict[str, dict[str, float | int]]) -> None:
    rank_zero_info(f"{TAG} Results:")
    for split, metrics in results.items():
        rank_zero_info(f"{TAG}   [{split}]")
        for key, value in sorted(metrics.items()):
            rank_zero_info(f"{TAG}     {key}: {value}")


def main() -> None:
    args = build_argparser().parse_args()
    config = namespace_to_config(args)
    train_indices = load_indices(args.train_indices_path, "train")
    val_indices = load_indices(args.val_indices_path, "val")
    results = run_finetune_once(config, train_indices, val_indices, log_subdir="single_run")
    print_results(results)


if __name__ == "__main__":
    main()

