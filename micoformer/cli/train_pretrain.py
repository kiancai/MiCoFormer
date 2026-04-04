from __future__ import annotations

import argparse

from micoformer.runners.pretrain import (
    PretrainRunConfig,
    load_indices,
    run_pretrain_once,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Stage 0 Pretraining")

    # 0.输入与切分参数
    p.add_argument("--h5ad_path", type=str, required=True)
    p.add_argument("--train_indices_path", type=str, required=True)
    p.add_argument("--val_indices_path", type=str, required=True)

    # 1.模型版本开关
    p.add_argument("--token_embedding_mode", type=str, default="taxon_path", choices=["taxon", "taxon_path"])
    p.add_argument("--use_taxonomy_bias", action="store_true", default=False)

    # 2.1.模型主体参数
    p.add_argument("--d_model", type=int, default=256)                # token embedding 的维度，也是模型中间层的维度
    p.add_argument("--nhead", type=int, default=8)                    # 多头注意力中的头数
    p.add_argument("--num_layers", type=int, default=6)               # Transformer Encoder 层数
    p.add_argument("--ff_dim", type=int, default=1024)                # FeedForward 层的中间维度
    p.add_argument("--num_abundance_bins", type=int, default=40)      # 丰度分箱数量

    # 2.2.模型主体参数的协议参数
    p.add_argument("--abundance_mode", type=str, default="abs_log_bins", choices=["abs_log_bins", "rank_bins"])
    p.add_argument("--min_abundance", type=float, default=4e-6)       # 最小丰度阈值
    p.add_argument("--max_seq_len", type=int, default=1024)           # 每个样本保留的最大物种数 (截断长度)

    # 3.1.预训练中的训练主体参数
    p.add_argument("--batch_size", type=int, default=32)              # 每个 batch 的样本数
    p.add_argument("--mask_prob", type=float, default=0.15)           # 预训练 Mask 概率
    p.add_argument("--dropout", type=float, default=0.1)              # Dropout 概率
    p.add_argument("--lr", type=float, default=3e-4)                  # 学习率
    p.add_argument("--weight_decay", type=float, default=1e-2)        # 权重衰减 (L2 正则化)
    p.add_argument("--warmup_ratio", type=float, default=0.02)        # Warmup 占总 optimizer steps 的比例

    # 3.2.预训练中的协议参数
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "plateau"])
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)    # plateau 降学习率的乘法因子
    p.add_argument("--lr_plateau_patience", type=int, default=2)      # plateau 在多少次验证无改善后降 LR
    p.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)   # plateau 的最小学习率

    # 4.预算与验证协议参数
    p.add_argument("--budget_mode", type=str, default="epoch", choices=["epoch", "step"])
    p.add_argument("--max_epochs", type=int, default=None)            # epoch 模式下的最大训练轮数
    p.add_argument("--max_steps", type=int, default=None)             # step 模式下的最大训练步数
    p.add_argument("--val_interval_epochs", type=int, default=None)   # epoch 模式下每多少个 epoch 验证一次
    p.add_argument("--val_interval_steps", type=int, default=None)    # step 模式下每多少步验证一次
    p.add_argument("--limit_train_batches", type=float, default=1.0)  # 每 Epoch 仅使用部分训练数据
    p.add_argument("--limit_val_batches", type=float, default=1.0)    # 每 Epoch 仅使用部分验证数据

    # 5.运行与工程参数
    p.add_argument("--devices", type=int, default=1)                  # 使用的 GPU/设备 数量
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)                    # 随机种子，用于可复现
    p.add_argument("--accumulate_grad_batches", type=int, default=1)  # 梯度累积步数
    p.add_argument("--gradient_clip_val", type=float, default=1.0)    # 梯度裁剪阈值
    p.add_argument("--num_workers", type=int, default=4)              # DataLoader 的 num_workers，默认为 4
    p.add_argument("--log_dir", type=str, default="outputs/pretrain") # 日志保存目录
    p.add_argument("--no_progress_bar", action="store_true", default=False)  # 关闭进度条（远程服务器/nohup 运行时避免刷屏）
    return p


def namespace_to_config(args: argparse.Namespace) -> PretrainRunConfig:
    return PretrainRunConfig(
        h5ad_path=args.h5ad_path,
        token_embedding_mode=args.token_embedding_mode,
        use_taxonomy_bias=args.use_taxonomy_bias,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        num_abundance_bins=args.num_abundance_bins,
        abundance_mode=args.abundance_mode,
        min_abundance=args.min_abundance,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        mask_prob=args.mask_prob,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_plateau_factor=args.lr_plateau_factor,
        lr_plateau_patience=args.lr_plateau_patience,
        lr_plateau_min_lr=args.lr_plateau_min_lr,
        budget_mode=args.budget_mode,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        val_interval_epochs=args.val_interval_epochs,
        val_interval_steps=args.val_interval_steps,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
        no_progress_bar=args.no_progress_bar,
    )


def main() -> None:
    args = build_argparser().parse_args()
    config = namespace_to_config(args)
    train_indices = load_indices(args.train_indices_path, "train")
    val_indices = load_indices(args.val_indices_path, "val")
    run_pretrain_once(config, train_indices, val_indices)


if __name__ == "__main__":
    main()

