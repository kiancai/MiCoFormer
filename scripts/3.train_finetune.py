"""
scripts/3.train_finetune.py — MiCoFormer 下游分类微调入口脚本

只支持显式索引模式：
  - 必须提供 --train_indices_path 和 --val_indices_path
  - 可选提供 --test_indices_path

使用示例：

  # epoch 模式
  python scripts/3.train_finetune.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --pretrained_ckpt outputs/pretrain/checkpoints/best.ckpt \
      --train_indices_path data/processed/splits/finetune/train.npy \
      --val_indices_path data/processed/splits/finetune/val.npy \
      --test_indices_path data/processed/splits/finetune/test.npy \
      --label_fields Phenotype \
      --budget_mode epoch --max_epochs 20 --val_interval_epochs 1

  # step 模式
  python scripts/3.train_finetune.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --pretrained_ckpt outputs/pretrain/checkpoints/best.ckpt \
      --train_indices_path data/processed/splits/finetune/train.npy \
      --val_indices_path data/processed/splits/finetune/val.npy \
      --label_fields Phenotype \
      --budget_mode step --max_steps 5000 --val_interval_steps 200
"""

from __future__ import annotations

import argparse

import numpy as np
from lightning.pytorch.utilities import rank_zero_info

from micoformer.utils.train_utils import int_or_float, str2bool
from micoformer.workflows.finetune import (
    FinetuneRunConfig,
    build_label_configs,
    run_finetune_once,
)


TAG = "[train_finetune]"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Downstream Classification Fine-tuning")

    # 0.输入与切分参数
    p.add_argument("--h5ad_path", "--h5ad", dest="h5ad_path", type=str, required=True)
    p.add_argument("--train_indices_path", "--train_indices", dest="train_indices_path", type=str, required=True)
    p.add_argument("--val_indices_path", "--val_indices", dest="val_indices_path", type=str, required=True)
    p.add_argument("--test_indices_path", "--test_indices", dest="test_indices_path", type=str, default=None)
    p.add_argument("--pretrained_ckpt", type=str, required=True)

    # 1.下游任务标签参数
    p.add_argument("--label_fields", type=str, nargs="+", required=True)
    p.add_argument("--label_values", type=str, nargs="*", default=None,
                   help=(
                       "限定每个任务的有效取值，格式为 'Field=v1,v2'，多字段空格分隔。"
                       "示例：--label_values \"Phenotype=Health,Disease\" \"Smoking=Yes,No\""
                   ))

    # 2.1.分类头 / pooling 参数
    p.add_argument("--pooling_mode", type=str, default="mean_pool",
                   choices=["sample", "mean_pool", "sample_and_mean"])
    p.add_argument("--head_hidden_dim", type=int, default=0)          # 0 表示线性分类头，>0 表示 MLP hidden dim
    p.add_argument("--head_dropout", type=float, default=0.1)         # 分类头 dropout
    p.add_argument("--freeze_encoder", type=str2bool, default=False,
                   metavar="BOOL")                                      # 是否冻结 encoder（true/false/yes/no/1/0），默认 false

    # 2.2.数据协议参数
    p.add_argument("--batch_size", type=int, default=32)              # 每个 batch 的样本数
    p.add_argument("--max_seq_len", type=int, default=1024)           # 每个样本保留的最大物种数 (截断长度)

    # 3.1.微调中的训练主体参数
    p.add_argument("--lr_head", type=float, default=1e-3)             # 分类头学习率
    p.add_argument("--lr_encoder", type=float, default=1e-5)          # Encoder 学习率（仅在不冻结时生效）
    p.add_argument("--weight_decay", type=float, default=1e-2)        # 权重衰减 (L2 正则化)
    p.add_argument("--warmup_ratio", type=float, default=0.1)         # Warmup 占总 optimizer steps 的比例

    # 3.2.微调中的协议参数
    # lr_scheduler_type 决定学习率下降方式：
    # - cosine：warmup 后按 cosine 平滑衰减
    # - plateau：warmup 后根据主指标 (val/{task}/f1_macro) 是否停滞来自动降 LR
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "plateau"])
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)       # plateau 降学习率的乘法因子
    p.add_argument("--lr_plateau_patience", type=int, default=2)         # plateau 在多少次验证无改善后降 LR
    p.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)      # plateau 的最小学习率

    # 4. 预算与验证协议参数
    # budget_mode 决定"训练预算"的单位：
    # - epoch：更适合当前这种数据规模不算特别大的实验
    # - step：更适合超大数据集或只想固定 optimizer 更新次数的场景
    p.add_argument("--budget_mode", type=str, default="epoch", choices=["epoch", "step"])
    p.add_argument("--max_epochs", type=int, default=None)               # epoch 模式下的最大训练轮数
    p.add_argument("--max_steps", type=int, default=None)                # step 模式下的最大训练步数
    p.add_argument("--val_interval_epochs", type=int, default=None)      # epoch 模式下每多少个 epoch 验证一次
    p.add_argument("--val_interval_steps", type=int, default=None)       # step 模式下每多少步验证一次
    p.add_argument("--early_stopping_patience", type=int, default=10)    # Early stopping patience，0 表示禁用
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0)  # Early stopping 最小改善阈值

    # 4.1. Trainer 控制参数（与 pretrain 对称）
    p.add_argument("--gradient_clip_val", type=float, default=1.0)          # 梯度裁剪阈值
    p.add_argument("--accumulate_grad_batches", type=int, default=1)        # 梯度累积步数
    p.add_argument("--limit_train_batches", type=int_or_float, default=1.0)    # 每 Epoch 仅使用部分训练数据 (float=比例 / int=绝对 batch 数)
    p.add_argument("--limit_val_batches", type=int_or_float, default=1.0)      # 每 Epoch 仅使用部分验证数据 (float=比例 / int=绝对 batch 数)

    # 5. 运行与工程参数
    p.add_argument("--devices", type=int, default=1)                     # 使用的 GPU/设备 数量
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)                       # 随机种子，用于可复现
    p.add_argument("--num_workers", type=int, default=4)                 # DataLoader 的 num_workers
    p.add_argument("--log_dir", type=str, default="tmp/logs")            # 日志保存目录
    p.add_argument(
        "--run_name",
        type=str,
        default="finetune_stage0",
        help="日志/checkpoint 子目录名，对应 run_finetune_once 的 log_subdir。",
    )
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
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        lr_head=args.lr_head,
        lr_encoder=args.lr_encoder,
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
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        devices=args.devices,
        precision=args.precision,
        seed=args.seed,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
        no_progress_bar=args.no_progress_bar,
    )


def _print_results(results: dict) -> None:
    rank_zero_info(f"{TAG} Results:")
    for split, metrics in results.items():
        if not isinstance(metrics, dict):
            # best_model_path / best_score 等标量字段直接打印
            rank_zero_info(f"{TAG}   {split}: {metrics}")
            continue
        rank_zero_info(f"{TAG}   [{split}]")
        for key, value in sorted(metrics.items()):
            rank_zero_info(f"{TAG}     {key}: {value}")


def main() -> None:
    args = build_argparser().parse_args()
    config = _args_to_config(args)
    # nargs="*" 返回 list[str]，join 后传给 build_label_configs（支持 'Field=v1,v2' 格式）
    label_values_str = " ".join(args.label_values) if args.label_values else None
    label_configs = build_label_configs(args.label_fields, label_values_str)

    rank_zero_info(f"{TAG} Loading train indices from {args.train_indices_path} ...")
    train_indices = np.load(args.train_indices_path)
    rank_zero_info(f"{TAG} Loading val indices from {args.val_indices_path} ...")
    val_indices = np.load(args.val_indices_path)
    test_indices = None
    if args.test_indices_path is not None:
        rank_zero_info(f"{TAG} Loading test indices from {args.test_indices_path} ...")
        test_indices = np.load(args.test_indices_path)

    results = run_finetune_once(
        config,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        label_configs=label_configs,
        log_subdir=args.run_name,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
