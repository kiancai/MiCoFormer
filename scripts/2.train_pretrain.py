"""
scripts/2.train_pretrain.py — MiCoFormer 预训练入口脚本

使用示例：

  # epoch 模式
  python scripts/2.train_pretrain.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --train_indices_path data/processed/splits/train.npy \
      --val_indices_path data/processed/splits/val.npy \
      --budget_mode epoch --max_epochs 20 --val_interval_epochs 3

  # step 模式
  python scripts/2.train_pretrain.py \
      --h5ad_path data/processed/microbiome_dataset.h5ad \
      --train_indices_path data/processed/splits/train.npy \
      --val_indices_path data/processed/splits/val.npy \
      --budget_mode step --max_steps 15000 --val_interval_steps 500
"""

import argparse

import numpy as np
from lightning.pytorch.utilities import rank_zero_info

from micoformer.utils.train_utils import int_or_float
from micoformer.workflows.pretrain import PretrainRunConfig, run_pretrain_once


TAG = "[train_pretrain]"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Stage 0 Pretraining")

    # 0.输入与切分参数
    p.add_argument("--h5ad_path", type=str, required=True)
    p.add_argument("--train_indices_path", type=str, required=True)
    p.add_argument("--val_indices_path", type=str, required=True)

    # 1.模型版本开关
    # V4 R2:距离驱动的 attention bias
    #   none  : baseline,无 bias
    #   taxo  : 离散 7-bucket 查 varp['taxo_dist']
    #   phylo : 3 层 MLP 查 varp['phylo_dist'](V5 默认)
    p.add_argument("--bias_type", type=str, default="phylo", choices=["none", "taxo", "phylo"])
    p.add_argument("--phylo_mlp_hidden", type=int, default=64,
                   help="phylo bias MLP 隐藏层维度(仅 --bias_type phylo 时生效;V5 默认 64,3 层 MLP)")

    # V5 新增:三段相加 + PMA + metadata 多任务
    p.add_argument("--abundance_encoding", type=str, default="mlp", choices=["mlp", "bin"],
                   help="V5:abundance 输入编码方式;mlp=连续 MLP(默认),bin=旧离散 embedding")
    p.add_argument("--abundance_value_transform", type=str, default="rclr_sigma",
                   choices=["rclr_sigma", "rclr", "rank", "presence", "raw"],
                   help="V5 §4.2:present-only abundance 数值写法(编码消融);rclr_sigma=现状默认,"
                        "rclr=去σ,rank=排名,presence=只打勾(MLM退化、慎用),raw=相对丰度原值")
    p.add_argument("--abundance_loss", type=str, default="huber", choices=["huber", "bin_ce"],
                   help="V5:abundance MLM loss;huber=连续回归(默认),bin_ce=旧 bin 分类")
    p.add_argument("--no_phylo_pe", action="store_true", default=False,
                   help="V5:禁用 PhyloPE(默认启用)")
    p.add_argument("--phylo_pe_hidden", type=int, default=128,
                   help="PhyloPE 投影 MLP 中间维度,默认 128")
    p.add_argument("--pooling_mode", type=str, default="pma", choices=["pma", "mean_pool"],
                   help="V5:sample-level pooling;pma(默认) | mean_pool")
    p.add_argument("--pma_nhead", type=int, default=4)
    p.add_argument("--pma_k", type=int, default=1)
    p.add_argument("--no_metadata_task", action="store_true", default=False,
                   help="V5:禁用 EnvCategory 多任务监督(默认启用)")
    p.add_argument("--metadata_loss_weight", type=float, default=0.3,
                   help="V5:λ_meta(metadata loss 权重),默认 0.3")
    p.add_argument("--huber_beta", type=float, default=1.0)

    # 2.1.模型主体参数
    p.add_argument("--d_model", type=int, default=256)                # token embedding 的维度，也是模型中间层的维度
    p.add_argument("--nhead", type=int, default=8)                    # 多头注意力中的头数
    p.add_argument("--num_layers", type=int, default=6)               # Transformer Encoder 层数
    p.add_argument("--ff_dim", type=int, default=None,
                   help="FeedForward 绝对维度，与 --ff_ratio 互斥。不指定时使用 ff_ratio。")
    p.add_argument("--ff_ratio", type=int, default=None,
                   help="FeedForward 比例（dim_ff = d_model × ff_ratio），与 --ff_dim 互斥。默认 4。")
    p.add_argument("--num_abundance_bins", type=int, default=40)      # 丰度分箱数量

    # 2.2.模型主体参数的协议参数
    p.add_argument("--abundance_mode", type=str, default="abs_log_bins", choices=["abs_log_bins", "rank_bins"])
    p.add_argument("--min_abundance", type=float, default=4e-6)       # 最小丰度阈值
    p.add_argument("--max_seq_len", type=int, default=1024)           # 每个样本保留的最大物种数 (截断长度)

    # 3.1.预训练中的训练主体参数
    p.add_argument("--batch_size", type=int, default=32,
                   help="per-GPU micro-batch（DDP 下有效 batch = batch_size × devices × accumulate_grad_batches）")
    p.add_argument("--mask_prob", type=float, default=0.15)           # 预训练 Mask 概率
    p.add_argument("--dropout", type=float, default=0.1)              # Dropout 概率
    p.add_argument("--lr", type=float, default=3e-4)                  # 学习率
    p.add_argument("--weight_decay", type=float, default=1e-2)        # 权重衰减 (L2 正则化)
    p.add_argument("--warmup_ratio", type=float, default=0.02)        # Warmup 占总 optimizer steps 的比例

    # 3.2.预训练中的协议参数
    # lr_scheduler_type 决定学习率下降方式：
    # - cosine：warmup 后按 cosine 平滑衰减
    # - plateau：warmup 后根据 val/loss 是否停滞来自动降 LR
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["cosine", "plateau"])
    p.add_argument("--lr_plateau_factor", type=float, default=0.5)       # plateau 降学习率的乘法因子
    p.add_argument("--lr_plateau_patience", type=int, default=2)         # plateau 在多少次验证无改善后降 LR
    p.add_argument("--lr_plateau_min_lr", type=float, default=1e-6)      # plateau 的最小学习率

    # 4. 预算与验证协议参数
    # budget_mode 决定"训练预算"的单位：
    # - epoch：更适合当前这种数据规模不算特别大的实验
    # - step：更适合超大数据集或只想固定 optimizer 更新次数的场景
    p.add_argument("--budget_mode", type=str, default="epoch", choices=["epoch", "step"])
    p.add_argument("--max_epochs", type=int, default=None)             # epoch 模式下的最大训练轮数
    p.add_argument("--max_steps", type=int, default=None)              # step 模式下的最大训练步数
    p.add_argument("--val_interval_epochs", type=int, default=None)    # epoch 模式下每多少个 epoch 验证一次
    p.add_argument("--val_interval_steps", type=int, default=None)     # step 模式下每多少步验证一次
    p.add_argument("--limit_train_batches", type=int_or_float, default=1.0)   # 每 Epoch 仅使用部分训练数据 (float=比例 / int=绝对 batch 数)
    p.add_argument("--limit_val_batches", type=int_or_float, default=1.0)     # 每 Epoch 仅使用部分验证数据 (float=比例 / int=绝对 batch 数)

    # 4.1. Early stopping（0=禁用，与 finetune 对称）
    p.add_argument("--early_stopping_patience", type=int, default=0,)  #Early stopping patience（0 表示禁用）
    p.add_argument("--early_stopping_min_delta", type=float, default=0.0,)  #Early stopping 最小改善阈值。
    p.add_argument("--save_top_k", type=int, default=3,
                   help="ModelCheckpoint 保留数：-1=保存每个验证 ckpt（长训练回头看用），>0=只留最优 K 个")

    # 5. 运行与工程参数
    p.add_argument("--devices", type=int, default=1)                   # 使用的 GPU/设备 数量（单节点内卡数）
    p.add_argument("--num_nodes", type=int, default=1)                 # 多节点 DDP 节点数（单节点保持 1）
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])
    p.add_argument("--seed", type=int, default=42)                     # 随机种子，用于可复现
    p.add_argument("--accumulate_grad_batches", type=int, default=1)   # 梯度累积步数
    p.add_argument("--gradient_clip_val", type=float, default=1.0)     # 梯度裁剪阈值
    p.add_argument("--grad_checkpointing", action="store_true", default=False,
                   help="激活重算（以时间换显存，单卡可开更大 batch）；默认关，本次正式训练不用")
    p.add_argument("--num_workers", type=int, default=4)               # DataLoader 的 num_workers
    p.add_argument("--log_dir", type=str, default="tmp/logs")          # 日志保存目录
    p.add_argument(
        "--run_name",
        type=str,
        default="pretrain_stage0",
        help="日志/checkpoint 子目录名，对应 run_pretrain_once 的 log_subdir。",
    )
    p.add_argument("--no_progress_bar", action="store_true", default=False)

    # DAPT 续训:从已有 ckpt 加载初始权重(非 trainer resume,只 init state_dict)
    p.add_argument(
        "--init_from_ckpt", type=str, default=None,
        help="从该 ckpt 加载初始 state_dict;buffer 缺失会自动跳过(strict=False)。用于 DAPT 二阶段。",
    )

    return p


def _args_to_config(args: argparse.Namespace) -> PretrainRunConfig:
    return PretrainRunConfig(
        h5ad_path=args.h5ad_path,
        bias_type=args.bias_type,
        phylo_mlp_hidden=args.phylo_mlp_hidden,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        ff_dim=args.ff_dim,
        ff_ratio=args.ff_ratio,
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
        num_nodes=args.num_nodes,
        precision=args.precision,
        seed=args.seed,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        grad_checkpointing=args.grad_checkpointing,
        num_workers=args.num_workers,
        log_dir=args.log_dir,
        no_progress_bar=args.no_progress_bar,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        save_top_k=args.save_top_k,
        # V5
        abundance_encoding=args.abundance_encoding,
        abundance_value_transform=args.abundance_value_transform,
        abundance_loss=args.abundance_loss,
        use_phylo_pe=not args.no_phylo_pe,
        phylo_pe_hidden=args.phylo_pe_hidden,
        pooling_mode=args.pooling_mode,
        pma_nhead=args.pma_nhead,
        pma_k=args.pma_k,
        use_metadata_task=not args.no_metadata_task,
        metadata_loss_weight=args.metadata_loss_weight,
        huber_beta=args.huber_beta,
        init_from_ckpt=args.init_from_ckpt,
    )


def main():
    args = build_argparser().parse_args()
    config = _args_to_config(args)

    rank_zero_info(f"{TAG} Loading train indices from {args.train_indices_path} ...")
    train_indices = np.load(args.train_indices_path)
    rank_zero_info(f"{TAG} Loading val indices from {args.val_indices_path} ...")
    val_indices = np.load(args.val_indices_path)

    run_pretrain_once(config, train_indices, val_indices, log_subdir=args.run_name)


if __name__ == "__main__":
    main()
