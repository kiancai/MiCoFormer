import argparse
import numpy as np
import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from micoformer.datamodules.pretrain_datamodule import MiCoDataModule
from micoformer.models.pretrain_module import MiCoFormerModule


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Stage 0 Pretraining")

    # 0.输入与切分参数
    p.add_argument("--h5ad", type=str, required=True)
    p.add_argument("--train_indices", type=str, required=True)
    p.add_argument("--val_indices", type=str, required=True)

    # 1.模型版本开关
    p.add_argument("--token_embedding_mode", type=str, default="taxon_path", choices=["taxon", "taxon_path"])
    p.add_argument("--use_taxonomy_bias", action="store_true", default=False)

    # 2.1.模型主体参数
    p.add_argument("--d_model", type=int, default=256)                #token embedding 的维度，也是模型中间层的维度
    p.add_argument("--nhead", type=int, default=8)                    # 多头注意力中的头数
    p.add_argument("--num_layers", type=int, default=6)               # Transformer Encoder 层数
    p.add_argument("--ff", type=int, default=1024)                    # FeedForward 层的中间维度
    p.add_argument("--num_abundance_bins", type=int, default=40)      # 丰度分箱数量

    # 2.2.模型主体参数的协议参数
    p.add_argument("--abundance_mode", type=str, default="abs_log_bins", choices=["abs_log_bins", "rank_bins"])
    p.add_argument("--min_abundance", type=float, default=4e-6)       # 最小丰度阈值
    p.add_argument("--max_seq_len", type=int, default=1024)           # 每个样本保留的最大物种数 (截断长度)

    # 3.预训练中的训练参数
    p.add_argument("--batch_size", type=int, default=32)              # 每个 batch 的样本数
    p.add_argument("--mask_prob", type=float, default=0.15)           # 预训练 Mask 概率
    p.add_argument("--dropout", type=float, default=0.1)              # Dropout 概率
    p.add_argument("--lr", type=float, default=3e-4)                  # 学习率
    p.add_argument("--weight_decay", type=float, default=1e-2)        # 权重衰减 (L2 正则化)
    p.add_argument("--warmup_ratio", type=float, default=0.02)        # Warmup 占总 optimizer steps 的比例
    p.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "plateau"])
    p.add_argument("--plateau_factor", type=float, default=0.5)       # plateau 降学习率的乘法因子
    p.add_argument("--plateau_patience", type=int, default=2)         # plateau 在多少次验证无改善后降 LR
    p.add_argument("--plateau_min_lr", type=float, default=1e-6)      # plateau 的最小学习率

    # 4. 预算与验证协议参数
    p.add_argument("--budget_mode", type=str, default="epoch", choices=["epoch", "step"])
    p.add_argument("--max_epochs", type=int, default=100)             # 最大训练轮数
    p.add_argument("--max_steps", type=int, default=None)             # 仅在 step 模式下生效的最大训练步数
    p.add_argument("--check_val_every_n_epoch", type=int, default=1)  # epoch 模式下每多少个 epoch 验证一次
    p.add_argument("--val_check_interval", type=int, default=None)    # step 模式下每多少步验证一次
    p.add_argument("--limit_train_batches", type=float, default=1.0)  # 每 Epoch 仅使用部分训练数据
    p.add_argument("--limit_val_batches", type=float, default=1.0)    # 每 Epoch 仅使用部分验证数据

    # 5. 运行与工程参数
    p.add_argument("--devices", type=int, default=1)                  # 使用的 GPU/设备 数量
    p.add_argument("--precision", type=str, default="auto", choices=["auto", "16-mixed", "32"])  # 训练精度
    p.add_argument("--seed", type=int, default=42)                    # 随机种子，用于可复现
    p.add_argument("--accumulate_grad_batches", type=int, default=1)  # 梯度累积步数
    p.add_argument("--gradient_clip_val", type=float, default=1.0)    # 梯度裁剪阈值
    p.add_argument("--num_workers", type=int, default=4)              # DataLoader 的 num_workers，默认为 4
    p.add_argument("--log_dir", type=str, default="tmp/logs")         # 日志保存目录
    p.add_argument("--no_progress_bar", action="store_true", default=False)  # 关闭进度条（远程服务器/nohup 运行时避免刷屏）

    return p


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup_ratio must satisfy 0 <= warmup_ratio < 1.")

    if not 0.0 < args.plateau_factor < 1.0:
        raise ValueError("--plateau_factor must satisfy 0 < plateau_factor < 1.")
    if args.plateau_patience < 0:
        raise ValueError("--plateau_patience must be >= 0.")
    if args.plateau_min_lr < 0:
        raise ValueError("--plateau_min_lr must be >= 0.")

    if args.check_val_every_n_epoch <= 0:
        raise ValueError("--check_val_every_n_epoch must be >= 1.")

    if args.budget_mode == "epoch":
        if args.max_epochs <= 0:
            raise ValueError("--max_epochs must be > 0 when budget_mode=epoch.")
        if args.max_steps is not None:
            raise ValueError("--max_steps is step-mode only. Do not pass it when budget_mode=epoch.")
        if args.val_check_interval is not None:
            raise ValueError("--val_check_interval is step-mode only. Use --check_val_every_n_epoch in epoch mode.")
    else:
        if args.max_steps is None or args.max_steps <= 0:
            raise ValueError("--max_steps must be provided and > 0 when budget_mode=step.")


def main():
    args = build_argparser().parse_args()
    validate_args(args)
    L.seed_everything(args.seed, workers=True)

    # 自动根据设备选择精度，避免 CPU 下使用 16-mixed 报错
    if args.precision == "auto":
        chosen_precision = "16-mixed" if torch.cuda.is_available() else "32"
    else:
        chosen_precision = args.precision
    print(f"Using precision={chosen_precision}")

    # 加载分割索引（由 scripts/1.make_splits.py 分别生成的 .npy 文件）
    print(f"Loading train indices from {args.train_indices} ...")
    train_indices = np.load(args.train_indices)
    print(f"Loading val indices from {args.val_indices} ...")
    val_indices = np.load(args.val_indices)
    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")
    print(f"Budget mode: {args.budget_mode}")
    if args.budget_mode == "epoch":
        print(f"Max epochs: {args.max_epochs}, check_val_every_n_epoch={args.check_val_every_n_epoch}")
    else:
        print(f"Max steps: {args.max_steps}, val_check_interval={args.val_check_interval}")
    print(f"LR scheduler: {args.lr_scheduler}, warmup_ratio={args.warmup_ratio}")

    # 1. 初始化数据模块
    print(f"Initializing DataModule...")
    dm = MiCoDataModule(
        h5ad_path=args.h5ad,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,               # 每个样本保留的最大物种数 (截断长度)，默认 1024
        mask_prob=args.mask_prob,                   # 预训练 Mask 概率 (默认 15%)
        num_abundance_bins=args.num_abundance_bins, # 丰度分箱数量
        min_abundance=args.min_abundance,           # 最小丰度阈值
        abundance_mode=args.abundance_mode,         # 丰度编码模式（"abs_log_bins" 或 "rank_bins"）
        token_embedding_mode=args.token_embedding_mode,  # 选择 token embedding 方式
        use_taxonomy_bias=args.use_taxonomy_bias,         # R2：taxonomy 距离注意力偏置
    )
    
    # 2. 初始化模型
    print(f"Initializing Model with d_model={args.d_model}, layers={args.num_layers}")
    print(f"Token embedding mode: {args.token_embedding_mode}")
    model = MiCoFormerModule(
        genus_vocab_size=dm.genus_vocab_size,     # taxon 模式使用；taxon_path 模式传 None 亦可
        total_abundance_bins=dm.total_abundance_bins,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        pad_taxon_id=dm.special_ids["pad_taxon_id"],
        pad_bin_id=dm.special_ids["pad_bin_id"],
        token_embedding_mode=args.token_embedding_mode,
        rank_vocab_sizes=dm.rank_vocab_sizes,
        use_taxonomy_bias=args.use_taxonomy_bias,   # R2：taxonomy 距离注意力偏置
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler=args.lr_scheduler,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_min_lr=args.plateau_min_lr,
        budget_mode=args.budget_mode,
    )

    # 3. 设置日志记录器与回调（CSV 用于离线查看，TensorBoard 用于实时监控）
    csv_logger = CSVLogger(save_dir=args.log_dir, name="pretrain_stage0")
    tb_logger  = TensorBoardLogger(save_dir=args.log_dir, name="pretrain_stage0")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        filename="micoformer-{epoch:02d}-{val/loss:.4f}"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. 初始化 Lightning Trainer
    trainer_kwargs = dict(
        devices=args.devices,
        precision=chosen_precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, # 梯度裁剪
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=[csv_logger, tb_logger],
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.log_dir,
    )
    if args.budget_mode == "epoch":
        trainer_kwargs["max_epochs"] = args.max_epochs
        trainer_kwargs["check_val_every_n_epoch"] = args.check_val_every_n_epoch
    else:
        trainer_kwargs["max_steps"] = args.max_steps
    if args.no_progress_bar:
        trainer_kwargs["enable_progress_bar"] = False
    if args.budget_mode == "step" and args.val_check_interval is not None:
        trainer_kwargs["val_check_interval"] = args.val_check_interval
    trainer = L.Trainer(**trainer_kwargs)

    # 5. 开始训练
    print("Starting training...")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
