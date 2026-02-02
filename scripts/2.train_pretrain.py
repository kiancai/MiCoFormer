import argparse
import os
import numpy as np
import anndata as ad
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from micoformer.datamodules.mico_datamodule import MiCoDataModule
from micoformer.models.module import MiCoFormerModule


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer Stage 0 Pretraining (预训练启动脚本)")

    # --- 数据相关参数 (Data) ---
    p.add_argument("--h5ad", type=str, required=True, help="处理好的 AnnData (.h5ad) 文件路径")
    p.add_argument("--batch_size", type=int, default=32, help="批次大小")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader 的并行加载进程数")
    p.add_argument("--max_seq_len", type=int, default=1024, help="每个样本保留的最大物种数 (截断长度)")
    p.add_argument("--mask_prob", type=float, default=0.15, help="预训练 Mask 概率 (默认 15%)")
    
    # 丰度分箱参数
    p.add_argument("--num_abundance_bins", type=int, default=40, help="丰度分箱数量")
    p.add_argument("--min_abundance", type=float, default=4e-6, help="最小丰度阈值")
    p.add_argument("--abundance_mode", type=str, default="abs_log_bins", choices=["abs_log_bins", "rank_bins"], help="丰度编码模式")

    # --- 模型架构参数 (Model Architecture) ---
    p.add_argument("--d_model", type=int, default=256, help="Transformer 隐层维度")
    p.add_argument("--nhead", type=int, default=8, help="多头注意力的头数")
    p.add_argument("--num_layers", type=int, default=6, help="Encoder 层数")
    p.add_argument("--ff", type=int, default=1024, help="FeedForward 层的中间维度")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout 概率")

    # --- 优化器与 Scheduler 参数 (Optimizer) ---
    p.add_argument("--lr", type=float, default=3e-4, help="学习率 (Learning Rate)")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="权重衰减 (L2 正则化)")
    p.add_argument("--warmup_steps", type=int, default=2000, help="Warmup 步数")
    p.add_argument("--max_steps", type=int, default=100000, help="最大训练步数 (用于 Cosine Decay 计算)")

    # --- 训练控制参数 (Training Control) ---
    p.add_argument("--max_epochs", type=int, default=100, help="最大训练轮数")
    p.add_argument("--devices", type=int, default=1, help="使用的 GPU/设备 数量")
    p.add_argument("--precision", type=str, default="16-mixed", help="训练精度 (推荐 16-mixed)")
    p.add_argument("--accumulate_grad_batches", type=int, default=1, help="梯度累积步数")
    p.add_argument("--gradient_clip_val", type=float, default=1.0, help="梯度裁剪阈值")
    p.add_argument("--limit_train_batches", type=float, default=1.0, help="每 Epoch 仅使用部分训练数据")
    p.add_argument("--limit_val_batches", type=float, default=1.0, help="每 Epoch 仅使用部分验证数据")
    p.add_argument("--log_dir", type=str, default="tmp/logs", help="日志保存目录")

    return p


def main():
    args = build_argparser().parse_args()

    print(f"Reading metadata from {args.h5ad} to generate splits...")
    ada = ad.read_h5ad(args.h5ad, backed="r")
    n_samples = ada.n_obs
    all_indices = np.random.permutation(n_samples)
    
    # 95% 训练, 5% 验证
    # n_val = int(n_samples * 0.05)
    # train_indices = all_indices[:-n_val]
    # val_indices = all_indices[-n_val:]
    train_indices = all_indices[:1000]
    val_indices = all_indices[1000:1100]
    print(f"Total samples: {n_samples}. Train: {len(train_indices)}, Val: {len(val_indices)}")

    # 1. 初始化数据模块
    print(f"Initializing DataModule...")
    dm = MiCoDataModule(
        h5ad_path=args.h5ad,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=None,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        mask_prob=args.mask_prob,
        num_abundance_bins=args.num_abundance_bins,
        min_abundance=args.min_abundance,
        abundance_mode=args.abundance_mode,
    )
    
    # 手动调用 setup 以便获取 vocab_size 等信息用于初始化模型
    dm.setup()

    # 2. 初始化模型
    print(f"Initializing Model with d_model={args.d_model}, layers={args.num_layers}")
    model = MiCoFormerModule(
        vocab_size=dm.vocab_size,             # 从数据中动态获取
        num_abundance_bins=dm.num_abundance_bins, # 从数据中动态获取
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        pad_taxon_id=dm.special_ids["pad_taxon_id"], # 修正参数名
        pad_bin_id=dm.special_ids["pad_bin_id"],     # 新增参数
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )

    # 3. 设置日志记录器与回调
    logger = CSVLogger(save_dir=args.log_dir, name="pretrain_stage0")
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        filename="micoformer-{epoch:02d}-{val/loss:.4f}"
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # 4. 初始化 Lightning Trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps, # 确保与 Scheduler 一致
        devices=args.devices,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val, # 梯度裁剪
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        default_root_dir=args.log_dir,
    )

    # 5. 开始训练
    print("Starting training...")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

