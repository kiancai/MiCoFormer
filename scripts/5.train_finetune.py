"""
scripts/5.train_finetune.py — 下游分类微调训练入口

单次训练：
  python scripts/5.train_finetune.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --train_indices splits/train.npy --val_indices splits/val.npy \
      --pretrained_ckpt path/to/pretrained.ckpt \
      --label_fields Phenotype \
      --label_values '{"Phenotype": ["Health", "Disease"]}'

K-fold 自动循环：
  python scripts/5.train_finetune.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --kfold_dir splits/kfold_PRJNA123456/ \
      --pretrained_ckpt path/to/pretrained.ckpt \
      --label_fields Phenotype \
      --label_values '{"Phenotype": ["Health", "Disease"]}' \
      --log_dir tmp/logs/finetune_kfold
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

from micoformer.datamodules.classification_datamodule import ClassificationDataModule
from micoformer.models.classifier_module import MiCoFormerClassifier
from micoformer.models.module import MiCoFormerModule


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiCoFormer downstream classification fine-tuning")

    # --- 数据 ---
    p.add_argument("--h5ad", type=str, required=True, help=".h5ad file path")
    p.add_argument("--train_indices", type=str, default=None, help="Train indices .npy (single run)")
    p.add_argument("--val_indices", type=str, default=None, help="Val indices .npy (single run)")
    p.add_argument("--test_indices", type=str, default=None, help="Test indices .npy (optional)")
    p.add_argument("--kfold_dir", type=str, default=None, help="K-fold directory (auto-detect folds)")

    # --- 标签 ---
    p.add_argument(
        "--label_fields", type=str, nargs="+", required=True,
        help="obs label fields (e.g., Phenotype Sample_Site)",
    )
    p.add_argument(
        "--label_values", type=str, default=None,
        help='JSON dict mapping field -> valid values, e.g., \'{"Phenotype": ["Health", "Disease"]}\'',
    )

    # --- 预训练模型 ---
    p.add_argument("--pretrained_ckpt", type=str, required=True, help="Pretrained checkpoint path")

    # --- 分类模型配置 ---
    p.add_argument("--pooling_mode", type=str, default="mean_pool",
                    choices=["sample", "mean_pool", "sample_and_mean"])
    p.add_argument("--head_hidden_dim", type=int, default=0, help="0=linear probe, >0=MLP hidden dim")
    p.add_argument("--head_dropout", type=float, default=0.1)
    p.add_argument("--freeze_encoder", action="store_true", default=False)

    # --- 优化器 ---
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_encoder", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--max_epochs", type=int, default=100)

    # --- 训练控制 ---
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--precision", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="tmp/logs/finetune")
    p.add_argument("--no_progress_bar", action="store_true", default=False)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0=disable)")
    p.add_argument("--val_check_interval", type=int, default=None)
    p.add_argument("--gradient_clip_val", type=float, default=1.0)

    return p


def _parse_label_values(label_values_str: str | None) -> dict[str, list[str]] | None:
    if label_values_str is None:
        return None
    return json.loads(label_values_str)


def _get_data_params_from_ckpt(ckpt_path: str) -> dict:
    """从预训练 checkpoint 的 hparams 中读取数据参数，确保下游与预训练一致。"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]
    return {
        "token_embedding_mode": hp.get("token_embedding_mode", "taxon_path"),
        "use_taxonomy_bias": hp.get("use_taxonomy_bias", False),
    }


def run_single(
    args,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray | None,
    log_subdir: str,
) -> dict:
    """执行单次训练，返回最佳验证指标。"""
    L.seed_everything(args.seed, workers=True)

    # 解析标签配置
    label_values_map = _parse_label_values(args.label_values)
    label_configs = []
    for field in args.label_fields:
        cfg = {"field": field}
        if label_values_map and field in label_values_map:
            cfg["values"] = label_values_map[field]
        label_configs.append(cfg)

    # 从 checkpoint 读数据配置
    ckpt_data_params = _get_data_params_from_ckpt(args.pretrained_ckpt)

    # 数据模块
    dm = ClassificationDataModule(
        h5ad_path=args.h5ad,
        label_configs=label_configs,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_seq_len=args.max_seq_len,
        token_embedding_mode=ckpt_data_params["token_embedding_mode"],
        use_taxonomy_bias=ckpt_data_params["use_taxonomy_bias"],
    )

    # 构建任务配置
    task_configs = [
        {"name": cfg["field"], "num_classes": cfg["num_classes"]}
        for cfg in dm.task_configs
    ]

    # 精度设置
    if args.precision == "auto":
        chosen_precision = "16-mixed" if torch.cuda.is_available() else "32"
    else:
        chosen_precision = args.precision

    # 分类模型
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

    # 日志 & 回调
    csv_logger = CSVLogger(save_dir=args.log_dir, name=log_subdir)
    tb_logger = TensorBoardLogger(save_dir=args.log_dir, name=log_subdir)

    # 监控第一个任务的 f1_macro 作为 checkpoint 选择依据
    primary_task = task_configs[0]["name"]
    monitor_metric = f"val/{primary_task}/f1_macro"

    callbacks = [
        ModelCheckpoint(
            monitor=monitor_metric,
            mode="max",
            save_top_k=1,
            filename="best-{epoch:02d}-{" + monitor_metric + ":.4f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    if args.patience > 0:
        callbacks.append(
            EarlyStopping(monitor=monitor_metric, mode="max", patience=args.patience)
        )

    # Trainer
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

    # 训练
    print(f"Starting fine-tuning (log_subdir={log_subdir}) ...")
    trainer.fit(model, datamodule=dm)

    # 测试（如果有 test 集）
    results = {}
    if test_indices is not None:
        test_results = trainer.test(model, datamodule=dm, ckpt_path="best")
        if test_results:
            results["test"] = test_results[0]

    # 收集最佳验证指标
    if trainer.callback_metrics:
        results["val"] = {k: v.item() if hasattr(v, "item") else v for k, v in trainer.callback_metrics.items()}

    return results


def run_kfold(args) -> None:
    """自动检测 kfold_dir 中的 fold 文件并循环训练。"""
    # 检测所有 fold
    pattern = os.path.join(args.kfold_dir, "fold_*_train.npy")
    train_files = sorted(glob.glob(pattern))
    if not train_files:
        raise FileNotFoundError(f"No fold files found matching {pattern}")

    fold_indices = []
    for f in train_files:
        m = re.search(r"fold_(\d+)_train\.npy", os.path.basename(f))
        if m:
            fold_indices.append(int(m.group(1)))
    fold_indices.sort()
    print(f"Detected {len(fold_indices)} folds: {fold_indices}")

    all_results = []
    for fold_i in fold_indices:
        train_path = os.path.join(args.kfold_dir, f"fold_{fold_i}_train.npy")
        val_path = os.path.join(args.kfold_dir, f"fold_{fold_i}_val.npy")

        print(f"\n{'='*60}")
        print(f"Fold {fold_i}")
        print(f"{'='*60}")

        train_idx = np.load(train_path)
        val_idx = np.load(val_path)

        results = run_single(
            args,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=None,
            log_subdir=f"fold_{fold_i}",
        )
        all_results.append(results)

    # 汇总各 fold 的 val 指标
    print(f"\n{'='*60}")
    print("K-fold Summary")
    print(f"{'='*60}")

    # 收集所有 val 指标的 key
    val_keys = set()
    for r in all_results:
        if "val" in r:
            val_keys.update(r["val"].keys())

    # 只汇总包含任务指标的 key
    for key in sorted(val_keys):
        values = []
        for r in all_results:
            if "val" in r and key in r["val"]:
                v = r["val"][key]
                if isinstance(v, (int, float)):
                    values.append(v)
        if values:
            mean = np.mean(values)
            std = np.std(values)
            print(f"  {key}: {mean:.4f} +/- {std:.4f}")


def main():
    args = build_argparser().parse_args()

    if args.kfold_dir is not None:
        # K-fold 模式
        run_kfold(args)
    elif args.train_indices is not None and args.val_indices is not None:
        # 单次训练模式
        train_idx = np.load(args.train_indices)
        val_idx = np.load(args.val_indices)
        test_idx = np.load(args.test_indices) if args.test_indices else None
        results = run_single(
            args,
            train_indices=train_idx,
            val_indices=val_idx,
            test_indices=test_idx,
            log_subdir="finetune",
        )
        print("\nResults:")
        for split, metrics in results.items():
            print(f"  [{split}]")
            if isinstance(metrics, dict):
                for k, v in sorted(metrics.items()):
                    print(f"    {k}: {v}")
    else:
        raise ValueError("Must provide either --kfold_dir or --train_indices + --val_indices")


if __name__ == "__main__":
    main()
