"""
scripts/3.make_finetune_splits.py — 生成下游分类任务的分割索引

两种模式：
  - kfold: 单 study 内 Stratified K-fold
  - ood:   跨 study OOD 分割

使用示例：

  # K-fold（单 study 内 5 折交叉验证）
  python scripts/3.make_finetune_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --mode kfold \
      --filter_field Project_ID --filter_values PRJNA123456 \
      --label_field Phenotype --label_values Health Disease \
      --num_folds 5 --seed 42 \
      --output_dir data/processed/splits/finetune/kfold_PRJNA123456/

  # OOD（跨 study 泛化评估）
  python scripts/3.make_finetune_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --mode ood \
      --train_field Split_Group --train_values A \
      --val_field Split_Group --val_values B \
      --test_field Split_Group --test_values C \
      --label_field Phenotype --label_values Health Disease \
      --output_dir data/processed/splits/finetune/ood_ABC/
"""

from __future__ import annotations

import argparse

from micoformer.workflows.splits import make_finetune_kfold, make_finetune_ood


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate finetune split indices (K-fold or OOD)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--h5ad", type=str, required=True, help=".h5ad file path")
    p.add_argument("--mode", type=str, required=True, choices=["kfold", "ood"], help="Split mode")

    # 标签筛选（两种模式都需要）
    p.add_argument("--label_field", type=str, required=True, help="obs label field for classification")
    p.add_argument("--label_values", type=str, nargs="+", default=None, help="Valid label values (others excluded)")

    # K-fold 参数
    p.add_argument("--filter_field", type=str, default=None, help="[kfold] obs field to filter samples")
    p.add_argument("--filter_values", type=str, nargs="+", default=None, help="[kfold] values to keep")
    p.add_argument("--num_folds", type=int, default=5, help="[kfold] number of folds")

    # OOD 参数
    p.add_argument("--train_field", type=str, default=None, help="[ood] obs field for train split")
    p.add_argument("--train_values", type=str, nargs="+", default=None, help="[ood] values for train")
    p.add_argument("--val_field", type=str, default=None, help="[ood] obs field for val split")
    p.add_argument("--val_values", type=str, nargs="+", default=None, help="[ood] values for val")
    p.add_argument("--test_field", type=str, default=None, help="[ood] obs field for test split")
    p.add_argument("--test_values", type=str, nargs="+", default=None, help="[ood] values for test")

    # 共用
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for .npy files")

    return p


def main():
    args = build_argparser().parse_args()

    if args.mode == "kfold":
        make_finetune_kfold(
            h5ad=args.h5ad,
            label_field=args.label_field,
            label_values=args.label_values,
            output_dir=args.output_dir,
            filter_field=args.filter_field,
            filter_values=args.filter_values,
            num_folds=args.num_folds,
            seed=args.seed,
        )
    elif args.mode == "ood":
        make_finetune_ood(
            h5ad=args.h5ad,
            label_field=args.label_field,
            label_values=args.label_values,
            output_dir=args.output_dir,
            train_field=args.train_field,
            train_values=args.train_values,
            val_field=args.val_field,
            val_values=args.val_values,
            test_field=args.test_field,
            test_values=args.test_values,
        )


if __name__ == "__main__":
    main()
