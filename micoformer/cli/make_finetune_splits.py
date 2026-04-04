from __future__ import annotations

import argparse

from micoformer.runners.splits import (
    make_finetune_kfold_splits,
    make_finetune_ood_splits,
)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate finetune split indices (K-fold or OOD)",
    )
    p.add_argument("--h5ad_path", "--h5ad", dest="h5ad_path", type=str, required=True, help=".h5ad file path")
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


def main() -> None:
    args = build_argparser().parse_args()
    if args.mode == "kfold":
        result = make_finetune_kfold_splits(
            h5ad_path=args.h5ad_path,
            label_field=args.label_field,
            label_values=args.label_values,
            output_dir=args.output_dir,
            seed=args.seed,
            filter_field=args.filter_field,
            filter_values=args.filter_values,
            num_folds=args.num_folds,
        )
        print(f"Mode: {result['mode']}")
        print(f"Candidates after filtering: {result['num_candidates']}")
        print(f"Label distribution: {result['label_distribution']}")
        for fold_result in result["fold_sizes"]:
            print(
                f"Fold {fold_result['fold']}: "
                f"train={fold_result['train']}, val={fold_result['val']}"
            )
        print(f"Saved to {result['output_dir']}")
        return

    result = make_finetune_ood_splits(
        h5ad_path=args.h5ad_path,
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
    print(f"Mode: {result['mode']}")
    print(f"Split sizes: {result['split_sizes']}")
    print(f"Split distributions: {result['split_distributions']}")
    print(f"Saved to {result['output_dir']}")


if __name__ == "__main__":
    main()

