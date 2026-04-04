"""
scripts/4.make_finetune_splits.py — 生成下游分类任务的分割索引

两种模式：
  - kfold: 单 study 内 Stratified K-fold
  - ood:   跨 study OOD 分割

使用示例：

  # K-fold（单 study 内 5 折交叉验证）
  python scripts/4.make_finetune_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --mode kfold \
      --filter_field Project_ID --filter_values PRJNA123456 \
      --label_field Phenotype --label_values Health Disease \
      --num_folds 5 --seed 42 \
      --output_dir data/processed/splits/finetune/kfold_PRJNA123456/

  # OOD（跨 study 泛化评估）
  python scripts/4.make_finetune_splits.py \
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
import os

import anndata as ad
import numpy as np
from sklearn.model_selection import StratifiedKFold


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


def _filter_by_label(
    obs_col: np.ndarray,
    indices: np.ndarray,
    label_values: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """过滤有效标签样本，返回 (filtered_indices, labels_for_filtered)。"""
    col_str = np.array([str(v).strip() for v in obs_col[indices]])

    if label_values is not None:
        valid_set = set(label_values)
        mask = np.array([v in valid_set for v in col_str])
    else:
        # 排除 NaN/None 等
        mask = np.array([v.lower() not in ("nan", "none", "<na>", "") for v in col_str])

    filtered = indices[mask]
    labels = col_str[mask]
    return filtered, labels


def run_kfold(args) -> None:
    print(f"Reading obs from {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    try:
        obs = adata.obs
        n_total = len(obs)

        # Step 1: 按 filter_field 筛选样本
        if args.filter_field is not None:
            if args.filter_field not in obs.columns:
                raise ValueError(f"filter_field '{args.filter_field}' not in obs")
            field_vals = obs[args.filter_field].to_numpy()
            candidate_mask = np.isin(np.array([str(v) for v in field_vals]), args.filter_values)
            candidate_indices = np.where(candidate_mask)[0]
            print(f"Filtered by {args.filter_field}={args.filter_values}: {len(candidate_indices)} / {n_total}")
        else:
            candidate_indices = np.arange(n_total)

        # Step 2: 按 label_field 筛选有效标签
        label_col = obs[args.label_field].to_numpy()
        indices, labels = _filter_by_label(label_col, candidate_indices, args.label_values)
        print(f"After label filtering ({args.label_field}): {len(indices)} samples")

        if len(indices) == 0:
            raise ValueError("No valid samples after filtering.")

        # 打印类别分布
        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    # Step 3: Stratified K-fold
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        train_global = indices[train_idx]
        val_global = indices[val_idx]

        train_path = os.path.join(args.output_dir, f"fold_{fold_i}_train.npy")
        val_path = os.path.join(args.output_dir, f"fold_{fold_i}_val.npy")
        np.save(train_path, train_global)
        np.save(val_path, val_global)
        print(f"Fold {fold_i}: train={len(train_global)}, val={len(val_global)} -> {train_path}")

    print(f"Done. {args.num_folds} folds saved to {args.output_dir}")


def run_ood(args) -> None:
    print(f"Reading obs from {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    try:
        obs = adata.obs
        label_col = obs[args.label_field].to_numpy()

        splits = {}
        for split_name, field, values in [
            ("train", args.train_field, args.train_values),
            ("val", args.val_field, args.val_values),
            ("test", args.test_field, args.test_values),
        ]:
            if field is None or values is None:
                if split_name in ("train", "val"):
                    raise ValueError(f"--{split_name}_field and --{split_name}_values are required for OOD mode")
                continue

            if field not in obs.columns:
                raise ValueError(f"{split_name}_field '{field}' not in obs")

            field_vals = obs[field].to_numpy()
            candidate_mask = np.isin(np.array([str(v) for v in field_vals]), values)
            candidate_indices = np.where(candidate_mask)[0]

            # 过滤有效标签
            filtered, labels = _filter_by_label(label_col, candidate_indices, args.label_values)
            splits[split_name] = filtered

            unique, counts = np.unique(labels, return_counts=True)
            dist_str = ", ".join(f"{u}={c}" for u, c in zip(unique, counts))
            print(f"{split_name}: {len(filtered)} samples ({dist_str})")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    os.makedirs(args.output_dir, exist_ok=True)
    for split_name, indices in splits.items():
        path = os.path.join(args.output_dir, f"{split_name}.npy")
        np.save(path, indices)
        print(f"Saved {split_name} ({len(indices)}) -> {path}")

    print(f"Done. OOD splits saved to {args.output_dir}")


def main():
    args = build_argparser().parse_args()

    if args.mode == "kfold":
        run_kfold(args)
    elif args.mode == "ood":
        run_ood(args)


if __name__ == "__main__":
    main()
