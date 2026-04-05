"""数据分割 workflow：预训练与微调的索引分割逻辑（不含 argparse）"""

from __future__ import annotations

import os

import anndata as ad
import numpy as np


def make_pretrain_split(
    h5ad: str,
    field: str,
    values: list[str],
    output: str,
) -> np.ndarray:
    """从 .h5ad 的 obs 中按字段取值筛选样本索引，保存为 .npy"""
    print(f"Reading obs from {h5ad} ...")
    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        obs = adata.obs

        if field not in obs.columns:
            available = ", ".join(sorted(obs.columns))
            raise ValueError(
                f"Field '{field}' not found in obs. Available fields: {available}"
            )

        field_values = obs[field].to_numpy()
        mask = np.isin(field_values, values)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            unique_vals = sorted(set(str(v) for v in np.unique(field_values)))
            raise ValueError(
                f"No samples matched field='{field}' values={values}. "
                f"Unique values in this field: "
                f"{unique_vals[:20]}{'...' if len(unique_vals) > 20 else ''}"
            )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    print(f"Field: {field}")
    print(f"Values: {values}")
    print(f"Matched: {len(indices)} / {len(obs)} samples")

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.save(output, indices)
    print(f"Saved to {output}")
    return indices


def _filter_by_label(
    obs_col: np.ndarray,
    indices: np.ndarray,
    label_values: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """过滤有效标签样本，返回 (filtered_indices, labels_for_filtered)"""
    col_str = np.array([str(v).strip() for v in obs_col[indices]])

    if label_values is not None:
        valid_set = set(label_values)
        mask = np.array([v in valid_set for v in col_str])
    else:
        mask = np.array([v.lower() not in ("nan", "none", "<na>", "") for v in col_str])

    filtered = indices[mask]
    labels = col_str[mask]
    return filtered, labels


def make_finetune_kfold(
    h5ad: str,
    label_field: str,
    label_values: list[str] | None,
    output_dir: str,
    *,
    filter_field: str | None = None,
    filter_values: list[str] | None = None,
    num_folds: int = 5,
    seed: int = 42,
) -> None:
    """生成 Stratified K-fold 分割索引"""
    print(f"Reading obs from {h5ad} ...")
    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        obs = adata.obs
        n_total = len(obs)

        # Step 1: 按 filter_field 筛选样本
        if filter_field is not None:
            if filter_field not in obs.columns:
                raise ValueError(f"filter_field '{filter_field}' not in obs")
            field_vals = obs[filter_field].to_numpy()
            candidate_mask = np.isin(
                np.array([str(v) for v in field_vals]), filter_values
            )
            candidate_indices = np.where(candidate_mask)[0]
            print(
                f"Filtered by {filter_field}={filter_values}: "
                f"{len(candidate_indices)} / {n_total}"
            )
        else:
            candidate_indices = np.arange(n_total)

        # Step 2: 按 label_field 筛选有效标签
        label_col = obs[label_field].to_numpy()
        indices, labels = _filter_by_label(label_col, candidate_indices, label_values)
        print(f"After label filtering ({label_field}): {len(indices)} samples")

        if len(indices) == 0:
            raise ValueError("No valid samples after filtering.")

        unique, counts = np.unique(labels, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    # Step 3: Stratified K-fold
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    os.makedirs(output_dir, exist_ok=True)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        train_global = indices[train_idx]
        val_global = indices[val_idx]

        train_path = os.path.join(output_dir, f"fold_{fold_i}_train.npy")
        val_path = os.path.join(output_dir, f"fold_{fold_i}_val.npy")
        np.save(train_path, train_global)
        np.save(val_path, val_global)
        print(
            f"Fold {fold_i}: train={len(train_global)}, "
            f"val={len(val_global)} -> {train_path}"
        )

    print(f"Done. {num_folds} folds saved to {output_dir}")


def make_finetune_ood(
    h5ad: str,
    label_field: str,
    label_values: list[str] | None,
    output_dir: str,
    *,
    train_field: str,
    train_values: list[str],
    val_field: str,
    val_values: list[str],
    test_field: str | None = None,
    test_values: list[str] | None = None,
) -> None:
    """生成 OOD（跨 study）分割索引"""
    print(f"Reading obs from {h5ad} ...")
    adata = ad.read_h5ad(h5ad, backed="r")
    try:
        obs = adata.obs
        label_col = obs[label_field].to_numpy()

        splits: dict[str, np.ndarray] = {}
        for split_name, field, values in [
            ("train", train_field, train_values),
            ("val", val_field, val_values),
            ("test", test_field, test_values),
        ]:
            if field is None or values is None:
                if split_name in ("train", "val"):
                    raise ValueError(
                        f"{split_name}_field and {split_name}_values "
                        f"are required for OOD mode"
                    )
                continue

            if field not in obs.columns:
                raise ValueError(f"{split_name}_field '{field}' not in obs")

            field_vals = obs[field].to_numpy()
            candidate_mask = np.isin(
                np.array([str(v) for v in field_vals]), values
            )
            candidate_indices = np.where(candidate_mask)[0]

            filtered, filt_labels = _filter_by_label(
                label_col, candidate_indices, label_values
            )
            splits[split_name] = filtered

            unique, counts = np.unique(filt_labels, return_counts=True)
            dist_str = ", ".join(f"{u}={c}" for u, c in zip(unique, counts))
            print(f"{split_name}: {len(filtered)} samples ({dist_str})")
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    os.makedirs(output_dir, exist_ok=True)
    for split_name, split_indices in splits.items():
        path = os.path.join(output_dir, f"{split_name}.npy")
        np.save(path, split_indices)
        print(f"Saved {split_name} ({len(split_indices)}) -> {path}")

    print(f"Done. OOD splits saved to {output_dir}")
