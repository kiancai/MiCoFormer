from __future__ import annotations

import os

import anndata as ad
import numpy as np


def make_pretrain_split(
    *,
    h5ad_path: str,
    field: str,
    values: list[str],
    output_path: str,
) -> dict[str, object]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs
    try:
        if field not in obs.columns:
            available = ", ".join(sorted(obs.columns))
            raise ValueError(f"Field '{field}' not found in obs. Available fields: {available}")

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

        total = len(obs)
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, indices)
    return {
        "field": field,
        "values": values,
        "matched": len(indices),
        "total": total,
        "output_path": output_path,
    }


def filter_by_label(
    obs_col: np.ndarray,
    indices: np.ndarray,
    label_values: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    # 过滤有效标签样本，返回 (filtered_indices, labels_for_filtered)
    col_str = np.array([str(v).strip() for v in obs_col[indices]])
    if label_values is not None:
        valid_set = set(label_values)
        mask = np.array([value in valid_set for value in col_str])
    else:
        mask = np.array([value.lower() not in ("nan", "none", "<na>", "") for value in col_str])

    filtered = indices[mask]
    labels = col_str[mask]
    return filtered, labels


def stratified_kfold_indices(
    labels: np.ndarray,
    *,
    num_folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    # 纯 numpy 实现的分层 K-fold，避免额外依赖 sklearn
    if num_folds < 2:
        raise ValueError("--num_folds must be >= 2.")

    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    fold_buckets: list[list[int]] = [[] for _ in range(num_folds)]

    for label in np.unique(labels):
        class_indices = np.where(labels == label)[0]
        rng.shuffle(class_indices)
        class_chunks = np.array_split(class_indices, num_folds)
        for fold_i, chunk in enumerate(class_chunks):
            fold_buckets[fold_i].extend(chunk.tolist())

    splits: list[tuple[np.ndarray, np.ndarray]] = []
    all_indices = np.arange(len(labels))
    for fold_i in range(num_folds):
        val_idx = np.array(sorted(fold_buckets[fold_i]), dtype=np.int64)
        val_mask = np.zeros(len(labels), dtype=bool)
        val_mask[val_idx] = True
        train_idx = all_indices[~val_mask]
        splits.append((train_idx, val_idx))
    return splits


def make_finetune_kfold_splits(
    *,
    h5ad_path: str,
    label_field: str,
    label_values: list[str] | None,
    output_dir: str,
    seed: int,
    filter_field: str | None = None,
    filter_values: list[str] | None = None,
    num_folds: int = 5,
) -> dict[str, object]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs
        n_total = len(obs)

        if filter_field is not None:
            if filter_field not in obs.columns:
                raise ValueError(f"filter_field '{filter_field}' not in obs")
            field_vals = obs[filter_field].to_numpy()
            candidate_mask = np.isin(np.array([str(v) for v in field_vals]), filter_values)
            candidate_indices = np.where(candidate_mask)[0]
        else:
            candidate_indices = np.arange(n_total)

        label_col = obs[label_field].to_numpy()
        indices, labels = filter_by_label(label_col, candidate_indices, label_values)
        if len(indices) == 0:
            raise ValueError("No valid samples after filtering.")

        unique, counts = np.unique(labels, return_counts=True)
        label_distribution = {str(label): int(count) for label, count in zip(unique, counts)}
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    os.makedirs(output_dir, exist_ok=True)

    fold_sizes = []
    for fold_i, (train_idx, val_idx) in enumerate(
        stratified_kfold_indices(labels, num_folds=num_folds, seed=seed)
    ):
        train_global = indices[train_idx]
        val_global = indices[val_idx]
        np.save(os.path.join(output_dir, f"fold_{fold_i}_train.npy"), train_global)
        np.save(os.path.join(output_dir, f"fold_{fold_i}_val.npy"), val_global)
        fold_sizes.append(
            {
                "fold": fold_i,
                "train": int(len(train_global)),
                "val": int(len(val_global)),
            }
        )

    return {
        "mode": "kfold",
        "output_dir": output_dir,
        "num_folds": num_folds,
        "num_candidates": int(len(indices)),
        "label_distribution": label_distribution,
        "fold_sizes": fold_sizes,
    }


def make_finetune_ood_splits(
    *,
    h5ad_path: str,
    label_field: str,
    label_values: list[str] | None,
    output_dir: str,
    train_field: str,
    train_values: list[str],
    val_field: str,
    val_values: list[str],
    test_field: str | None = None,
    test_values: list[str] | None = None,
) -> dict[str, object]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs
        label_col = obs[label_field].to_numpy()

        split_configs = [
            ("train", train_field, train_values),
            ("val", val_field, val_values),
            ("test", test_field, test_values),
        ]

        split_sizes: dict[str, int] = {}
        split_distributions: dict[str, dict[str, int]] = {}
        split_arrays: dict[str, np.ndarray] = {}

        for split_name, field, values in split_configs:
            if field is None or values is None:
                if split_name in ("train", "val"):
                    raise ValueError(f"--{split_name}_field and --{split_name}_values are required for OOD mode")
                continue

            if field not in obs.columns:
                raise ValueError(f"{split_name}_field '{field}' not in obs")

            field_vals = obs[field].to_numpy()
            candidate_mask = np.isin(np.array([str(v) for v in field_vals]), values)
            candidate_indices = np.where(candidate_mask)[0]
            filtered, labels = filter_by_label(label_col, candidate_indices, label_values)
            split_arrays[split_name] = filtered
            split_sizes[split_name] = int(len(filtered))

            unique, counts = np.unique(labels, return_counts=True)
            split_distributions[split_name] = {
                str(label): int(count) for label, count in zip(unique, counts)
            }
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    os.makedirs(output_dir, exist_ok=True)
    for split_name, indices in split_arrays.items():
        np.save(os.path.join(output_dir, f"{split_name}.npy"), indices)

    return {
        "mode": "ood",
        "output_dir": output_dir,
        "split_sizes": split_sizes,
        "split_distributions": split_distributions,
    }
