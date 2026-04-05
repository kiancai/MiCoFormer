"""数据切分 workflow：统一按 obs 字段筛选样本 + 三种分配策略（不含 argparse）。

核心心智模型：
  切分 = 筛选（按 obs 的若干列）+ 分配（single / kfold / ood）

三个 public 函数对应三种分配策略：
  - make_split_single : 筛选后直接作为一组（适合 pretrain 单 split 或 finetune holdout）
  - make_split_kfold  : 筛选后做 Stratified K-fold（必须提供 label）
  - make_split_ood    : 不同 subset 各自筛选一次（跨 study/跨 group 的 OOD 场景）

label_field 是可选组件：提供时会额外过滤掉标签为无效值的样本。
"""

from __future__ import annotations

import os

import anndata as ad
import numpy as np


FilterSpec = list[tuple[str, list[str]]]


# ---------- 内部辅助 ----------


def _apply_filters(
    obs,
    filters: FilterSpec,
    total: int,
) -> np.ndarray:
    """按 filters 里每个 (field, values) 逐步 AND 过滤，返回命中索引。

    空 filters 返回全部索引。
    """
    if not filters:
        return np.arange(total)

    mask = np.ones(total, dtype=bool)
    for field, values in filters:
        if field not in obs.columns:
            available = ", ".join(sorted(obs.columns))
            raise ValueError(
                f"Field '{field}' not found in obs. Available fields: {available}"
            )
        col = np.array([str(v) for v in obs[field].to_numpy()])
        mask &= np.isin(col, [str(v) for v in values])
    return np.where(mask)[0]


def _filter_by_label(
    obs,
    indices: np.ndarray,
    label_field: str,
    label_values: list[str] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """按 label_field 过滤：只保留标签在 label_values 中（或非空）的样本。

    返回 (filtered_indices, labels_for_filtered)。
    """
    if label_field not in obs.columns:
        available = ", ".join(sorted(obs.columns))
        raise ValueError(
            f"label_field '{label_field}' not found in obs. Available: {available}"
        )
    col = np.array([str(v).strip() for v in obs[label_field].to_numpy()[indices]])
    if label_values is not None:
        valid = set(str(v) for v in label_values)
        mask = np.array([v in valid for v in col])
    else:
        mask = np.array([v.lower() not in ("nan", "none", "<na>", "") for v in col])
    return indices[mask], col[mask]


def _select_samples(
    adata,
    filters: FilterSpec,
    label_field: str | None = None,
    label_values: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """统一筛选内核：先按 filters 选样本，再（可选）按 label 过滤。

    返回 (indices, labels_or_None)。
    """
    obs = adata.obs
    indices = _apply_filters(obs, filters, total=len(obs))
    if label_field is None:
        return indices, None
    return _filter_by_label(obs, indices, label_field, label_values)


def _format_filters(filters: FilterSpec) -> str:
    """打印用：把 filters 格式化为 'field1=a,b AND field2=c' 形式。"""
    if not filters:
        return "(no filter)"
    return " AND ".join(f"{f}={','.join(str(v) for v in vs)}" for f, vs in filters)


def _print_label_dist(labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")


def _open_adata(h5ad: str):
    print(f"Reading obs from {h5ad} ...")
    return ad.read_h5ad(h5ad, backed="r")


def _close_adata(adata) -> None:
    if getattr(adata, "file", None) is not None:
        adata.file.close()


def _save_indices(indices: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.save(path, indices)
    print(f"Saved {len(indices)} indices -> {path}")


# ---------- Public 策略函数 ----------


def make_split_single(
    h5ad: str,
    filters: FilterSpec,
    output: str,
    *,
    label_field: str | None = None,
    label_values: list[str] | None = None,
) -> np.ndarray:
    """按 filters（+ 可选 label 过滤）筛选样本 → 保存为单个 .npy。

    Args:
        filters: [(field, values), ...]，多个之间为 AND；可为空表示不按字段筛选。
        output: 输出 .npy 文件路径。
        label_field, label_values: 可选标签过滤；label_values=None 表示只要求非空。
    """
    adata = _open_adata(h5ad)
    try:
        indices, labels = _select_samples(adata, filters, label_field, label_values)
        n_total = len(adata.obs)
    finally:
        _close_adata(adata)

    print(f"Filters: {_format_filters(filters)}")
    if label_field is not None:
        print(f"Label filter: {label_field}={label_values}")
    print(f"Matched: {len(indices)} / {n_total} samples")
    if labels is not None and len(labels) > 0:
        _print_label_dist(labels)

    if len(indices) == 0:
        raise ValueError("No samples matched the given filters.")

    _save_indices(indices, output)
    return indices


def make_split_kfold(
    h5ad: str,
    filters: FilterSpec,
    label_field: str,
    label_values: list[str] | None,
    output_dir: str,
    *,
    num_folds: int = 5,
    seed: int = 42,
) -> None:
    """筛选 → Stratified K-fold → fold_{i}_{train,val}.npy。

    label_field 必需（K-fold 需要标签做分层）。
    """
    adata = _open_adata(h5ad)
    try:
        indices, labels = _select_samples(adata, filters, label_field, label_values)
    finally:
        _close_adata(adata)

    print(f"Filters: {_format_filters(filters)}")
    print(f"Label filter: {label_field}={label_values}")
    print(f"After filtering: {len(indices)} samples")
    if len(indices) == 0:
        raise ValueError("No valid samples after filtering.")
    _print_label_dist(labels)

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
            f"Fold {fold_i}: train={len(train_global)}, val={len(val_global)} "
            f"-> {train_path}"
        )

    print(f"Done. {num_folds} folds saved to {output_dir}")


def make_split_ood(
    h5ad: str,
    subset_filters: dict[str, FilterSpec],
    output_dir: str,
    *,
    label_field: str | None = None,
    label_values: list[str] | None = None,
) -> None:
    """按 subset_filters 里每个 subset 各自的 filters 筛选 → {subset_name}.npy。

    Args:
        subset_filters: {"train": [(field,values),...], "val": [...], "test": [...]}；
            键名任意，最终 .npy 文件按键名命名。
        label_field, label_values: 可选；若提供则对每个 subset 做相同的 label 过滤。
    """
    if not subset_filters:
        raise ValueError("subset_filters must contain at least one subset.")

    adata = _open_adata(h5ad)
    results: dict[str, np.ndarray] = {}
    try:
        for name, filters in subset_filters.items():
            if not filters:
                raise ValueError(
                    f"subset '{name}' has empty filters; provide at least one (field, values)."
                )
            indices, labels = _select_samples(adata, filters, label_field, label_values)
            results[name] = indices
            dist = (
                ", ".join(
                    f"{u}={c}"
                    for u, c in zip(*np.unique(labels, return_counts=True))
                )
                if labels is not None and len(labels) > 0
                else "no label"
            )
            print(
                f"{name}: {len(indices)} samples  "
                f"[{_format_filters(filters)}] ({dist})"
            )
    finally:
        _close_adata(adata)

    os.makedirs(output_dir, exist_ok=True)
    for name, indices in results.items():
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, indices)
        print(f"Saved {name} ({len(indices)}) -> {path}")

    print(f"Done. OOD splits saved to {output_dir}")
