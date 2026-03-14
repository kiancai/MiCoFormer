"""
scripts/1.make_splits.py — 生成训练/验证/测试集的样本索引并保存为 .npz

设计原则：
  按「组」而非「样本」切分，保证同一组的所有样本只落入一个集合，
  避免同一 study 的样本同时出现在训练集和测试集（防止数据泄露）。

可用的分组字段（来自 obs metadata）：
  ─────────────────────────────────────────────────────────────────────
  Project_ID   : SRA/ENA 项目 ID，232 个唯一值（推荐：study-level OOD）
  PMID         : 发表论文 PubMed ID，381 个唯一值（比 Project_ID 更细）
  Split_Group  : 数据库预设分组（A=74557样本/320项目, B=13901/61, C=1880/14）
                 ★ 已验证无泄露：同一 Project_ID 100% 不跨 Split_Group
                 ★ 推荐直接用此字段，避免重新切分引入随机性
  ─────────────────────────────────────────────────────────────────────
  以下字段可用于「分层采样」，确保训练/验证集表型分布相似：
  Sample_Site  : 采样部位（Nasopharynx/Sputum/Nasal/... 共 20+ 种）
  Phenotype    : 表型/疾病（119 种）
  Case_Or_Control : case / control
  Is_Healthy   : True / False
  Continent    : 大陆（Asia/Europe/North America/...）
  ─────────────────────────────────────────────────────────────────────

使用示例：

  # 方案 A：直接使用数据库预设的 Split_Group（最推荐，无随机性）
  python scripts/1.make_splits.py \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --mode preset \\
      --preset_field Split_Group \\
      --preset_train A --preset_val B --preset_test C \\
      --output data/processed/splits/preset_ABC.npz

  # 方案 B：按 Project_ID 随机切分（train=80%, val=10%, test=10%）
  python scripts/1.make_splits.py \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --mode random \\
      --group_by Project_ID \\
      --val_ratio 0.1 --test_ratio 0.1 \\
      --seed 42 \\
      --output data/processed/splits/project_random.npz

  # 方案 C：按 PMID 随机切分（更细粒度的 study split）
  python scripts/1.make_splits.py \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --mode random \\
      --group_by PMID \\
      --val_ratio 0.1 --test_ratio 0.1 \\
      --seed 42 \\
      --output data/processed/splits/pmid_random.npz

训练脚本加载方式：
  python scripts/2.train_pretrain.py \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --splits data/processed/splits/preset_ABC.npz \\
      ...
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="生成 study-level 数据分割索引（train/val/test）并保存为 .npz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--h5ad", type=str, required=True, help=".h5ad 数据文件路径")
    p.add_argument(
        "--output", type=str, required=True, help="输出 .npz 文件路径（如 data/processed/splits/preset_ABC.npz）"
    )

    sub = p.add_subparsers(dest="mode", required=True)

    # ── 方案 A：直接使用已有分组字段 ──────────────────────────────────────
    preset = sub.add_parser("preset", help="直接将已有字段的某些值映射为 train/val/test")
    preset.add_argument(
        "--field",
        type=str,
        default="Split_Group",
        help="obs 中的分组字段（默认 Split_Group）",
    )
    preset.add_argument("--train", type=str, nargs="+", default=["A"], help="作为训练集的字段值（默认 A）")
    preset.add_argument("--val",   type=str, nargs="+", default=["B"], help="作为验证集的字段值（默认 B）")
    preset.add_argument("--test",  type=str, nargs="+", default=["C"], help="作为测试集的字段值（默认 C）")

    # ── 方案 B：按任意字段的唯一组随机切分 ──────────────────────────────────
    rand = sub.add_parser("random", help="按分组字段的唯一值随机切分为 train/val/test")
    rand.add_argument(
        "--group_by",
        type=str,
        default="Project_ID",
        help="用于分组的 obs 字段（默认 Project_ID）",
    )
    rand.add_argument("--val_ratio",  type=float, default=0.1,  help="验证集比例（按组数计，默认 0.1）")
    rand.add_argument("--test_ratio", type=float, default=0.1,  help="测试集比例（按组数计，默认 0.1）")
    rand.add_argument("--seed",       type=int,   default=42,   help="随机种子（默认 42）")

    return p


def _preset_split(obs, field: str, train_vals, val_vals, test_vals):
    """直接按字段值划分样本索引。"""
    all_indices = np.arange(len(obs))
    field_values = obs[field].to_numpy()

    train_mask = np.isin(field_values, train_vals)
    val_mask   = np.isin(field_values, val_vals)
    test_mask  = np.isin(field_values, test_vals)

    # 检查是否有样本未被分配
    unassigned = ~(train_mask | val_mask | test_mask)
    if unassigned.any():
        print(f"Warning: {unassigned.sum()} 个样本的 {field} 值不在指定范围内，已忽略。")

    return (
        all_indices[train_mask],
        all_indices[val_mask],
        all_indices[test_mask],
    )


def _random_group_split(obs, group_by: str, val_ratio: float, test_ratio: float, seed: int):
    """按分组字段的唯一组随机打乱后切分，保证组不跨集合。"""
    rng = np.random.default_rng(seed)

    groups = obs[group_by].to_numpy()
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n = len(unique_groups)
    n_test = max(1, int(n * test_ratio))
    n_val  = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError(
            f"val_ratio={val_ratio} + test_ratio={test_ratio} 覆盖了所有组（共 {n} 组），请降低比例。"
        )

    train_groups = set(unique_groups[:n_train])
    val_groups   = set(unique_groups[n_train: n_train + n_val])
    test_groups  = set(unique_groups[n_train + n_val:])

    all_indices = np.arange(len(obs))
    train_idx = all_indices[np.isin(groups, list(train_groups))]
    val_idx   = all_indices[np.isin(groups, list(val_groups))]
    test_idx  = all_indices[np.isin(groups, list(test_groups))]

    return train_idx, val_idx, test_idx, n_train, n_val, len(test_groups)


def main():
    args = build_argparser().parse_args()

    print(f"Reading obs from {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs.copy()
    try:
        if adata.file is not None:
            adata.file.close()
    except Exception:
        pass

    if args.mode == "preset":
        train_idx, val_idx, test_idx = _preset_split(
            obs,
            field=args.field,
            train_vals=args.train,
            val_vals=args.val,
            test_vals=args.test,
        )
        print(f"Mode: preset  |  field={args.field}")
        print(f"  train values={args.train}  val values={args.val}  test values={args.test}")

    elif args.mode == "random":
        train_idx, val_idx, test_idx, n_tr, n_v, n_te = _random_group_split(
            obs,
            group_by=args.group_by,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        total_groups = obs[args.group_by].nunique()
        print(f"Mode: random  |  group_by={args.group_by}  seed={args.seed}")
        print(f"  Groups: total={total_groups}  train={n_tr}  val={n_v}  test={n_te}")

    print(f"\nSplit stats:")
    print(f"  Train: {len(train_idx):>7} samples")
    print(f"  Val:   {len(val_idx):>7} samples")
    print(f"  Test:  {len(test_idx):>7} samples")
    print(f"  Total: {len(train_idx) + len(val_idx) + len(test_idx):>7} / {len(obs)}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    np.savez(
        args.output,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
    )
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
