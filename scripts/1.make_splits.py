"""
scripts/1.make_splits.py — 从 .h5ad 的 obs 中按字段取值筛选样本索引，保存为 .npy

一次只生成一组索引。需要 train/val/test 多组时，分别运行多次。

使用示例：

  # 选出 Split_Group == A 的所有样本作为训练集
  python scripts/1.make_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --field Split_Group --values A \
      --output data/processed/splits/train.npy

  # 选出 Split_Group == B 的所有样本作为验证集
  python scripts/1.make_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --field Split_Group --values B \
      --output data/processed/splits/val.npy

  # 多个取值合并：Split_Group 为 B 或 C 的都归为一组
  python scripts/1.make_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --field Split_Group --values B C \
      --output data/processed/splits/val_bc.npy

可用的 obs 字段（参考）：
  Split_Group  : 数据库预设分组（A/B/C）
  Project_ID   : SRA/ENA 项目 ID（232 个唯一值）
  PMID         : PubMed 论文 ID（381 个唯一值）
  Sample_Site  : 采样部位
  Phenotype    : 表型/疾病
  Continent    : 大陆

参数说明：
  --h5ad    : 输入 .h5ad 文件路径
  --field   : obs 中用于筛选的字段名
  --values  : 该字段中要选取的一个或多个值（空格分隔）
  --output  : 输出 .npy 文件路径（保存匹配样本的整数索引数组）
"""

from __future__ import annotations

import argparse
import os

import anndata as ad
import numpy as np


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从 .h5ad 的 obs 中按字段取值筛选样本索引，保存为 .npy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--h5ad", type=str, required=True, help=".h5ad 数据文件路径")
    p.add_argument("--field", type=str, required=True, help="obs 中用于筛选的字段名（如 Split_Group, Project_ID）")
    p.add_argument("--values", type=str, nargs="+", required=True, help="该字段中要选取的值（可多个，空格分隔）")
    p.add_argument("--output", type=str, required=True, help="输出 .npy 文件路径")
    return p


def main():
    args = build_argparser().parse_args()

    print(f"Reading obs from {args.h5ad} ...")
    adata = ad.read_h5ad(args.h5ad, backed="r")
    obs = adata.obs
    try:
        # 检查字段是否存在
        if args.field not in obs.columns:
            available = ", ".join(sorted(obs.columns))
            raise ValueError(f"Field '{args.field}' not found in obs. Available fields: {available}")

        field_values = obs[args.field].to_numpy()
        mask = np.isin(field_values, args.values)
        indices = np.where(mask)[0]

        # 检查是否有匹配
        if len(indices) == 0:
            unique_vals = sorted(set(str(v) for v in np.unique(field_values)))
            raise ValueError(
                f"No samples matched field='{args.field}' values={args.values}. "
                f"Unique values in this field: {unique_vals[:20]}{'...' if len(unique_vals) > 20 else ''}"
            )
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    print(f"Field: {args.field}")
    print(f"Values: {args.values}")
    print(f"Matched: {len(indices)} / {len(obs)} samples")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    np.save(args.output, indices)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
