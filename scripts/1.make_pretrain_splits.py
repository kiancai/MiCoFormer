"""
scripts/1.make_pretrain_splits.py — 从 .h5ad 的 obs 中按字段取值筛选样本索引，保存为 .npy

一次只生成一组索引。需要 train/val/test 多组时，分别运行多次。

使用示例：

  # 选出 Split_Group == A 的所有样本作为训练集
  python scripts/1.make_pretrain_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --field Split_Group --values A \
      --output data/processed/splits/train.npy

  # 选出 Split_Group == B 的所有样本作为验证集
  python scripts/1.make_pretrain_splits.py \
      --h5ad data/processed/microbiome_dataset.h5ad \
      --field Split_Group --values B \
      --output data/processed/splits/val.npy

  # 多个取值合并：Split_Group 为 B 或 C 的都归为一组
  python scripts/1.make_pretrain_splits.py \
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

from micoformer.workflows.splits import make_pretrain_split


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从 .h5ad 的 obs 中按字段取值筛选样本索引，保存为 .npy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--h5ad", type=str, required=True, help=".h5ad 数据文件路径")
    p.add_argument("--field", type=str, required=True, help="obs 中用于筛选的字段名")
    p.add_argument("--values", type=str, nargs="+", required=True, help="该字段中要选取的值")
    p.add_argument("--output", type=str, required=True, help="输出 .npy 文件路径")
    return p


def main():
    args = build_argparser().parse_args()
    make_pretrain_split(
        h5ad=args.h5ad,
        field=args.field,
        values=args.values,
        output=args.output,
    )


if __name__ == "__main__":
    main()
