from __future__ import annotations

import argparse

from micoformer.runners.splits import make_pretrain_split


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="从 .h5ad 的 obs 中按字段取值筛选样本索引，保存为 .npy",
    )
    p.add_argument("--h5ad_path", "--h5ad", dest="h5ad_path", type=str, required=True, help=".h5ad 数据文件路径")
    p.add_argument("--field", type=str, required=True, help="obs 中用于筛选的字段名（如 Split_Group, Project_ID）")
    p.add_argument("--values", type=str, nargs="+", required=True, help="该字段中要选取的值（可多个，空格分隔）")
    p.add_argument("--output_path", "--output", dest="output_path", type=str, required=True, help="输出 .npy 文件路径")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    result = make_pretrain_split(
        h5ad_path=args.h5ad_path,
        field=args.field,
        values=args.values,
        output_path=args.output_path,
    )
    print(f"Field: {result['field']}")
    print(f"Values: {result['values']}")
    print(f"Matched: {result['matched']} / {result['total']} samples")
    print(f"Saved to {result['output_path']}")


if __name__ == "__main__":
    main()

