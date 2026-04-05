"""
scripts/1.make_splits.py — 统一的数据切分 CLI

三种分配策略（--strategy）：
  - single : 按 filters 筛选 → 单个 .npy（pretrain 单 split / finetune holdout 均可）
  - kfold  : 筛选 → Stratified K-fold → fold_{i}_{train,val}.npy（必须提供 label）
  - ood    : 多个 subset 各自筛选 → {subset_name}.npy（跨 study/跨 group）

筛选语法：
  --filters "field=v1,v2" "field2=v3"    # 多个之间 AND；single/kfold 用
  --subset-filters train:"field=A" val:"field=B" test:"field=C"  # ood 专用

使用示例：

  # 1) pretrain 单 split：选 Split_Group=A 做训练集
  python scripts/1.make_splits.py --strategy single \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --filters "Split_Group=A" \\
      --output data/processed/splits/pretrain_train.npy

  # 2) finetune 单 study 内 5-fold
  python scripts/1.make_splits.py --strategy kfold \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --filters "Project_ID=PRJNA123456" \\
      --label-field Phenotype --label-values Health Disease \\
      --num-folds 5 --seed 42 \\
      --output data/processed/splits/finetune/kfold_PRJNA123456/

  # 3) OOD 跨 group 评估
  python scripts/1.make_splits.py --strategy ood \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --subset-filters train:"Split_Group=A" val:"Split_Group=B" test:"Split_Group=C" \\
      --label-field Phenotype --label-values Health Disease \\
      --output data/processed/splits/finetune/ood_ABC/

  # 4) finetune holdout（新增能力）：单次 train/val 切分
  python scripts/1.make_splits.py --strategy single \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --filters "Project_ID=PRJNA123456" \\
      --label-field Phenotype --label-values Health Disease \\
      --output data/processed/splits/finetune/holdout_train.npy

  # 5) pretrain 时加 label 过滤（新增能力）：只用健康样本做预训练
  python scripts/1.make_splits.py --strategy single \\
      --h5ad data/processed/microbiome_dataset.h5ad \\
      --filters "Split_Group=A" \\
      --label-field Phenotype --label-values Health \\
      --output data/processed/splits/pretrain_healthy.npy
"""

from __future__ import annotations

import argparse

from micoformer.workflows.splits import (
    FilterSpec,
    make_split_kfold,
    make_split_ood,
    make_split_single,
)


def _parse_filter_token(token: str) -> tuple[str, list[str]]:
    """解析 'field=v1,v2' → ('field', ['v1', 'v2'])"""
    if "=" not in token:
        raise argparse.ArgumentTypeError(
            f"Filter must be in 'field=v1,v2' form, got: {token!r}"
        )
    field, values_str = token.split("=", 1)
    field = field.strip()
    values = [v.strip() for v in values_str.split(",") if v.strip()]
    if not field or not values:
        raise argparse.ArgumentTypeError(
            f"Empty field or values in filter: {token!r}"
        )
    return field, values


def _parse_filters(tokens: list[str] | None) -> FilterSpec:
    if not tokens:
        return []
    return [_parse_filter_token(t) for t in tokens]


def _parse_subset_filter_token(token: str) -> tuple[str, tuple[str, list[str]]]:
    """解析 'subset_name:field=v1,v2' → ('subset_name', ('field', ['v1','v2']))"""
    if ":" not in token:
        raise argparse.ArgumentTypeError(
            f"Subset filter must be in 'name:field=v1,v2' form, got: {token!r}"
        )
    name, rest = token.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"Empty subset name in: {token!r}")
    return name, _parse_filter_token(rest)


def _parse_subset_filters(tokens: list[str]) -> dict[str, FilterSpec]:
    result: dict[str, FilterSpec] = {}
    for t in tokens:
        name, spec = _parse_subset_filter_token(t)
        result.setdefault(name, []).append(spec)
    return result


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified data split CLI (single / kfold / ood)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--h5ad", type=str, required=True, help=".h5ad input path")
    p.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=["single", "kfold", "ood"],
        help="Allocation strategy",
    )

    # single / kfold 的筛选条件
    p.add_argument(
        "--filters",
        nargs="*",
        default=None,
        help="Filter specs like 'field=v1,v2' (multiple = AND). Used by single/kfold.",
    )

    # ood 的 subset 筛选
    p.add_argument(
        "--subset-filters",
        nargs="*",
        default=None,
        help="Subset filter specs like 'name:field=v1,v2' (multiple same name = AND). OOD only.",
    )

    # label 过滤（可选）
    p.add_argument("--label-field", type=str, default=None, help="obs label field (optional for single/ood; required for kfold)")
    p.add_argument("--label-values", type=str, nargs="+", default=None, help="Valid label values (others excluded); omit to keep all non-empty")

    # kfold 专用
    p.add_argument("--num-folds", type=int, default=5, help="[kfold] number of folds")
    p.add_argument("--seed", type=int, default=42, help="[kfold] random seed")

    # 输出
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path: file for single, directory for kfold/ood",
    )

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.strategy == "single":
        make_split_single(
            h5ad=args.h5ad,
            filters=_parse_filters(args.filters),
            output=args.output,
            label_field=args.label_field,
            label_values=args.label_values,
        )
    elif args.strategy == "kfold":
        if args.label_field is None:
            raise SystemExit("--label-field is required for kfold strategy.")
        make_split_kfold(
            h5ad=args.h5ad,
            filters=_parse_filters(args.filters),
            label_field=args.label_field,
            label_values=args.label_values,
            output_dir=args.output,
            num_folds=args.num_folds,
            seed=args.seed,
        )
    elif args.strategy == "ood":
        if not args.subset_filters:
            raise SystemExit("--subset-filters is required for ood strategy.")
        make_split_ood(
            h5ad=args.h5ad,
            subset_filters=_parse_subset_filters(args.subset_filters),
            output_dir=args.output,
            label_field=args.label_field,
            label_values=args.label_values,
        )


if __name__ == "__main__":
    main()
