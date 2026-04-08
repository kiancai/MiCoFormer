"""Export one sample-index `.npy` file from `.h5ad.obs`.

Examples:
  python scripts/1.make_splits.py --h5ad <input.h5ad> --output <output.npy>
  python scripts/1.make_splits.py --h5ad <input.h5ad> --filters "Split_Group=A" --output <output.npy>
  python scripts/1.make_splits.py --h5ad <input.h5ad> --filters "Project_ID=PRJ..." "Phenotype=Health,Disease" --output <output.npy>

Filter syntax:
  - Each token looks like: `field=v1,v2`
  - Multiple values inside one token are OR
  - Multiple tokens are ANDed together
"""

from __future__ import annotations
import argparse
from micoformer.workflows.splits import make_split


TAG = "[make_splits]"


# 将命令行 filter 字符串解析成内部使用的 `(field, values)` 结构
def _parse_filter_token(token: str) -> tuple[str, list[str]]:

    # 要求 token 至少包含一个 `=`，
    if "=" not in token:
        raise argparse.ArgumentTypeError(
            f"Filter must be in 'field=v1,v2' form, got: {token!r}"
        )

    field, values_str = token.split("=", 1) # 只用`=`切一次，得到字段名和允许值
    field = field.strip()                   # 去掉字段名前后的空格
    # 按逗号分割允许值 list，并再一次去掉每个值前后的空白
    values = [v.strip() for v in values_str.split(",") if v.strip()]

    # 二次检查，字段名不能为空；允许值列表不能为空
    if not field or not values:
        raise argparse.ArgumentTypeError(
            f"Empty field or values in filter: {token!r}"
        )

    return field, values


def build_argparser() -> argparse.ArgumentParser:

    p = argparse.ArgumentParser(
        description="Export one sample subset from h5ad.obs into a single .npy file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,  # 模块 docstring 放到帮助信息末尾
    )

    p.add_argument("--h5ad", type=str, required=True, help=".h5ad input path")
    p.add_argument(
        "--filters", nargs="*", default=None,
        help="Filter specs like 'field=v1,v2'. Multiple tokens are ANDed together.",
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Output .npy path for the selected sample indices.",
    )

    return p


def main() -> None:
    # 1.解析命令行
    args = build_argparser().parse_args()
    filters = []
    if args.filters:
        filters = [_parse_filter_token(token) for token in args.filters]

    # 2.调用 make_split(...) 输出索引文件
    print(f"{TAG} h5ad={args.h5ad}")
    print(f"{TAG} output={args.output}")
    print(f"{TAG} filters={args.filters if args.filters else '(no filter)'}")
    make_split(
        h5ad=args.h5ad,
        filters=filters,
        output=args.output,
    )


if __name__ == "__main__":
    main()
