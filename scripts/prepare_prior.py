#!/usr/bin/env python3
"""Prepare MiCoFormer fixed features from prior data V1.
从 V1 原始先验生成 MiCoFormer 固定特征。
"""

import argparse
from pathlib import Path

from micoformer.data.priors import build_prior

# Pin the source archive before processing. / 处理前核对指定原始先验。
PRIOR_VERSION = "V1"
PRIOR_SHA256 = "0723cfa4e16f97300a639e871c261a7388e6e1fdf0ee51e317252497bb6aecbc"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare MiCoFormer fixed features from the V1 prior archive.",
        epilog="Run from the repository root: python -m scripts.prepare_prior",
    )
    parser.add_argument(
        "--prior",
        type=Path,
        default=Path("data/reference/priors_v1.npz"),
        help="Path to the V1 prior archive (default: %(default)s). "
        "Its version and SHA256 are checked. "
        "Relative paths start from the current working directory.",
    )
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("data/prepared"),
        help="Output directory (default: %(default)s). Writes prior_assets.npz "
        "and prior_assets.json without overwriting existing files. "
        "Relative paths start from the current working directory.",
    )
    args = parser.parse_args()
    print(
        build_prior(
            prior=args.prior,
            output=args.prepared / "prior_assets.npz",
            expected_version=PRIOR_VERSION,
            expected_sha256=PRIOR_SHA256,
        )
    )


if __name__ == "__main__":
    main()
