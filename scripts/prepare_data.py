#!/usr/bin/env python3
"""Prepare MiCoFormer pretraining splits from MCFCorpusV3.1.

Select analysis_primary rows, then save train/validation indices and provenance.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np

from micoformer.data.preparation import prepare_splits

# Pin the corpus file before selection. / 选样前核对指定语料文件。
CORPUS_VERSION = "MCFCorpusV3.1"
CORPUS_SHA256 = "ab41012e7b3a04cbdee21b182abcee7363b8084a59f815e0efecdccc005ae6c3"

# Random row split, not grouped by Study. / 按样本行随机划分，不按 Study 分组。
SEED = 42
VALIDATION_FRACTION = 0.005


def select_rows(adata: ad.AnnData) -> np.ndarray:
    """Return selected original row indices. / 返回入选样本的原始行号。"""
    obs = adata.obs
    # V3.1 already defines QC-passing representatives. / V3.1 已确定通过 QC 的代表行。
    selected = obs["analysis_primary"].to_numpy(dtype=bool, copy=True)

    return np.flatnonzero(selected).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select MCFCorpusV3.1 analysis_primary samples and create "
        f"random training/validation splits ({VALIDATION_FRACTION:.1%} validation, "
        f"seed {SEED}).",
        epilog="Example: python scripts/prepare_data.py "
        "--corpus /path/to/MCFCorpusV3.1.gg2.h5ad --output data/prepared",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Required path to the MCFCorpusV3.1 H5AD file; "
        "its version and SHA256 are checked.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prepared"),
        help="New directory for sample indices and provenance (default: %(default)s). "
        "Must not already exist. Relative paths start from the current working directory.",
    )
    args = parser.parse_args()

    print(
        prepare_splits(
            corpus=args.corpus,
            output=args.output,
            select_rows=select_rows,
            expected_version=CORPUS_VERSION,
            expected_sha256=CORPUS_SHA256,
            seed=SEED,
            validation_fraction=VALIDATION_FRACTION,
        )
    )


if __name__ == "__main__":
    main()
