#!/usr/bin/env python3
"""Prepare MiCoFormer pretraining splits from MCFCorpusV4.3.

Select analysis_primary rows, then save train/validation indices and provenance.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from micoformer.data.preparation import prepare_splits

# Pin the corpus file before selection. / 选样前核对指定语料文件。
CORPUS_VERSION = "MCFCorpusV4.3"
CORPUS_SHA256 = "4227daacc976ba1361361fa35080781462da657b070640aa20f7617b56907405"

# Random row split, not grouped by Study. / 按样本行随机划分，不按 Study 分组。
SEED = 42
VALIDATION_FRACTION = 0.005


def select_rows(source: h5py.File) -> np.ndarray:
    """Return selected original row indices. / 返回入选样本的原始行号。"""
    # V4.3 retains V4.2's QC-passing representatives. / V4.3沿用V4.2的QC代表行。
    selected = source["obs/analysis_primary"][:].astype(bool)

    return np.flatnonzero(selected).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select MCFCorpusV4.3 analysis_primary samples and create "
        f"random training/validation splits ({VALIDATION_FRACTION:.1%} validation, "
        f"seed {SEED}).",
        epilog="Run from the repository root: python -m scripts.prepare_data",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/corpus/MCFCorpusV4.3.gg2.h5ad"),
        help="Path to the MCFCorpusV4.3 H5AD file (default: %(default)s). "
        "Its version and SHA256 are checked. "
        "Relative paths start from the current working directory.",
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
