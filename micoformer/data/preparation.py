"""Prepare sample splits and record their source corpus.
生成样本划分并记录来源语料。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np

from micoformer.utils.io import sha256_file


def prepare_splits(
    *,
    corpus: Path,
    output: Path,
    select_rows: Callable,
    expected_version: str,
    expected_sha256: str,
    seed: int,
    validation_fraction: float,
) -> Path:
    """Select rows, split train/validation, and save indices to a new directory.
    选取样本行，划分训练/验证集，并将行号保存到新目录。

    Returns the preparation.json path. / 返回 preparation.json 路径。
    """
    corpus, output = Path(corpus).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"choose a new preparation directory: {output}")

    adata = ad.read_h5ad(corpus, backed="r")
    try:
        # Check the corpus before selection. / 选样前核对语料身份。
        declared = adata.uns["corpus_version"]
        if declared != expected_version:
            raise ValueError(f"expected {expected_version}, found {declared}")
        digest = sha256_file(corpus)
        if digest != expected_sha256:
            raise ValueError("corpus SHA256 mismatch")
        identity = {
            "path": str(corpus),
            "sha256": digest,
            "version": declared,
            "shape": list(adata.shape),
        }
        rows = np.sort(select_rows(adata))
        vocabulary = np.asarray(adata.var_names, dtype=str)
    finally:
        adata.file.close()

    # Keep the existing random row split; no Study grouping. / 沿用按行随机划分，不按 Study 分组。
    permutation = np.random.default_rng(seed).permutation(rows.size)
    n_val = int(round(rows.size * validation_fraction))
    if not 0 < n_val < rows.size:
        raise ValueError("selection produces an empty training or validation set")
    arrays = {
        "all": rows,
        "train": np.sort(rows[permutation[n_val:]]),
        "val": np.sort(rows[permutation[:n_val]]),
    }

    output.mkdir(parents=True)
    files = {}
    for name, values in arrays.items():
        path = output / f"{name}_rows.npy"
        np.save(path, values, allow_pickle=False)
        files[name] = {
            "file": path.name,
            "count": int(values.size),
            "sha256": sha256_file(path),
        }
    np.save(output / "vocabulary.npy", vocabulary, allow_pickle=False)

    manifest = {
        "corpus": identity,
        "splits": files,
        "split_method": "sample-level random permutation; not Study-held-out",
        "seed": seed,
        "validation_fraction": validation_fraction,
    }
    path = output / "preparation.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
