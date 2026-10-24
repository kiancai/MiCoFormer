"""Read embedded tree assets and construct full-composition branch masses.
从语料内读取树资产，构造完整组成的树枝质量。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from anndata.io import read_elem
import h5py
from scipy import sparse
import torch


@dataclass(frozen=True)
class UniFracAssets:
    incidence: sparse.csr_matrix
    branch_lengths: torch.Tensor
    identity: dict[str, Any]


def load_unifrac_assets(corpus: Path) -> UniFracAssets:
    """Read the corpus-aligned tree; corpus identity is checked during preparation.
    读取与语料菌属列对齐的树；整库身份由准备与训练流程统一核对。
    """
    with h5py.File(corpus, "r") as source:
        incidence = read_elem(source["varm/unifrac_incidence"]).tocsr().astype(np.float64)
        tree = source["uns/unifrac"]
        lengths = tree["branch_lengths"][:].astype(np.float32)
        identity = {
            "schema_version": int(tree["schema_version"][()]),
            "variant": tree["variant"].asstr()[()],
            "feature_semantics": tree["feature_semantics"].asstr()[()],
            "source_manifest_sha256": json.loads(tree["provenance"].asstr()[()])["source_manifest_sha256"],
        }
    return UniFracAssets(
        incidence,
        torch.from_numpy(lengths),
        identity,
    )


def full_composition_edge_mass(
    samples: Sequence[dict[str, Any]], incidence: sparse.csr_matrix
) -> np.ndarray:
    """Map full relative abundances [B, V] to branch masses [B, E].
    将完整相对丰度 [B, V] 映射为树枝质量 [B, E]，不使用 Top-K 截取视图。
    """
    indptr = np.r_[0, np.cumsum([len(sample["full_indices"]) for sample in samples])]
    composition = sparse.csr_matrix(
        (
            np.concatenate([sample["full_relative"] for sample in samples]),
            np.concatenate([sample["full_indices"] for sample in samples]),
            indptr,
        ),
        shape=(len(samples), incidence.shape[0]),
    )
    return (composition @ incidence).toarray().astype(np.float32)
