"""Load genus data and build phylogenetic/protein priors from the unified corpus.
从统一语料读取菌属名称、已观测编号，并构建进化与蛋白先验表。
"""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import torch


@dataclass(frozen=True)
class GenusData:
    prior_table: torch.Tensor
    vocabulary: tuple[str, ...]
    observed_genus_ids: np.ndarray


def load_genus_data(corpus: str | Path) -> GenusData:
    """Load genus names and observed IDs, and build the prior table from V4.1.
    从 V4.1 读取菌属名称与已观测编号，并构建先验表。
    """
    # Read only genus-level metadata, not the abundance matrix.
    # 只读取菌属层面的特征，不把整个丰度矩阵装入内存。
    with h5py.File(corpus, "r") as source:
        version = source["uns/corpus_version"].asstr()[()]
        if version != "MCFCorpusV4.1":
            raise ValueError(f"expected MCFCorpusV4.1, found {version}")
        phy = source["varm/position_encoding"][:].astype(np.float32)
        feat = source["varm/protein_features"][:].astype(np.float32)
        names = tuple(source["var/_index"].asstr()[:])
        observed = np.flatnonzero(source["var/n_samples_observed"][:] > 0).astype(np.int64) + 2

    # Center and scale PHY globally to preserve relative distances.
    # 系统发育坐标统一中心化和缩放，保留菌属间距离的比例。
    centered = phy - phy.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    scale = float(np.median(norms[norms > 0]))
    phy_unit = centered / scale
    # Normalize every genus protein vector to unit length.
    # 对全部菌属的蛋白向量逐行做 L2 归一化。
    norms = np.linalg.norm(feat, axis=1)
    prot_unit = feat / norms[:, None]
    prior_table = np.zeros((8116, 512), dtype=np.float32)
    # Scale by sqrt(512 / 2)=16 to approximately match the abundance LayerNorm's initial magnitude.
    # 乘 sqrt(512 / 2)=16，使合并先验与丰度分支 LayerNorm 的初始数值尺度接近。
    prior_table[2:, :32] = phy_unit * np.sqrt(512 / 2)
    prior_table[2:, 32:] = prot_unit * np.sqrt(512 / 2)
    return GenusData(torch.from_numpy(prior_table), names, observed)
