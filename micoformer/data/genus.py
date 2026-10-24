"""Read the released phylogenetic/protein prior from the unified corpus.
从统一语料读取已处理的进化与蛋白先验、菌属名称和已观测编号。
"""
from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np
import torch
from micoformer.config import ModelConfig


@dataclass(frozen=True)
class GenusData:
    prior_table: torch.Tensor
    vocabulary: tuple[str, ...]
    observed_genus_ids: np.ndarray


def load_genus_data(corpus: str | Path) -> GenusData:
    """Read the frozen V4.3 prior without repeating its preprocessing.
    读取 V4.3 固定先验，不重复中心化、缩放或逐属归一化。
    """
    # Read only genus-level metadata, not the abundance matrix.
    # 只读取菌属层面的特征，不把整个丰度矩阵装入内存。
    with h5py.File(corpus, "r") as source:
        version = source["uns/corpus_version"].asstr()[()]
        if version != "MCFCorpusV4.3":
            raise ValueError(f"expected MCFCorpusV4.3, found {version}")
        prior = source["varm/genus_prior"][:].astype(np.float32)
        names = tuple(source["var/_index"].asstr()[:])
        observed = np.flatnonzero(source["var/n_samples_observed"][:] > 0).astype(np.int64) + 2

    config = ModelConfig()
    if prior.shape != (config.vocab_size - 2, config.d_model) or len(names) != prior.shape[0]:
        raise ValueError("corpus prior shape does not match the model vocabulary and width")
    if not np.isfinite(prior).all():
        raise ValueError("corpus prior contains nonfinite values")
    # PAD and UNK stay zero and were excluded from release-level statistics.
    # PAD 与 UNK 保持全零，不参与发布时的均值和尺度统计。
    prior_table = np.zeros((config.vocab_size, config.d_model), dtype=np.float32)
    prior_table[2:] = prior
    return GenusData(torch.from_numpy(prior_table), names, observed)
