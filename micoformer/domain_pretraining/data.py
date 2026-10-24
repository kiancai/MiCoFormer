"""Exact row coverage and count-weighted sequential microbatches.

单卡顺序执行原来每卡的 B128 局部关系池，不把关系池扩大到 B384。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import islice

import numpy as np
import torch
from torch.utils.data import Sampler

from micoformer.mpc_fixed_pretraining.model import MPCFixedModelConfig
from micoformer.mpc_unifrac.support import NAMES


@dataclass(frozen=True)
class DomainModelConfig(MPCFixedModelConfig):
    """Allow just the two approved depths; retain all other MPC guards."""

    def __post_init__(self) -> None:
        if self.num_layers not in (6, 12):
            raise ValueError("domain ablation supports only 6 or 12 encoder layers")
        reference = asdict(self)
        reference["num_layers"] = 12
        MPCFixedModelConfig(**reference)


class ExactBatchSampler(Sampler[list[int]]):
    """Visit every row once, without a singleton UniFrac pool or padding.

    A final singleton borrows one row from its preceding batch (127 + 2).
    No batch exceeds the memory-tested microbatch size.
    """

    def __init__(self, size: int, *, seed: int, epoch: int, shuffle: bool,
                 batch_size: int = 128, start_batch: int = 0) -> None:
        if size < 2 or batch_size < 3:
            raise ValueError("UniFrac needs at least two rows and batch_size >= 3")
        if shuffle:
            generator = torch.Generator().manual_seed(seed + epoch)
            order = torch.randperm(size, generator=generator).tolist()
        else:
            order = list(range(size))
        self.batches = [order[i:i + batch_size] for i in range(0, size, batch_size)]
        if len(self.batches[-1]) == 1:
            self.batches[-1].insert(0, self.batches[-2].pop())
        if not 0 <= start_batch <= len(self.batches):
            raise ValueError("invalid resume batch cursor")
        self.start_batch = start_batch

    def __iter__(self):
        return iter(self.batches[self.start_batch:])

    def __len__(self) -> int:
        return len(self.batches) - self.start_batch


def accumulation_groups(iterator, count: int = 3):
    iterator = iter(iterator)
    while group := list(islice(iterator, count)):
        yield group


def target_counts(batch) -> np.ndarray:
    mlm = int(np.asarray(batch["mlm_mask"]).sum())
    query = int(np.asarray(batch["query_mask"]).sum())
    return np.asarray([mlm, query, 2 * query, len(batch["row_ids"])], dtype=np.int64)


def accumulation_loss(output, weights, totals: np.ndarray):
    """Same per-component global target mean as the three-rank DDP recipe.

    Equal averaging of microbatch means would misweight different target counts,
    especially the final partial update. Backward frees each microbatch graph.
    """
    if np.any(totals <= 0):
        raise ValueError("empty loss component")
    return sum(float(weights[name]) * getattr(output, name + "_loss")
               * (getattr(output, name + "_count") / int(totals[i]))
               for i, name in enumerate(NAMES))
