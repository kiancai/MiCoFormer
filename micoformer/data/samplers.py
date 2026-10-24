"""Length-bucketed sampling for distributed training.
用于分布式训练的长度分桶采样。
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler


class LengthBucketSampler(DistributedSampler[int]):
    """Group similar lengths after epoch-wise rank sharding.
    逐轮随机分配各卡样本后，再将相近长度组批。
    """

    def __init__(
        self,
        dataset: Dataset[Any],
        lengths: np.ndarray,
        *,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int,
    ) -> None:
        # Keep the standard coverage and minimal padding across ranks.
        # 沿用标准分布式采样的样本覆盖及最少重复补齐规则。
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank,
            shuffle=True, seed=seed, drop_last=False,
        )
        self.lengths = lengths
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[int]:
        # The parent shuffles each epoch; stable sorting keeps ties randomized.
        # 父类逐轮打乱样本，稳定排序让同长度样本仍保持随机顺序。
        indices = np.fromiter(super().__iter__(), dtype=np.int64, count=self.num_samples)
        indices = indices[np.argsort(self.lengths[indices], kind="stable")]
        complete = self.num_samples // self.batch_size
        boundary = complete * self.batch_size
        batches = indices[:boundary].reshape(complete, self.batch_size)
        # Use the same batch permutation on all ranks to align length ranges.
        # 各卡共用批次排列，让同一步的长度区间尽量接近。
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(complete, generator=generator).numpy()
        # Keep the partial batch last so DataLoader never mixes bucket boundaries.
        # 不足一批的样本留在最后，避免 DataLoader 重新混合分桶边界。
        return iter(np.concatenate((batches[order].reshape(-1), indices[boundary:])).tolist())
