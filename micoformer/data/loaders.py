"""Build DataLoaders for training, validation, and embedding extraction.
创建训练、验证和 embedding 提取所需的数据读取器。
"""
from __future__ import annotations

import multiprocessing as mp
from multiprocessing.sharedctypes import Synchronized
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
from micoformer.config import RunConfig
from micoformer.data.datasets import CorpusDataset
from micoformer.data.samplers import LengthBucketSampler
from micoformer.data.unifrac import UniFracAssets
from micoformer.data.collate import (
    PretrainingCollator,
    collate_unmasked,
)


@dataclass
class TrainingData:
    """Bundle per-rank readers, samplers, and shared epoch state.
    集中保存当前进程的读取器、采样器和共享轮数状态。
    """
    train_loader: DataLoader[dict[str, np.ndarray]]
    val_loader: DataLoader[dict[str, np.ndarray]]

    train_sampler: DistributedSampler[int]
    val_sampler: DistributedSampler[int]

    train_epoch: Synchronized
    val_epoch: Synchronized

    train_rows: np.ndarray
    val_rows: np.ndarray

    @classmethod
    def from_config(
        cls,
        config: RunConfig,
        *,
        rank: int,
        world: int,
        observed: np.ndarray,
        unifrac_assets: UniFracAssets | None = None,
    ) -> TrainingData:
        """Build training/validation readers from prepared splits.
        根据已准备的划分创建训练和验证读取器。
        """
        # Read corpus row IDs [N_train]/[N_val]; the abundance matrix stays on disk.
        # 读取语料行号 [N_train]/[N_val]，丰度矩阵仍按需从磁盘读取。
        train_rows = np.asarray(
            np.load(config.data.train_rows, mmap_mode="r", allow_pickle=False)
        )
        val_rows = np.asarray(
            np.load(config.data.val_rows, mmap_mode="r", allow_pickle=False)
        )
        # Use the configured sequence limit and retain full support for negative sampling.
        # 按模型配置截取序列，保留完整菌属集合用于负例排除。
        dataset_options = {
            "max_seq_len": config.model.max_seq_len,
            "include_full_abundance": unifrac_assets is not None,
        }
        train_dataset = CorpusDataset(config.data.corpus, train_rows, **dataset_options)
        val_dataset = CorpusDataset(config.data.corpus, val_rows, **dataset_options)
        # Bucket training lengths after rank sharding, or keep random batches.
        # 各卡分片后按训练长度分桶，也可配置为原来的随机组批。
        sampler_options = {
            "num_replicas": world,
            "rank": rank,
            "seed": config.training.seed,
        }
        train_sampler: DistributedSampler[int]
        if config.training.length_bucketing:
            train_sampler = LengthBucketSampler(
                train_dataset,
                train_dataset.sequence_lengths(),
                batch_size=config.training.batch_size_per_rank,
                **sampler_options,
            )
        else:
            train_sampler = DistributedSampler(
                train_dataset, shuffle=True, drop_last=False, **sampler_options,
            )
        # Validation uses a fixed order, partitioned across the same ranks.
        # 验证按相同进程数分片，保持固定顺序。
        val_sampler: DistributedSampler[int] = DistributedSampler(
            val_dataset,
            num_replicas=world,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        # Samplers repeat indices when needed so every rank has equal sample counts.
        # 无法均分时由采样器补少量重复位置，使各进程样本数相同。
        # Persistent workers read shared int64 epoch values; validation stays at 0.
        # 持久 worker 读取共享 int64 轮数；训练逐轮更新，验证保持为 0。
        train_epoch = mp.Value("q", 0)
        val_epoch = mp.Value("q", 0)
        # Read CPU worker count and per-GPU batch size from the training config.
        # CPU 加载进程数和每卡批大小统一取自训练配置。
        worker_count = config.training.workers_per_rank
        effective_batch = config.training.batch_size_per_rank
        # Shared train/validation settings; the final partial batch is retained.
        # 训练与验证共用读取设置，保留最后不足一批的样本。
        common: dict[str, Any] = {
            "batch_size": effective_batch,
            "num_workers": worker_count,
            # Keep workers between epochs; enable prefetching only with workers.
            # 跨轮保留 worker，仅在有 worker 时启用预取。
            "persistent_workers": worker_count > 0,
            "prefetch_factor": 2 if worker_count > 0 else None,
            "drop_last": False,
        }
        # Connect sample reading, rank sampling, and task collation; stream 1 is training.
        # 连接样本读取、进程采样与任务组批；stream=1 用于训练。
        train_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            collate_fn=PretrainingCollator(
                observed_genus_ids=observed,
                seed=config.training.seed,
                stream=1,
                shared_epoch=train_epoch,
                mlm_mask_probability=config.training.mlm_mask_probability,
                query_mask_probability=config.training.query_mask_probability,
                unifrac_incidence=unifrac_assets.incidence if unifrac_assets is not None else None,
            ),
            **common,
        )
        # Stream 2 and epoch 0 keep validation masks and negative sampling repeatable.
        # 验证使用 stream=2、epoch=0，在组批不变时复现遮盖与负例。
        val_loader = DataLoader(
            val_dataset,
            sampler=val_sampler,
            collate_fn=PretrainingCollator(
                observed_genus_ids=observed,
                seed=config.training.seed,
                stream=2,
                shared_epoch=val_epoch,
                mlm_mask_probability=config.training.mlm_mask_probability,
                query_mask_probability=config.training.query_mask_probability,
                unifrac_incidence=unifrac_assets.incidence if unifrac_assets is not None else None,
            ),
            **common,
        )
        # DataLoaders fetch samples lazily when the training loop iterates over them.
        # 训练循环遍历读取器时才逐批取样，不在此处加载整个语料。
        return cls(
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            val_sampler=val_sampler,
            train_epoch=train_epoch,
            val_epoch=val_epoch,
            train_rows=train_rows,
            val_rows=val_rows,
        )

    def set_train_epoch(self, epoch: int) -> None:
        """Update training order and task randomness for an epoch.
        更新当前轮的训练顺序与任务随机过程。
        """
        # The sampler controls order; the shared value controls worker-side targets.
        # 采样器控制样本顺序，共享值控制 worker 生成的遮盖与负例。
        self.train_sampler.set_epoch(epoch)
        self.train_epoch.value = epoch

    def reset_validation(self) -> None:
        """Keep validation order and task randomness at epoch 0.
        将验证顺序与任务随机过程固定在 epoch=0。
        """
        self.val_sampler.set_epoch(0)
        self.val_epoch.value = 0

    def close(self) -> None:
        """Stop persistent workers and close dataset file handles.
        结束持久 worker，并关闭数据集文件句柄。
        """
        for loader in (self.train_loader, self.val_loader):
            # A persistent iterator exists only after iteration has started.
            # 开始遍历后才可能存在持久迭代器；通过 PyTorch 内部接口结束其 worker。
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None
            loader.dataset.close()

    @staticmethod
    def prediction_loader(
        corpus: Path, rows: np.ndarray, *, max_seq_len: int, batch_size: int = 128, workers: int = 0
    ) -> DataLoader:
        """Build an ordered, unmasked reader for embedding extraction.
        创建按给定行号顺序读取、不施加遮盖的 embedding 提取读取器。
        """
        # Standalone extraction uses its own batch/worker settings, without rank sharding.
        # 独立提取使用自己的批大小和 worker 设置，不按训练 rank 分片。
        return DataLoader(
            CorpusDataset(corpus, rows, max_seq_len=max_seq_len),
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            # Pad variable-length samples without MLM/Query targets or negatives.
            # 只补齐不同长度的样本，不生成 MLM/Query 目标和负例。
            collate_fn=collate_unmasked,
        )
