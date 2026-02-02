from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import anndata as ad
import lightning as L
from torch.utils.data import DataLoader, Subset

from micoformer.data.datasets import AnnDataDataset
from micoformer.data.collate import MiCoCollator
from lightning.pytorch.utilities import rank_zero_info


class MiCoDataModule(L.LightningDataModule):

    def __init__(
        self,
        *,
        h5ad_path: str,
        train_indices: Optional[Sequence[int]] = None, 
        val_indices: Optional[Sequence[int]] = None,   
        test_indices: Optional[Sequence[int]] = None,  
        batch_size: int = 32,
        num_workers: int = 4,
        max_seq_len: Optional[int] = 1024,
        mask_prob: float = 0.15,               # 预训练时 token 被掩码的概率（MLM 任务）
        num_abundance_bins: int = 40,          # 丰度分箱数量 (不含 PAD/MASK)
        min_abundance: float = 4e-6,           # 最小丰度阈值 (低于此值归入第一箱)
        abundance_mode: str = "abs_log_bins",  # 丰度编码方式："abs_log_bins" 或 "rank_bins"
    ) -> None:

        super().__init__()
        self.h5ad_path = h5ad_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        
        # 丰度分箱配置
        self.cfg_num_abundance_bins = num_abundance_bins
        self.min_abundance = min_abundance
        self.abundance_mode = abundance_mode
        
        # 索引参数
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        self.persistent_workers = True
        self.pin_memory = True

        self.special_ids = {
            "pad_taxon_id": 0,
            "sample_token_id": 1,
            "pad_bin_id": 0,
            "mask_bin_id": 1,
        }
        self.num_abundance_bins = self.cfg_num_abundance_bins + 2
        self.vocab_size = self._peek_vocab_size()

        # 数据集占位符 (在 setup 阶段初始化)
        self.train_dataset: Optional[AnnDataDataset] = None
        self.val_dataset: Optional[AnnDataDataset] = None
        self.test_dataset: Optional[AnnDataDataset] = None

    def _peek_vocab_size(self) -> int:
        # 只读取 h5ad 的特征维度信息，避免为了拿 vocab_size 提前构建完整 dataset
        adata = ad.read_h5ad(self.h5ad_path, backed="r")
        try:
            n_taxa = int(adata.n_vars)
        finally:
            # 及时关闭 backed 文件句柄，避免占用文件资源
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        return n_taxa + 2

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # setup 方法在每个进程上都会被调用
        base_dataset = AnnDataDataset(
            h5ad_path=self.h5ad_path,
            backed=None,   # 强制全部加载到内存
            max_seq_len=self.max_seq_len,
            num_abundance_bins=self.cfg_num_abundance_bins,
            min_abundance=self.min_abundance,
            abundance_mode=self.abundance_mode,
        )

        # 执行划分逻辑
        train_idx = np.array(self.train_indices, dtype=np.int64) if self.train_indices is not None else None
        val_idx = np.array(self.val_indices, dtype=np.int64) if self.val_indices is not None else None
        test_idx = np.array(self.test_indices, dtype=np.int64) if self.test_indices is not None else None

        if any(x is not None for x in [self.train_indices, self.val_indices, self.test_indices]):
            rank_zero_info("Applying external split from injected indices.")

        # Subset
        self.train_dataset = Subset(base_dataset, train_idx.tolist()) if train_idx is not None else None
        self.val_dataset = Subset(base_dataset, val_idx.tolist()) if val_idx is not None else None
        self.test_dataset = Subset(base_dataset, test_idx.tolist()) if test_idx is not None else None
        
        # 打印统计信息
        stats = []
        if self.train_dataset: stats.append(f"Train={len(self.train_dataset)}")
        if self.val_dataset: stats.append(f"Val={len(self.val_dataset)}")
        if self.test_dataset: stats.append(f"Test={len(self.test_dataset)}")
        if stats:
            rank_zero_info(f"Split stats: {', '.join(stats)}")

        # 稳健起见，setup 后再与实际 dataset 对齐一次
        self.vocab_size = base_dataset.vocab_size
        self.num_abundance_bins = base_dataset.num_abundance_bins
        self.special_ids = base_dataset.special_ids

    # DataLoaders 构建
    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        collate_function = MiCoCollator(
            pad_taxon_id=self.special_ids["pad_taxon_id"],
            sample_token_id=self.special_ids["sample_token_id"],
            pad_bin_id=self.special_ids["pad_bin_id"],
            mask_bin_id=self.special_ids["mask_bin_id"],
            mask_prob=self.mask_prob,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collate_function,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not loaded (train_indices is None).")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not loaded (val_indices is None).")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not loaded (test_indices is None).")
        return self._create_dataloader(self.test_dataset, shuffle=False)
