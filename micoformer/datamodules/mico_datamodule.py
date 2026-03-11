from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import anndata as ad
import lightning as L
from torch.utils.data import DataLoader, Subset

from micoformer.data.datasets import AnnDataDataset, build_taxon_path_ids
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
        batch_size: int = 32,                  # 批次大小 
        num_workers: int = 4,                  # 数据加载线程数
        max_seq_len: Optional[int] = 1024,     # 最大序列长度
        mask_prob: float = 0.15,               # 预训练时 token 被掩码的概率（MLM 任务）
        num_abundance_bins: int = 40,          # 丰度分箱数量 (不含 PAD/MASK)
        min_abundance: float = 4e-6,           # 最小丰度阈值 (低于此值归入第一箱)
        abundance_mode: str = "abs_log_bins",  # 丰度编码方式："abs_log_bins" 或 "rank_bins"
        token_embedding_mode: str = "taxon_path",  # token 嵌入方式："taxon" 或 "taxon_path"
    ) -> None:

        super().__init__()
        self.h5ad_path = h5ad_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob
        
        # 丰度分箱配置
        self.num_abundance_bins = num_abundance_bins    # 用户指定的真实 bin 数（不含 PAD/MASK）
        self.min_abundance = min_abundance
        self.abundance_mode = abundance_mode
        if token_embedding_mode not in {"taxon", "taxon_path"}:
            raise ValueError(
                f"Unknown token_embedding_mode: {token_embedding_mode}. "
                "Expected 'taxon' or 'taxon_path'."
            )
        self.token_embedding_mode = token_embedding_mode
        
        # 索引参数
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        self.persistent_workers = True  # 保持 worker 进程 alive，避免重复加载数据
        self.pin_memory = True          # 启用内存锁页，加速数据传输到 GPU

        self.special_ids = {
            "pad_taxon_id": 0,   # 0=PAD
            # taxon id 1 表示 UNK Taxon，但不算 special id
            "pad_bin_id": 0,     # 0=PAD
            "mask_bin_id": 1,    # 1=MASK
        }
        self.total_abundance_bins = self.num_abundance_bins + 2   # 含 PAD/MASK 的总数，传入模型
        self.genus_vocab_size, self.rank_vocab_sizes = self._peek_dataset_meta()

        # 数据集占位符 (在 setup 阶段初始化)
        self.train_dataset: Optional[AnnDataDataset] = None
        self.val_dataset: Optional[AnnDataDataset] = None
        self.test_dataset: Optional[AnnDataDataset] = None

    def _peek_dataset_meta(self) -> Tuple[int, Dict[str, int]]:
        # 只读取 h5ad 的必要元信息，避免为了拿配置提前构建完整 dataset
        adata = ad.read_h5ad(self.h5ad_path, backed="r")
        try:
            # 始终构建各 rank 词表，两种 embedding 模式均需要
            _, rank_vocab_sizes = build_taxon_path_ids(adata.var)
        finally:
            # 及时关闭 backed 文件句柄，避免占用文件资源
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        # genus_vocab_size：Genus 词表大小（0=PAD, 1=UNK, 2~=真实 genus）
        return rank_vocab_sizes["Genus"], rank_vocab_sizes

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # setup 方法在每个进程上都会被调用
        base_dataset = AnnDataDataset(
            h5ad_path=self.h5ad_path,
            max_seq_len=self.max_seq_len,
            num_abundance_bins=self.num_abundance_bins,
            min_abundance=self.min_abundance,
            abundance_mode=self.abundance_mode,
            token_embedding_mode=self.token_embedding_mode,
        )

        # Subset：直接使用初始化时传入的索引划分数据集
        self.train_dataset = Subset(base_dataset, self.train_indices) if self.train_indices is not None else None
        self.val_dataset = Subset(base_dataset, self.val_indices) if self.val_indices is not None else None
        self.test_dataset = Subset(base_dataset, self.test_indices) if self.test_indices is not None else None
        
        # 打印统计信息
        stats = []
        if self.train_dataset: stats.append(f"Train={len(self.train_dataset)}")
        if self.val_dataset: stats.append(f"Val={len(self.val_dataset)}")
        if self.test_dataset: stats.append(f"Test={len(self.test_dataset)}")
        if stats:
            rank_zero_info(f"Split stats: {', '.join(stats)}")

    # DataLoaders 构建
    def _create_dataloader(self, dataset, shuffle: bool) -> DataLoader:
        collate_function = MiCoCollator(
            pad_taxon_id=self.special_ids["pad_taxon_id"],
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
