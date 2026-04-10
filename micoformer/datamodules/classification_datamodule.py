from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import anndata as ad
import lightning as L
from torch.utils.data import DataLoader, Subset

from micoformer.data.datasets import AnnDataDataset, build_taxon_path_ids
from micoformer.data.classification_collate import ClassificationCollator
from lightning.pytorch.utilities import rank_zero_info


class _LabelWrappedSubset:
    """包装 Subset，在 __getitem__ 时附加标签。"""

    def __init__(
        self,
        subset: Subset,
        labels_array: np.ndarray,  # [n_samples, n_tasks]，全局索引
        task_names: List[str],
    ) -> None:
        self.subset = subset
        self.labels_array = labels_array
        self.task_names = task_names

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.subset[idx]
        # subset.indices[idx] 是全局索引
        global_idx = self.subset.indices[idx]
        item["labels"] = {
            name: int(self.labels_array[global_idx, ti])
            for ti, name in enumerate(self.task_names)
        }
        return item


TAG = "[cls_datamodule]"


class ClassificationDataModule(L.LightningDataModule):
    """下游分类任务的数据管道。复用 AnnDataDataset，在 collator 层附加标签。"""

    def __init__(
        self,
        *,
        h5ad_path: str,
        label_configs: List[Dict[str, Any]],
        # 每项形如 {"field": "Phenotype", "values": ["Health", "Disease"]}
        # values 指定有效值，其他值标记为 -1
        train_indices: Optional[Sequence[int]] = None,
        val_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        max_seq_len: Optional[int] = 1024,
        num_abundance_bins: int = 40,
        min_abundance: float = 4e-6,
        abundance_mode: str = "abs_log_bins",
    ) -> None:
        super().__init__()
        self.h5ad_path = h5ad_path
        self.label_configs = label_configs
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.num_abundance_bins = num_abundance_bins
        self.min_abundance = min_abundance
        self.abundance_mode = abundance_mode

        self.persistent_workers = True
        self.pin_memory = True

        self.special_ids = {
            "pad_taxon_id": 0,
            "pad_bin_id": 0,
        }
        self.total_abundance_bins = self.num_abundance_bins + 2

        self.task_configs: List[Dict[str, Any]] = []
        self._labels_array: Optional[np.ndarray] = None
        self._task_names: List[str] = []

        # 一次性读取 var（词表）和 obs（标签配置），避免重复打开 h5ad（P2-4）
        self._init_metadata()

    def _init_metadata(self) -> None:
        """一次性打开 h5ad，读取 var（词表）和 obs（标签配置）后关闭。"""
        adata = ad.read_h5ad(self.h5ad_path, backed="r")
        try:
            # 读取词表
            _, rank_vocab_sizes, _ = build_taxon_path_ids(adata.var)
            self.genus_vocab_size = rank_vocab_sizes["Genus"]
            self.rank_vocab_sizes = rank_vocab_sizes

            # 构建标签配置（P2-5：向量化标签填充）
            obs = adata.obs
            n_samples = len(obs)
            labels_array = np.full((n_samples, len(self.label_configs)), -1, dtype=np.int64)

            for ti, cfg in enumerate(self.label_configs):
                field = cfg["field"]
                valid_values = cfg.get("values", None)

                if field not in obs.columns:
                    available = ", ".join(sorted(obs.columns))
                    raise ValueError(
                        f"Label field '{field}' not found in obs. Available: {available}"
                    )

                col = obs[field].to_numpy()

                # 构建 label_to_id 映射
                if valid_values is not None:
                    label_to_id = {v: i for i, v in enumerate(valid_values)}
                else:
                    unique_vals = sorted(
                        set(str(v) for v in np.unique(col)
                            if str(v).lower() not in ("nan", "none", "<na>", ""))
                    )
                    label_to_id = {v: i for i, v in enumerate(unique_vals)}

                # 向量化标签填充（替代逐样本循环）
                col_str = np.array([str(v).strip() for v in col])
                for label_str, lid in label_to_id.items():
                    labels_array[col_str == label_str, ti] = lid

                self.task_configs.append({
                    "field": field,
                    "label_to_id": label_to_id,
                    "num_classes": len(label_to_id),
                })
                self._task_names.append(field)
        finally:
            if getattr(adata, "file", None) is not None:
                adata.file.close()

        self._labels_array = labels_array

    def setup(self, stage: Optional[str] = None) -> None:
        base_dataset = AnnDataDataset(
            h5ad_path=self.h5ad_path,
            max_seq_len=self.max_seq_len,
            num_abundance_bins=self.num_abundance_bins,
            min_abundance=self.min_abundance,
            abundance_mode=self.abundance_mode,
        )

        assert self._labels_array is not None

        def _wrap(indices: Optional[Sequence[int]]) -> Optional[_LabelWrappedSubset]:
            if indices is None:
                return None
            subset = Subset(base_dataset, indices)
            return _LabelWrappedSubset(subset, self._labels_array, self._task_names)

        self.train_dataset = _wrap(self.train_indices)
        self.val_dataset = _wrap(self.val_indices)
        self.test_dataset = _wrap(self.test_indices)

        # 打印统计信息
        stats = []
        if self.train_dataset:
            stats.append(f"Train={len(self.train_dataset)}")
        if self.val_dataset:
            stats.append(f"Val={len(self.val_dataset)}")
        if self.test_dataset:
            stats.append(f"Test={len(self.test_dataset)}")
        if stats:
            rank_zero_info(f"{TAG} Split stats: {', '.join(stats)}")

        # 打印标签分布
        for ti, cfg in enumerate(self.task_configs):
            field = cfg["field"]
            id_to_label = {v: k for k, v in cfg["label_to_id"].items()}
            for split_name, indices in [
                ("train", self.train_indices),
                ("val", self.val_indices),
                ("test", self.test_indices),
            ]:
                if indices is None:
                    continue
                split_labels = self._labels_array[indices, ti]
                valid_mask = split_labels >= 0
                n_valid = valid_mask.sum()
                counts = {}
                for lid in range(cfg["num_classes"]):
                    counts[id_to_label.get(lid, str(lid))] = int((split_labels == lid).sum())
                rank_zero_info(
                    f"{TAG}   {field} [{split_name}]: {n_valid} valid / {len(indices)} total, "
                    f"distribution={counts}"
                )

    def _create_dataloader(self, dataset: _LabelWrappedSubset, shuffle: bool) -> DataLoader:
        collator = ClassificationCollator(
            pad_taxon_id=self.special_ids["pad_taxon_id"],
            pad_bin_id=self.special_ids["pad_bin_id"],
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collator,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not loaded (train_indices is None).")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Val dataset not loaded (val_indices is None).")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not loaded (test_indices is None).")
        return self._create_dataloader(self.test_dataset, shuffle=False)
