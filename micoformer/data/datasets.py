from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import h5py
import numpy as np
from anndata.io import sparse_dataset
from torch.utils.data import Dataset


class CorpusDataset(Dataset[dict[str, Any]]):
    """Read individual samples and retain full genus support for negative sampling.
    读取单个样本，保留截取前的完整菌属集合供负例采样排除。
    """

    def __init__(self, corpus: str | Path, corpus_rows: np.ndarray, *, max_seq_len: int) -> None:
        # Selected corpus rows and sequence limit from the model configuration.
        # 选中的语料行号，以及模型配置提供的最大序列长度。
        self.corpus_rows = np.asarray(corpus_rows, dtype=np.int64)
        self.max_seq_len = max_seq_len
        # Open lazily and reuse the file connection within each process.
        # 首次读取时打开文件，各进程独立保存并复用连接。
        self._handle: h5py.File | None = None
        self._matrix = None
        self._pid: int | None = None
        self.corpus = Path(corpus).resolve()

    def __len__(self) -> int:
        """Number of selected samples. / 当前数据集的样本数。"""
        return int(self.corpus_rows.size)

    def _data(self):
        """Open disk X on first access or after a process change.
        首次读取或进程变化时，打开磁盘中的丰度矩阵 X。
        """
        if self._handle is None or self._pid != os.getpid():
            self.close()
            self._handle = h5py.File(self.corpus, "r")
            # Access CSR rows on demand without loading the full matrix.
            # 按需读取 CSR 行，避免加载整个矩阵。
            self._matrix = sparse_dataset(self._handle["X"])
            self._pid = os.getpid()
        return self._matrix

    def __getitem__(self, dataset_index: int) -> dict[str, Any]:
        """Return one sample before padding and masking.
        返回一个尚未补齐或遮盖的样本。
        """

        # Map dataset position to a corpus row; retain shape [1, V] for V genus columns.
        # 将数据集位置映射为语料行号，保留 [1, V]，V 为菌属列数。
        corpus_row_id = int(self.corpus_rows[dataset_index])
        sample_row = self._data()[corpus_row_id : corpus_row_id + 1, :]

        # Aligned [M] arrays: all nonzero genus columns and their corpus abundance.
        # 两个 [M] 数组逐项对应：全部非零菌属列号及其语料丰度。
        genus_columns = sample_row.indices.astype(np.int64, copy=False)
        corpus_abundance = sample_row.data.astype(np.float64, copy=False)

        # Sort by descending abundance, breaking ties by column; keep K=min(M, max_seq_len).
        # 丰度降序，同丰度按列号升序；保留 K=min(M, max_seq_len) 个位置。
        order = np.lexsort((genus_columns, -corpus_abundance))[:self.max_seq_len]

        # Center log abundance within the retained genera, without SD scaling.
        # 在保留的菌属内计算 log(a)-mean(log(a))，不除以标准差。
        log_abundance = np.log(corpus_abundance[order], dtype=np.float64)
        abundance = (log_abundance - log_abundance.mean(dtype=np.float64)).astype(np.float32)

        # Offset selected columns by 2 to reserve token IDs 0 and 1.
        # 所选列号加 2，为特殊 token 编号 0、1 留出位置。
        genus_ids = (genus_columns[order] + 2).astype(np.int64, copy=False)

        return {
            # Corpus row ID for tracking and deterministic masking.
            # 语料行号，用于追踪样本和生成确定性遮盖。
            "row_id": np.int64(corpus_row_id),
            # Model inputs [K]; the collator batches, pads, and selects masking targets.
            # 模型输入均为 [K]；collator 组批、补齐并生成遮盖标记，模型执行遮盖。
            "genus_ids": genus_ids,
            "abundance": abundance,
            # Full support [M] excludes all present genera from negative sampling.
            # 完整菌属集合 [M]，确保全部真实菌属都从负例中排除。
            "full_indices": genus_columns,
            "richness": np.int64(genus_columns.size),
        }

    def close(self) -> None:
        """Close the file and clear cached state. / 关闭文件并清空读取状态。"""
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._matrix = None
            self._pid = None
