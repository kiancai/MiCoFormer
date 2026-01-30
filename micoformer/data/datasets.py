from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import anndata as ad
from scipy import sparse as sp

from micoformer.data.binning import compute_log_bin_edges, bin_values_log, bin_values_rank


class AnnDataDataset:

    def __init__(
        self,
        *,
        h5ad_path: str,
        backed: Optional[str] = None,
        max_seq_len: Optional[int] = None,
        num_abundance_bins: int = 40,
        min_abundance: float = 4e-6,
        abundance_mode: str = "abs_log_bins",
    ) -> None:

        # 读取 .h5ad 文件
        self.adata = ad.read_h5ad(h5ad_path, backed=backed)
        self.is_backed = backed is not None

        # 记录样本总数 (N) 和 特征/物种总数 (V)
        self.n_samples = int(self.adata.n_obs)
        self.n_taxa = int(self.adata.n_vars)

        # 配置参数
        self.abundance_mode = abundance_mode

        self.num_real_bins = num_abundance_bins
        self.min_abundance = min_abundance

        if abundance_mode == "abs_log_bins":
            self.abund_bin_edges = compute_log_bin_edges(
                num_bins=self.num_real_bins,
                min_val=min_abundance,
                max_val=1.0
            )
        elif abundance_mode == "rank_bins":
            pass
        else:
            raise ValueError(f"Unknown abundance_mode: {abundance_mode}")
        
        # 总 Bin 数 = 真实 Bin + 2 (PAD=0, MASK=1)
        self.num_abund_bins_total = self.num_real_bins + 2

        self.pad_token_id = 0
        self.sample_token_id = 1
        self.pad_bin_id = 0
        self.mask_bin_id = 1

        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return self.n_samples

    def _row_nonzero(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        # 获取样本 i 的所有非 0 丰度菌 index 与 vals
        X = self.adata.X
        row = X[i]
        if sp.issparse(row):
            row = row.tocsr()
            idx = row.indices 
            vals = row.data   
        else:
            arr = np.asarray(row).ravel()
            idx = np.nonzero(arr)[0]
            vals = arr[idx]
        return idx, vals

    def _bin_abundance_abs(self, values: np.ndarray) -> np.ndarray:
        # 使用绝对丰度，在 log 空间下分 bin
        bins = bin_values_log(
            values=values,
            edges=self.abund_bin_edges,
            min_val=self.min_abundance,
            max_val=1.0,
            num_bins=self.num_real_bins
        )
        return bins + 2

    def _bin_abundance_rank(self, values: np.ndarray) -> np.ndarray:
        # 使用相对丰度 rank 分 bin
        bins = bin_values_rank(
            num_items=len(values),
            num_bins=self.num_real_bins
        )
        return bins + 2

    def __getitem__(self, i: int) -> Dict[str, Any]:

        idx, vals = self._row_nonzero(i)
        
        if idx.size == 0:
            taxon_ids = np.empty((0,), dtype=np.int64)
            abund_bins = np.empty((0,), dtype=np.int64)
        else:
            order = np.argsort(-vals)  # 按丰度值降序排列
            idx = idx[order]
            vals = vals[order]
            
            if self.max_seq_len is not None:
                idx = idx[: self.max_seq_len]
                vals = vals[: self.max_seq_len]

            taxon_ids = idx.astype(np.int64) + 2
            
            if self.abundance_mode == "abs_log_bins":
                abund_bins = self._bin_abundance_abs(vals).astype(np.int64)
            elif self.abundance_mode == "rank_bins":
                abund_bins = self._bin_abundance_rank(vals).astype(np.int64)
            else:
                abund_bins = np.zeros_like(vals, dtype=np.int64) # Should not happen

        item = {  # 组装返回字典
            "taxon_ids": taxon_ids,
            "abund_bins": abund_bins,
            "length": int(taxon_ids.shape[0]),
        }
        
        return item


    @property
    def vocab_size(self) -> int:
        return self.n_taxa + 2

    @property
    def num_abundance_bins(self) -> int:
        return self.num_abund_bins_total

    @property
    def special_ids(self) -> Dict[str, int]:
        return {
            "pad_token_id": self.pad_token_id,
            "sample_token_id": self.sample_token_id,
            "pad_bin_id": self.pad_bin_id,
            "mask_bin_id": self.mask_bin_id,
        }
