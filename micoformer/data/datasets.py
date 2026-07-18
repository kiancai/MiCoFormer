from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import anndata as ad
from scipy import sparse as sp

from lightning.pytorch.utilities import rank_zero_info

from micoformer.data.binning import compute_log_bin_edges, bin_values_log, bin_values_rank


TAG = "[dataset]"


# taxonomy path 中使用的标准层级顺序
# Domain 列只有 d__Bacteria / d__Archaea 两个有效值，参数量增量可忽略，但补全了 6 级 GG2 路径
RANK_COLUMNS = ("Domain", "Phylum", "Class", "Order", "Family", "Genus")
_GENUS_COL_IDX = RANK_COLUMNS.index("Genus")

# V5 §4.2:present-only abundance 数值写法（编码消融旋钮）。rclr_sigma = 现状默认。
#   rclr_sigma : (log-μ)/σ          —— present-only CLR 再 ÷σ（现状）
#   rclr       : log-μ              —— present-only CLR，去 σ
#   rank       : present 内降序排名归一到 (0,1]，丰度越高越接近 1
#   presence   : 全 1               —— 丢量级（MLM target 退化为常数，慎用）
#   raw        : 相对丰度原值
_VALID_VALUE_TRANSFORM = {"rclr_sigma", "rclr", "rank", "presence", "raw"}
_VALID_SAMPLE_VIEW_TARGETS = {"raw", "rclr_sigma", "rank", "func_bacformer", "phylo_32coord"}
_SAMPLE_VIEW_ALIASES = {
    "rclr": "rclr_sigma",
    "func": "func_bacformer",
    "phylo": "phylo_32coord",
}


def _normalize_tax_label(value: Any) -> str:
    # 将输入值标准化为字符串，缺失值统一映射为 __UNK__
    text = str(value).strip()
    # 覆盖: 空字符串, None, NaN, pandas <NA> 等情况
    if not text or text.lower() in ("nan", "none", "<na>"):
        return "__UNK__"
    return text

def build_taxon_path_ids(
    var_df,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, Dict[str, int]]]:
    # 从 adata.var 构建 taxon 的 taxonomy-path id 矩阵。
    # 强制执行严格模式：必须包含所有标准层级列，否则直接报错。
    # 返回:
    #   - path_ids:        [n_taxa, len(RANK_COLUMNS)]，顺序为 RANK_COLUMNS（V5 默认 6 列：Domain..Genus）
    #   - rank_vocab_sizes: 每个 rank 的词表大小（0=PAD，1=UNK，2~=真实值）
    #   - rank_mappings:   每个 rank 的完整 name→ID 字典（含 __PAD__ 和 __UNK__）
    n_taxa = len(var_df.index)
    path_ids = np.zeros((n_taxa, len(RANK_COLUMNS)), dtype=np.int64)
    rank_vocab_sizes: Dict[str, int] = {}
    rank_mappings: Dict[str, Dict[str, int]] = {}

    for col_idx, col_name in enumerate(RANK_COLUMNS):
        if col_name not in var_df.columns:
            raise ValueError(
                f"Missing required taxonomy column: '{col_name}'. "
                f"The dataset must contain all standard ranks: {RANK_COLUMNS}. "
                f"Available columns: {list(var_df.columns)}."
            )

        # 0 保留给 PAD（序列填充位），1 保留给 UNK（taxon 无此层级注释），真实值从 2 开始
        mapping: Dict[str, int] = {"__PAD__": 0, "__UNK__": 1}
        col_values = var_df[col_name].to_numpy()
        col_ids = np.zeros((n_taxa,), dtype=np.int64)

        for i, value in enumerate(col_values):
            key = _normalize_tax_label(value)
            if key not in mapping:
                mapping[key] = len(mapping)
            col_ids[i] = mapping[key]

        path_ids[:, col_idx] = col_ids
        rank_vocab_sizes[col_name] = len(mapping)
        rank_mappings[col_name] = mapping

    return path_ids, rank_vocab_sizes, rank_mappings


def save_taxon_vocab(
    rank_vocab_sizes: Dict[str, int],
    rank_mappings: Dict[str, Dict[str, int]],
    output_path: str,
) -> None:
    # 将词表（name→ID 映射 + 词表大小）保存为 JSON，供人类查阅及下游推理使用。
    # 约定：0=PAD，1=UNK，2~=真实值（与训练时完全一致）。
    vocab = {
        "rank_vocab_sizes": rank_vocab_sizes,
        "mappings": rank_mappings,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    rank_zero_info(f"{TAG} Taxon vocab saved to {output_path}")


def normalize_sample_view_names(names: Optional[Sequence[str] | str]) -> Tuple[str, ...]:
    if names is None:
        return ()
    if isinstance(names, str):
        raw_names = [x.strip() for x in names.replace(",", " ").split()]
    else:
        raw_names = [str(x).strip() for x in names]
    out = []
    for name in raw_names:
        if not name:
            continue
        canonical = _SAMPLE_VIEW_ALIASES.get(name, name)
        if canonical not in _VALID_SAMPLE_VIEW_TARGETS:
            raise ValueError(
                f"Unknown sample-view target {name!r}. "
                f"Expected one of {sorted(_VALID_SAMPLE_VIEW_TARGETS)}."
            )
        if canonical not in out:
            out.append(canonical)
    return tuple(out)


class AnnDataDataset:

    def __init__(
        self,
        *,
        h5ad_path: str,
        max_seq_len: Optional[int] = None,
        num_abundance_bins: int = 40,
        min_abundance: float = 4e-6,
        abundance_mode: str = "abs_log_bins",
        # V5 新增:abundance 编码模式
        #   "mlp" (默认): 输入侧用连续 MLP 编码 (encoder.abund_mlp);
        #                 仍然计算 abund_bins(MLM bin 标签兼容路径用,实际只在 abundance_loss='bin_ce' 时被消费)
        #   "bin":       旧路径,输入侧用 nn.Embedding 查表
        # 两种模式 __getitem__ 始终返回 abund_values 与 abund_bins(由 collator/module 按 flag 选用)
        abundance_encoding: str = "mlp",
        # V5 §4.2:present-only abundance 数值写法（旧消融旋钮，详见 _VALID_VALUE_TRANSFORM）。
        # 默认兼容旧行为：同一个写法同时用于 encoder 输入和 MLM 回归目标。
        abundance_value_transform: str = "rclr_sigma",
        # 2026-07 repr shaping:显式解耦输入和目标。None 表示沿用 abundance_value_transform。
        abundance_input_transform: Optional[str] = None,
        abundance_target_transform: Optional[str] = None,
        sample_view_heads: Optional[Sequence[str] | str] = None,
        protein_feat_path: Optional[str] = None,
        sample_view_target_index_map: Optional[np.ndarray] = None,
        backed: Optional[str] = None,
    ) -> None:
        if max_seq_len is not None and max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be > 0 when set, got {max_seq_len}")
        if abundance_mode not in {"abs_log_bins", "rank_bins"}:
            raise ValueError(f"Unknown abundance_mode: {abundance_mode}")
        if abundance_encoding not in {"mlp", "bin"}:
            raise ValueError(f"Unknown abundance_encoding: {abundance_encoding}")
        if abundance_value_transform not in _VALID_VALUE_TRANSFORM:
            raise ValueError(
                f"Unknown abundance_value_transform: {abundance_value_transform!r}. "
                f"Expected {sorted(_VALID_VALUE_TRANSFORM)}."
            )
        abundance_input_transform = abundance_input_transform or abundance_value_transform
        abundance_target_transform = abundance_target_transform or abundance_value_transform
        if abundance_input_transform not in _VALID_VALUE_TRANSFORM:
            raise ValueError(
                f"Unknown abundance_input_transform: {abundance_input_transform!r}. "
                f"Expected {sorted(_VALID_VALUE_TRANSFORM)}."
            )
        if abundance_target_transform not in _VALID_VALUE_TRANSFORM:
            raise ValueError(
                f"Unknown abundance_target_transform: {abundance_target_transform!r}. "
                f"Expected {sorted(_VALID_VALUE_TRANSFORM)}."
            )

        # 读取 .h5ad 文件
        self.adata = ad.read_h5ad(h5ad_path, backed=backed)

        # 记录样本总数 (N) 和 特征/物种总数 (V)
        self.n_samples = int(self.adata.n_obs)
        self.n_taxa = int(self.adata.n_vars)
        # 始终构建所有 rank 的 ID 矩阵（两种 embedding 模式都依赖它）：
        # - taxon_path 模式：使用完整的 5 列路径
        # - taxon（baseline）模式：只取 Genus 列作为 taxon_ids
        # rank_mappings 在训练时不需要，忽略第三个返回值
        self._rank_ids, self._rank_vocab_sizes, _ = build_taxon_path_ids(
            self.adata.var
        )

        # 配置参数
        self.abundance_mode = abundance_mode
        self.abundance_encoding = abundance_encoding
        self.abundance_value_transform = abundance_value_transform
        self.abundance_input_transform = abundance_input_transform
        self.abundance_target_transform = abundance_target_transform
        self.sample_view_heads = normalize_sample_view_names(sample_view_heads)

        self.protein_feat: Optional[np.ndarray] = None
        self.sample_view_phylo_coords: Optional[np.ndarray] = None
        if self.sample_view_heads:
            if "func_bacformer" in self.sample_view_heads:
                if protein_feat_path is None:
                    raise ValueError("sample view 'func_bacformer' requires protein_feat_path.")
                self.protein_feat = np.load(protein_feat_path).astype(np.float32, copy=False)
                if self.protein_feat.shape[0] != self.n_taxa:
                    raise ValueError(
                        f"protein_feat shape {self.protein_feat.shape} incompatible with n_taxa={self.n_taxa}."
                    )
            if "phylo_32coord" in self.sample_view_heads:
                if "position_encoding" not in self.adata.varm:
                    raise ValueError("sample view 'phylo_32coord' requires varm['position_encoding'].")
                self.sample_view_phylo_coords = np.asarray(
                    self.adata.varm["position_encoding"], dtype=np.float32
                )
                if self.sample_view_phylo_coords.shape[0] != self.n_taxa:
                    raise ValueError(
                        f"position_encoding shape {self.sample_view_phylo_coords.shape} incompatible with n_taxa={self.n_taxa}."
                    )
        self.sample_view_target_index_map = sample_view_target_index_map
        if self.sample_view_target_index_map is not None:
            if int(self.sample_view_target_index_map.shape[0]) != self.n_samples:
                raise ValueError(
                    "sample_view_target_index_map length must match n_samples: "
                    f"{self.sample_view_target_index_map.shape[0]} != {self.n_samples}."
                )

        self.num_abundance_bins = num_abundance_bins   # 用户指定的真实 bin 数（不含 PAD/MASK）
        self.min_abundance = min_abundance

        if abundance_mode == "abs_log_bins":
            self.abund_bin_edges = compute_log_bin_edges(
                num_bins=self.num_abundance_bins,
                min_val=min_abundance,
                max_val=1.0
            )

        # 总 bin 数 = 真实 bin 数 + 2（PAD=0, MASK=1）
        self.total_abundance_bins = self.num_abundance_bins + 2

        self.pad_taxon_id = 0   # 0=PAD（序列填充），不对应任何真实 taxon
        # taxon ID 约定：0=PAD，1=UNK（genus 无注释），2~=真实 genus
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
            num_bins=self.num_abundance_bins
        )
        return bins + 2

    def _bin_abundance_rank(self, values: np.ndarray) -> np.ndarray:
        # 使用相对丰度 rank 分 bin
        bins = bin_values_rank(
            num_items=len(values),
            num_bins=self.num_abundance_bins
        )
        return bins + 2

    @staticmethod
    def _make_abundance_values(
        transform: str,
        *,
        log_vals: np.ndarray,
        vals: np.ndarray,
        mu: float,
        sigma: float,
        n_full: int,
    ) -> np.ndarray:
        if transform in ("rclr_sigma", "rclr"):
            centered = log_vals - mu                       # present-only CLR（去 σ 即 rclr）
            if transform == "rclr":
                return centered.astype(np.float32)
            if sigma < 1e-6:                               # 单 taxon 等极端 → 兜底防除零
                return np.zeros_like(log_vals, dtype=np.float32)
            return (centered / (sigma + 1e-8)).astype(np.float32)
        if transform == "rank":
            # present 内降序排名归一（分母 n_full、保截断不变尺度）；丰度越高越接近 1
            kept = log_vals.shape[0]
            return ((n_full - np.arange(kept)) / float(n_full)).astype(np.float32)
        if transform == "presence":                        # 丢量级，全 1（MLM 退化，慎用）
            return np.ones_like(log_vals, dtype=np.float32)
        return vals.astype(np.float32)                      # raw：相对丰度原值

    @staticmethod
    def _fill_full_abundance_targets(
        idx: np.ndarray,
        vals: np.ndarray,
        *,
        n_taxa: int,
    ) -> Dict[str, np.ndarray]:
        raw = np.zeros((n_taxa,), dtype=np.float32)
        rclr = np.zeros((n_taxa,), dtype=np.float32)
        rank = np.zeros((n_taxa,), dtype=np.float32)
        if idx.size == 0:
            return {"raw": raw, "rclr_sigma": rclr, "rank": rank}

        vals = vals.astype(np.float32, copy=False)
        order = np.argsort(-vals)
        idx_sorted = idx[order]
        vals_sorted = vals[order]

        raw[idx_sorted] = vals_sorted
        log_vals = np.log(vals_sorted + np.float32(1e-10))
        mu = float(log_vals.mean())
        sigma = float(log_vals.std())
        centered = log_vals - mu
        if sigma >= 1e-6:
            rclr[idx_sorted] = (centered / (sigma + 1e-8)).astype(np.float32)

        n_full = int(idx_sorted.shape[0])
        rank[idx_sorted] = ((n_full - np.arange(n_full)) / float(n_full)).astype(np.float32)
        return {"raw": raw, "rclr_sigma": rclr, "rank": rank}

    def _make_sample_view_targets(self, sample_index: int) -> Dict[str, np.ndarray]:
        idx, vals = self._row_nonzero(sample_index)
        base = self._fill_full_abundance_targets(idx, vals, n_taxa=self.n_taxa)
        out: Dict[str, np.ndarray] = {}
        for view in self.sample_view_heads:
            if view in base:
                out[view] = base[view]
            elif view == "func_bacformer":
                if self.protein_feat is None:
                    raise RuntimeError("protein_feat not loaded for func_bacformer target.")
                out[view] = (vals.astype(np.float32, copy=False) @ self.protein_feat[idx]).astype(np.float32)
            elif view == "phylo_32coord":
                if self.sample_view_phylo_coords is None:
                    raise RuntimeError("position_encoding not loaded for phylo_32coord target.")
                out[view] = (vals.astype(np.float32, copy=False) @ self.sample_view_phylo_coords[idx]).astype(np.float32)
        return out

    def __getitem__(self, i: int) -> Dict[str, Any]:

        idx, vals = self._row_nonzero(i)

        # 处理空样本
        if idx.size == 0:
            taxon_ids = np.empty((0,), dtype=np.int64)
            abund_bins = np.empty((0,), dtype=np.int64)
            abund_values = np.empty((0,), dtype=np.float32)
            target_abund_values = np.empty((0,), dtype=np.float32)
            taxon_path_ids = np.empty((0, len(RANK_COLUMNS)), dtype=np.int64)
            # var_indices 是该 token 在 adata.var 中的行号（0~n_vars-1），
            # 用于下游按 var 索引查全局矩阵（如 varp['phylo_dist'] / varp['taxo_dist']）
            var_indices = np.empty((0,), dtype=np.int64)
        else:
            order = np.argsort(-vals)  # 按丰度值降序排列
            idx = idx[order]
            vals = vals[order]

            # V5：per-sample mu/sigma 在“截断之前”用全部非零值算
            # 这样 max_seq_len 改变时,保留 token 的 abund_values 尺度仍一致
            # 公式与 design 文档 §2.2 一致：log(vals+ε) → (x - mean)/std
            log_vals_full = np.log(vals.astype(np.float32) + 1e-10)
            mu = float(log_vals_full.mean())
            sigma = float(log_vals_full.std())
            n_full = int(log_vals_full.shape[0])  # 截断前非零数（rank 归一分母，保截断不变尺度）

            # 截断序列到最大长度
            if self.max_seq_len is not None:
                idx = idx[: self.max_seq_len]
                vals = vals[: self.max_seq_len]
                log_vals = log_vals_full[: self.max_seq_len]
            else:
                log_vals = log_vals_full

            # 两种模式统一用 Genus 列作为 taxon_ids（内容型 ID，语义稳定）
            # ID 约定：0=PAD，1=UNK（genus 无注释），2~=真实 genus
            taxon_ids = self._rank_ids[idx, _GENUS_COL_IDX]   # shape [L]
            taxon_path_ids = self._rank_ids[idx]               # shape [L, 6]
            # 携带 var 行号（绝对索引，0~n_vars-1），供下游 phylo/taxo dist 查表
            var_indices = idx.astype(np.int64)                 # shape [L]

            if self.abundance_mode == "abs_log_bins":
                abund_bins = self._bin_abundance_abs(vals).astype(np.int64)
            else:
                abund_bins = self._bin_abundance_rank(vals).astype(np.int64)

            # 2026-07 repr shaping:encoder 输入和 MLM 回归目标显式解耦。
            # 旧参数不传新字段时仍等价于 input_transform == target_transform == abundance_value_transform。
            abund_values = self._make_abundance_values(
                self.abundance_input_transform,
                log_vals=log_vals,
                vals=vals,
                mu=mu,
                sigma=sigma,
                n_full=n_full,
            )
            target_abund_values = self._make_abundance_values(
                self.abundance_target_transform,
                log_vals=log_vals,
                vals=vals,
                mu=mu,
                sigma=sigma,
                n_full=n_full,
            )

        item = {
            "taxon_ids": taxon_ids,
            "abund_bins": abund_bins,           # [L] int64 — bin 标签 / bin 输入路径用
            "abund_values": abund_values,        # [L] float32 — mlp 输入用
            "target_abund_values": target_abund_values,  # [L] float32 — huber MLM 回归标签用
            "taxon_path_ids": taxon_path_ids,    # [L, 6]
            "var_indices": var_indices,          # [L]，var 行号（0~n_vars-1）
            "length": int(taxon_ids.shape[0]),
        }
        if self.sample_view_heads:
            target_i = int(i)
            if self.sample_view_target_index_map is not None:
                target_i = int(self.sample_view_target_index_map[target_i])
            item["sample_view_targets"] = self._make_sample_view_targets(target_i)
        return item


    @property
    def genus_vocab_size(self) -> int:
        # 0=PAD, 1=UNK, 2~N=真实 genus
        return self._rank_vocab_sizes["Genus"]


    @property
    def special_ids(self) -> Dict[str, int]:
        return {
            "pad_taxon_id": self.pad_taxon_id,  # 0=PAD
            "pad_bin_id": self.pad_bin_id,       # 0=PAD
            "mask_bin_id": self.mask_bin_id,     # 1=MASK
        }

    @property
    def rank_vocab_sizes(self) -> Dict[str, int]:
        return self._rank_vocab_sizes
