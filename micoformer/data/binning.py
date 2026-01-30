import numpy as np


def compute_log_bin_edges(
    num_bins: int,
    min_val: float,
    max_val: float
) -> np.ndarray:
    # 根据输入的相对丰度上限（1.0）、下限、分 bin 数目，计算 bin edges
    log_min = np.log10(min_val)
    log_max = np.log10(max_val)
    edges = np.logspace(log_min, log_max, num_bins + 1)
    return edges


def bin_values_log(
    values: np.ndarray, 
    edges: np.ndarray, 
    min_val: float, 
    max_val: float, 
    num_bins: int
) -> np.ndarray:
    # 将数值映射到对数分 bin ID，且可以处理超出范围的数值
    values_clipped = np.clip(values, min_val, max_val)
    bins = np.digitize(values_clipped, edges, right=False)
    bins = bins - 1
    bins = np.clip(bins, 0, num_bins - 1)
    return bins


def bin_values_rank(
    num_items: int,
    num_bins: int
) -> np.ndarray:
    # 传入 num_items 个元素，将其分配到 num_bins 个 bin 中
    if num_items == 0:
        return np.array([], dtype=np.int64)
    if num_items == 1:  # 如果只有一个菌，直接分配到最高的 bin 中
        return np.array([num_bins - 1], dtype=np.int64)
    ranks = np.arange(num_items)
    normalized_pos = ranks / (num_items - 1)
    target_bin_float = (num_bins - 1) * (1.0 - normalized_pos)
    bins = np.round(target_bin_float).astype(np.int64)
    bins = np.clip(bins, 0, num_bins - 1)
    return bins

