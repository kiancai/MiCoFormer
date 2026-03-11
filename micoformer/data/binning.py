import numpy as np


def compute_log_bin_edges(
    num_bins: int,
    min_val: float,
    max_val: float
) -> np.ndarray:
    # 提供需要分箱的数值范围和分 bin 数目，计算对数分 bin edges
    # 输入的数值范围必须是正数，且 min_val 必须小于 max_val
    # 输出的 bin edges 包含 num_bins + 1 个值，分别对应 num_bins 个 bin 边界

    if num_bins <= 0:
        raise ValueError(f"num_bins must be > 0, got {num_bins}")
    if min_val <= 0 or max_val <= 0 or min_val >= max_val:
        raise ValueError(f"min_val must be < max_val, got min_val={min_val}, max_val={max_val}")

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
    if num_bins <= 0:
        raise ValueError(f"num_bins must be > 0, got {num_bins}")
    if min_val <= 0 or max_val <= 0 or min_val >= max_val:
        raise ValueError(f"min_val must be < max_val, got min_val={min_val}, max_val={max_val}")
    # 如果 num_items 为 0，返回空数组。应该不会出现这种情况
    if values.size == 0:
        return np.array([], dtype=np.int64)

    # 将数值映射到对数分 bin ID
    # clip 将超出范围的数值映射到 min_val 或 max_val
    values_clipped = np.clip(values, min_val, max_val)
    # digitize 将数值映射到 bin ID，范围为 [1, num_bins + 1]
    bins = np.digitize(values_clipped, edges, right=False)
    # 减 1 后范围为 [0, num_bins]
    bins = bins - 1
    # 保险起见，将超出范围的 bin ID 映射到 0 或 num_bins - 1，基本不会出现这种情况
    bins = np.clip(bins, 0, num_bins - 1)
    return bins


def bin_values_rank(
    num_items: int,
    num_bins: int
) -> np.ndarray:
    # 给出 items 数量与 bins 数量，返回每个 item 所属的 bin ID
    # 每个 bin 包含的 item 数量大致相同

    if num_bins <= 0:
        raise ValueError(f"num_bins must be > 0, got {num_bins}")
    # 如果 num_items 为 0，返回空数组。应该不会出现这种情况
    if num_items == 0:
        return np.array([], dtype=np.int64)
    # 如果只有一个菌，直接分配到最高的 bin 中
    if num_items == 1:
        return np.array([num_bins - 1], dtype=np.int64)
    
    ranks = np.arange(num_items)
    normalized_pos = ranks / (num_items - 1)    # 归一化到 [0, 1]
    target_bin_float = (num_bins - 1) * (1.0 - normalized_pos)
    bins = np.round(target_bin_float).astype(np.int64)  # 四舍五入到最近的整数
    bins = np.clip(bins, 0, num_bins - 1)
    return bins
