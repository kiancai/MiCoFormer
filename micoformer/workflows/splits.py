"""
    它只负责一个底层动作：
    从 `.h5ad` 的 `obs` 表里，按照若干筛选条件挑出一批样本，
    然后把这些样本对应的“全局行索引”保存为一个 `.npy` 文件。
"""

from __future__ import annotations

import os
import anndata as ad
import numpy as np


TAG = "[splits]"


# 核心数据结构，str 之间是 AND，list 之间是 OR，如：
#     [("Project_ID", ["PRJNA123456"]),
#      ("Phenotype", ["Health", "Disease"]),]
FilterSpec = list[tuple[str, list[str]]]

# 根据筛选条件，返回命中的样本索引
def _select_indices(
    obs,
    filters: FilterSpec,
) -> np.ndarray:
    
    total = len(obs)
    # 如果没有 filters，则表示“全选”
    if not filters:
        print(f"{TAG} Applying filters: (no filter) -> {total} samples")
        return np.arange(total)
    
    print(f"{TAG} Applying filters:")

    # `mask` 是布尔数组，长度等于样本总数。
    # 一开始设成全 True，后面每处理一个 filter，就用 AND 逐步剔除不符合条件的样本
    mask = np.ones(total, dtype=bool)

    # 依次遍历每个 `(field, values)`
    for i, (field, values) in enumerate(filters, start=1):
        if field not in obs.columns:
            # `field` 必须确实存在于 `obs` 里
            available = ", ".join(sorted(obs.columns))
            raise ValueError(
                f"Field '{field}' not found in obs. Available fields: {available}"
            )

        # 取出当前 filter 的允许值列表，并转成字符串
        col = np.array([str(v) for v in obs[field].to_numpy()])

        # 对比当前 filter 的允许值列表，返回一个布尔数组
        # True 表示该样本在这个字段上命中了允许值之一
        # `mask &= ...` 实现多个 filter 之间的 AND 关系
        mask &= np.isin(col, [str(v) for v in values])

        # `mask.sum()` 对布尔数组求和，得到当前所剩样本数
        survived = int(mask.sum())

        # 整理输出日志
        values_str = ",".join(str(v) for v in values)
        print(
            f"{TAG}   [{i}] {field}={values_str:<32} -> "
            f"{survived} / {total} survive"
        )

    # 返回所有 True 位置索引
    return np.where(mask)[0]


# 按 filters 从 `h5ad.obs` 里选择样本，并导出单个 `.npy` 索引文件
def make_split(
    h5ad: str,
    filters: FilterSpec,
    output: str,
) -> np.ndarray:
    
    print(f"{TAG} Reading obs from {h5ad} ...")
    adata = ad.read_h5ad(h5ad, backed="r")  # `backed="r"` 表示只读

    try:
        # 得到命中样本的索引
        indices = _select_indices(adata.obs, filters)
    finally:
        # 无论筛选的执行结果，都关闭底层文件。
        if adata.file is not None:
            adata.file.close()

    print(f"{TAG} Final: {len(indices)} samples")

    if len(indices) == 0:
        raise ValueError("No samples matched the given filters.")

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    np.save(output, indices)
    print(f"{TAG} Saved {len(indices)} indices -> {output}")

    return indices
