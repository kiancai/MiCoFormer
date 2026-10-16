from __future__ import annotations

from typing import Sequence
import numpy as np


def _row_rng(
    seed: int, stream: int, epoch: int, row_id: int, magic: int
) -> np.random.Generator:
    # Separate train/validation streams and task-specific randomness for each row.
    # 按语料行号建立随机流；stream 区分训练/验证，magic 区分任务。
    return np.random.default_rng(
        np.random.SeedSequence(
            [
                int(seed),
                int(stream),
                int(epoch),
                # Split the row ID into low/high 32-bit parts.
                # 将行号拆成低/高 32 位。
                int(row_id) & 0xFFFFFFFF,
                int(row_id) >> 32,
                int(magic),
            ]
        )
    )


def deterministic_mlm_mask(
    padding_mask: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    stream: int,
    epoch: int,
    probability: float,
) -> np.ndarray:
    """Select abundance-masking targets. / 选择需要遮盖丰度的 MLM 目标。"""
    # Inputs: padding [B, L], corpus row IDs [B]; output: target mask [B, L].
    # 输入为补齐标记 [B, L] 和语料行号 [B]；输出中 True 表示遮盖丰度。
    padding = np.asarray(padding_mask)
    rows = np.asarray(row_ids)
    mask = np.zeros_like(padding)
    for sample_index, row_id in enumerate(rows.tolist()):
        valid = np.flatnonzero(~padding[sample_index])
        rng = _row_rng(seed, stream, epoch, int(row_id), 0x4D4C4D)
        # Select each real token independently using the configured probability.
        # 按配置概率独立抽取真实 token，不包含 PAD。
        selected = valid[rng.random(valid.size) < probability]
        # Keep at least one target and one visible abundance.
        # 至少选择一个目标，并保留一个可见丰度。
        if not selected.size:
            selected = valid[[int(rng.integers(0, valid.size))]]
        if selected.size >= valid.size:
            selected = np.sort(selected)[: valid.size - 1]
        mask[sample_index, selected] = True
    return mask


def deterministic_query_mask(
    padding_mask: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    stream: int,
    epoch: int,
    probability: float,
) -> np.ndarray:
    """Select genera to remove and query. / 选择需要移除并预测的 Query 菌属。"""
    # Inputs: padding [B, L], corpus row IDs [B]; output: target mask [B, L].
    # 输入为补齐标记 [B, L] 和语料行号 [B]；输出中 True 表示 Query 正例。
    padding = np.asarray(padding_mask)
    rows = np.asarray(row_ids)
    mask = np.zeros_like(padding)
    for sample_index, row_id in enumerate(rows.tolist()):
        valid = np.flatnonzero(~padding[sample_index])
        rng = _row_rng(seed, stream, epoch, int(row_id), 0x5155)
        # Query selection is independent of MLM selection; targets may overlap.
        # Query 与 MLM 独立抽样，目标位置可以重叠。
        selected = valid[rng.random(valid.size) < probability]
        # Keep at least one target and two visible genera.
        # 至少选择一个目标，并保留两个可见菌属。
        if not selected.size:
            selected = valid[[int(rng.integers(0, valid.size))]]
        if selected.size > valid.size - 2:
            selected = np.sort(selected)[: valid.size - 2]
        mask[sample_index, selected] = True
    return mask


def deterministic_negative_ids(
    genus_ids: np.ndarray,
    query_mask: np.ndarray,
    full_supports: Sequence[np.ndarray],
    observed_genus_ids: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    stream: int,
    epoch: int,
) -> np.ndarray:
    """Sample one absent genus per Query target. / 为每个 Query 目标抽取一个样本中未出现的菌属。"""
    # genus_ids/query_mask: [B, L]; row_ids: [B].
    # 菌属编号和 Query 标记为 [B, L]，语料行号为 [B]。
    genus = np.asarray(genus_ids)
    query = np.asarray(query_mask)
    rows = np.asarray(row_ids)
    observed = np.asarray(observed_genus_ids)
    # Output: negative IDs [B, L], with zeros outside Query positions.
    # 输出为负例编号 [B, L]，非 Query 位置保持 0。
    result = np.zeros_like(genus)
    # Convert full, untruncated corpus column IDs to model IDs by adding 2.
    # 截取前的完整语料列号加 2，转换为模型编号。
    batch_pools = [np.asarray(support, dtype=np.int64) + 2 for support in full_supports]
    for sample_index, (support, row_id) in enumerate(zip(full_supports, rows.tolist())):
        support_array = np.asarray(support, dtype=np.int64) + 2
        # Prefer genera from other samples, excluding this sample's full support.
        # 优先使用同批其他样本的菌属，排除本样本截取前的全部真实菌属。
        other = [pool for index, pool in enumerate(batch_pools) if index != sample_index]
        primary = (
            np.unique(np.concatenate(other)) if other else np.empty(0, dtype=np.int64)
        )
        primary = primary[~np.isin(primary, support_array)]
        # The fallback pool uses all observed genera, with the same exclusion.
        # 后备池使用语料中全部已观测菌属，同样排除本样本的真实菌属。
        fallback = observed[~np.isin(observed, support_array)]
        positions = np.flatnonzero(query[sample_index])
        rng = _row_rng(seed, stream, epoch, int(row_id), 0x4E4547)
        # Switch pools when needed; sample one distinct negative per target.
        # 同批候选不足时切换至后备池；每个目标配一个负例，样本内不重复。
        # Results also depend on batch membership through the candidate pool.
        # 候选池受 batch 成员影响，因此负例也可能随组批变化。
        pool = primary if primary.size >= positions.size else fallback
        chosen = rng.choice(pool, size=positions.size, replace=False)
        result[sample_index, positions] = chosen
    return result
