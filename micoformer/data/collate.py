"""Batch samples for pretraining and unmasked embedding extraction.
组装预训练 batch，以及提取 embedding 时使用的无扰动 batch。
"""
from __future__ import annotations

from multiprocessing.sharedctypes import Synchronized
from typing import Any, Sequence
import numpy as np
import torch
from torch import Tensor
from .corruption import (
    deterministic_mlm_mask,
    deterministic_query_mask,
    deterministic_negative_ids,
)


class PretrainingCollator:
    """Pad samples and build independent MLM/Query targets.
补齐样本，生成独立的 MLM/Query 目标与负例。
"""

    def __init__(
        self,
        *,
        observed_genus_ids: np.ndarray,
        seed: int,
        stream: int,
        shared_epoch: Synchronized,
        mlm_mask_probability: float,
        query_mask_probability: float,
    ) -> None:
        # Observed genera use model IDs, with real genera starting at 2.
        # 已观测菌属使用模型编号，真实菌属从 2 开始。
        self.observed = np.asarray(observed_genus_ids, dtype=np.int64)
        self.seed = int(seed)
        # Separate train/validation streams; workers read the shared epoch.
        # stream 区分训练/验证随机流，worker 通过共享值读取当前轮数。
        self.stream = int(stream)
        self.shared_epoch = shared_epoch
        self.mlm_mask_probability = mlm_mask_probability
        self.query_mask_probability = query_mask_probability

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        # DataLoader supplies a list of variable-length CorpusDataset samples.
        # DataLoader 传入一组来自 CorpusDataset、长度可以不同的样本。
        # B is batch size; L is the longest sequence in this batch.
        # B 为本批样本数，L 为本批最长序列长度。
        batch_size = len(samples)
        max_length = max(int(item["genus_ids"].size) for item in samples)
        # IDs, abundance, and padding are [B, L]; ID 0 is PAD, padding=True ignores it.
        # 编号、丰度与补齐标记均为 [B, L]；编号 0 为 PAD，padding=True 表示忽略。
        genus = np.zeros((batch_size, max_length), dtype=np.int64)
        abundance = np.zeros((batch_size, max_length), dtype=np.float32)
        padding = np.ones((batch_size, max_length), dtype=np.bool_)
        # Metadata [B]: corpus row IDs and genus counts before truncation.
        # 元数据 [B]：语料行号和截取前的非零菌属数量。
        rows = np.asarray([item["row_id"] for item in samples], dtype=np.int64)
        richness = np.asarray([item["richness"] for item in samples], dtype=np.int64)
        # Full corpus column IDs exclude true genera from negative sampling.
        # 截取前的完整语料列号用于从负例候选中排除真实菌属。
        supports: list[np.ndarray] = []
        for index, sample in enumerate(samples):
            length = int(sample["genus_ids"].size)
            genus[index, :length] = sample["genus_ids"]
            abundance[index, :length] = sample["abundance"]
            padding[index, :length] = False
            supports.append(np.asarray(sample["full_indices"], dtype=np.int64))
        # Training updates the shared epoch; validation keeps it at 0.
        # 训练逐轮更新共享 epoch，验证保持为 0。
        epoch = self.shared_epoch.value
        # Independent bool masks [B, L]: Query removes genera; MLM hides abundance.
        # 两个布尔标记 [B, L] 独立且可重叠：Query 移除菌属，MLM 遮盖丰度。
        query = deterministic_query_mask(
            padding, rows, seed=self.seed, stream=self.stream, epoch=epoch,
            probability=self.query_mask_probability,
        )
        mlm = deterministic_mlm_mask(
            padding, rows, seed=self.seed, stream=self.stream, epoch=epoch,
            probability=self.mlm_mask_probability,
        )
        # Negative IDs [B, L]: one per Query positive, zero elsewhere.
        # 负例编号 [B, L]：每个 Query 正例对应一个负例，其余位置填 0。
        negatives = deterministic_negative_ids(
            genus,
            query,
            supports,
            self.observed,
            rows,
            seed=self.seed,
            stream=self.stream,
            epoch=epoch,
        )
        # Return inputs and targets; the model applies masking and genus removal.
        # 返回输入与任务标记，实际丰度遮盖和菌属移除由模型执行。
        return {
            "genus_ids": genus,
            "abundance": abundance,
            "padding_mask": padding,
            "query_mask": query,
            "mlm_mask": mlm,
            "negative_genus_ids": negatives,
            "row_ids": rows,
            "richness": richness,
            "corruption_epoch": np.int64(epoch),
        }


def batch_to_device(
    batch: dict[str, np.ndarray], device: torch.device
) -> dict[str, Tensor]:
    """Convert model inputs to tensors on the selected device.
将模型输入转为张量，放到指定 CPU/GPU；行号与统计信息保留在原 batch。
"""
    # Keep [B, L] shapes: integer IDs, float32 abundance, and bool masks.
    # 保持 [B, L] 形状：编号为整数，丰度为 float32，标记为 bool。
    return {
        "genus_ids": torch.as_tensor(batch["genus_ids"], device=device, dtype=torch.long),
        "abundance": torch.as_tensor(batch["abundance"], device=device, dtype=torch.float32),
        "padding_mask": torch.as_tensor(batch["padding_mask"], device=device, dtype=torch.bool),
        "query_mask": torch.as_tensor(batch["query_mask"], device=device, dtype=torch.bool),
        "mlm_mask": torch.as_tensor(batch["mlm_mask"], device=device, dtype=torch.bool),
        "negative_genus_ids": torch.as_tensor(
            batch["negative_genus_ids"], device=device, dtype=torch.long
        ),
    }


def collate_unmasked(samples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Pad samples for extraction, preserving input order and abundance.
提取时只补齐样本，保持输入顺序和丰度，不生成遮盖或负例。
"""
    # Pad to [B, L], where L is this batch's longest sequence.
    # 补齐为 [B, L]，L 为本批最长序列长度。
    length = max(len(sample["genus_ids"]) for sample in samples)
    genus = np.zeros((len(samples), length), dtype=np.int64)
    abundance = np.zeros(genus.shape, dtype=np.float32)
    padding = np.ones(genus.shape, dtype=np.bool_)
    # Fill real tokens in order; the remaining positions stay padded.
    # 按顺序填入真实菌属及丰度，其余位置保留为补齐位。
    for index, sample in enumerate(samples):
        n = len(sample["genus_ids"])
        genus[index, :n] = sample["genus_ids"]
        abundance[index, :n] = sample["abundance"]
        padding[index, :n] = False
    return {
        "genus_ids": genus,
        "abundance": abundance,
        "padding_mask": padding,
        "row_ids": np.asarray([sample["row_id"] for sample in samples], dtype=np.int64),
    }
