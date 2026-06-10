from __future__ import annotations

from typing import Any, Dict, List

import torch

from micoformer.data.pretrain_collate import (
    pad_sequences,
    pad_matrix_sequences,
    pad_float_sequences,
)


class ClassificationCollator:
    """下游分类任务的 collator：padding + 收集标签，不做 MLM masking。"""

    def __init__(self, *, pad_taxon_id: int, pad_bin_id: int) -> None:
        self.pad_taxon_id = pad_taxon_id
        self.pad_bin_id = pad_bin_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        # 将样本中的序列转为 Tensor
        taxon_seqs = [torch.as_tensor(b["taxon_ids"], dtype=torch.long) for b in batch]
        abund_seqs = [torch.as_tensor(b["abund_bins"], dtype=torch.long) for b in batch]
        # V5：连续 abund_values（若缺则零填充）
        abund_value_seqs = [
            torch.as_tensor(b.get("abund_values", torch.zeros(len(b["taxon_ids"]))), dtype=torch.float32)
            for b in batch
        ]
        # AnnDataDataset 始终返回 taxon_path_ids，无需条件检查
        taxon_path_seqs = [torch.as_tensor(b["taxon_path_ids"], dtype=torch.long) for b in batch]
        var_index_seqs = [torch.as_tensor(b["var_indices"], dtype=torch.long) for b in batch]

        # Padding
        token_ids = pad_sequences(taxon_seqs, self.pad_taxon_id)
        abund_bins = pad_sequences(abund_seqs, self.pad_bin_id)
        abund_values = pad_float_sequences(abund_value_seqs, pad_value=0.0)  # V5
        taxon_path_ids = pad_matrix_sequences(taxon_path_seqs, pad_value=0)
        var_indices = pad_sequences(var_index_seqs, pad_value=0)

        # Attention mask
        attention_mask = (token_ids != self.pad_taxon_id).to(torch.bool)

        # 收集标签（每个样本的 labels 是 dict: task_name → int）
        batch_out: Dict[str, Any] = {
            "token_ids": token_ids,
            "abund_bins": abund_bins,
            "abund_values": abund_values,        # V5
            "attention_mask": attention_mask,
            "taxon_path_ids": taxon_path_ids,
            "var_indices": var_indices,
        }

        # 汇总多任务标签
        if "labels" in batch[0]:
            task_names = list(batch[0]["labels"].keys())
            labels = {
                name: torch.tensor([b["labels"][name] for b in batch], dtype=torch.long)
                for name in task_names
            }
            batch_out["labels"] = labels

        return batch_out
