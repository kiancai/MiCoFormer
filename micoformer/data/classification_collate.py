from __future__ import annotations

from typing import Any, Dict, List

import torch

from micoformer.data.collate import pad_sequences, pad_matrix_sequences


class ClassificationCollator:
    """下游分类任务的 collator：padding + 收集标签，不做 MLM masking。"""

    def __init__(self, *, pad_taxon_id: int, pad_bin_id: int) -> None:
        self.pad_taxon_id = pad_taxon_id
        self.pad_bin_id = pad_bin_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:

        # 将样本中的序列转为 Tensor
        taxon_seqs = [torch.as_tensor(b["taxon_ids"], dtype=torch.long) for b in batch]
        abund_seqs = [torch.as_tensor(b["abund_bins"], dtype=torch.long) for b in batch]
        has_taxon_path = "taxon_path_ids" in batch[0]
        taxon_path_seqs = (
            [torch.as_tensor(b["taxon_path_ids"], dtype=torch.long) for b in batch]
            if has_taxon_path
            else None
        )

        # Padding
        token_ids = pad_sequences(taxon_seqs, self.pad_taxon_id)
        abund_bins = pad_sequences(abund_seqs, self.pad_bin_id)
        taxon_path_ids = (
            pad_matrix_sequences(taxon_path_seqs, pad_value=0)
            if has_taxon_path and taxon_path_seqs is not None
            else None
        )

        # Attention mask
        attention_mask = (token_ids != self.pad_taxon_id).to(torch.bool)

        # 收集标签（每个样本的 labels 是 dict: task_name → int）
        batch_out: Dict[str, Any] = {
            "token_ids": token_ids,
            "abund_bins": abund_bins,
            "attention_mask": attention_mask,
        }
        if taxon_path_ids is not None:
            batch_out["taxon_path_ids"] = taxon_path_ids

        # 汇总多任务标签
        if "labels" in batch[0]:
            task_names = list(batch[0]["labels"].keys())
            labels = {
                name: torch.tensor([b["labels"][name] for b in batch], dtype=torch.long)
                for name in task_names
            }
            batch_out["labels"] = labels

        return batch_out
