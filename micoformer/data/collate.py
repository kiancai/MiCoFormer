from __future__ import annotations

from typing import Any, Dict, List

import torch


def pad_sequences(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:

    if len(seqs) == 0:
        return torch.empty(0, dtype=torch.long)
    max_len = max(s.numel() for s in seqs)  # 找出本批次中最长的序列长度
    
    # 初始化全为 pad_value 的矩阵
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):  # 将实际数据填入
        L = s.numel()
        out[i, :L] = s
    return out


class MiCoCollator:

    def __init__(
        self,
        *,
        pad_token_id: int,
        sample_token_id: int,
        pad_bin_id: int,
        mask_bin_id: int,
        mask_prob: float = 0.15,
    ):
        self.pad_token_id = pad_token_id
        self.sample_token_id = sample_token_id
        self.pad_bin_id = pad_bin_id
        self.mask_bin_id = mask_bin_id
        self.mask_prob = mask_prob

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        # for 循环遍历这个 batch 中样本的 taxon_ids 与 abund_bins 转为 Tensor
        taxon_seqs = [torch.as_tensor(b["taxon_ids"], dtype=torch.long) for b in batch]
        abund_seqs = [torch.as_tensor(b["abund_bins"], dtype=torch.long) for b in batch]

        # 在序列头部插入 [SAMPLE] Token 与其对应的丰度占位符 pad
        sample_tok = torch.tensor([self.sample_token_id], dtype=torch.long)
        sample_bin = torch.tensor([self.pad_bin_id], dtype=torch.long)
        
        taxon_seqs = [torch.cat([sample_tok, s], dim=0) for s in taxon_seqs]
        abund_seqs = [torch.cat([sample_bin, s], dim=0) for s in abund_seqs]

        # Padding：将序列补齐到当前 Batch 的最大长度
        input_ids = pad_sequences(taxon_seqs, self.pad_token_id)
        abund_bins = pad_sequences(abund_seqs, self.pad_bin_id)

        # 构建 Attention Mask
        attention_mask = (input_ids != self.pad_token_id).to(torch.bool)

        # 只对“真实的物种位置”进行 Mask，避开 Padding 和 [SAMPLE] Token
        B, L = input_ids.shape            # B:Batch Size; L:Length;
        # 候选 Mask 区域
        valid = attention_mask.clone()  # 先复制 attention_mask (排除 Pad)
        valid[:, 0] = False             # 排除第 0 位 [SAMPLE] Token
        # 在 valid 为 True 的位置，且随机数 < mask_prob 时，才 Mask
        if self.mask_prob > 0:
            rand = torch.rand(B, L)  # 在 (B, L) 大小矩阵中生成 [0, 1) 随机数
            mask_positions = (rand < self.mask_prob) & valid
        else:                        # 只在特殊测试时才会完全不 mask 任何位置
            mask_positions = torch.zeros(B, L, dtype=torch.bool)
        
        # 复制一份原始的 abund_bins 作为标签
        labels_abund = abund_bins.clone()     

        # 应用 Mask，将被选中的位置的 abund_bins 替换为特殊的 mask_bin_id
        abund_bins = abund_bins.masked_fill(mask_positions, self.mask_bin_id)

        # 组装输出
        batch_out = {
            "input_ids": input_ids,           # [B, L]: 物种 ID 序列 (含 [SAMPLE] 和 Pad)
            "abund_bins": abund_bins,         # [B, L]: 丰度序列 (含 [MASK]、[SAMPLE]占位 和 Pad)
            "attention_mask": attention_mask, # [B, L]: 注意力掩码 (True=有效, False=Pad)
            "labels_abund": labels_abund,     # [B, L]: 真实标签 (用于计算 Loss)
            "mask_positions": mask_positions, # [B, L]: 布尔矩阵，指示哪些位置被 Mask 了
        }
        
        return batch_out
