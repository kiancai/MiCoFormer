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

def pad_matrix_sequences(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
    # 对形状 [L, D] 的变长序列进行 padding，输出 [B, L_max, D]。
    if len(seqs) == 0:
        return torch.empty(0, dtype=torch.long)
    max_len = max(s.shape[0] for s in seqs)
    width = seqs[0].shape[1]
    out = torch.full((len(seqs), max_len, width), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        L = s.shape[0]
        out[i, :L, :] = s
    return out


def pad_float_sequences(seqs: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    # 对形状 [L] 的 float 变长序列做 padding，输出 [B, L_max] float32。
    # V5 用于 abund_values（连续 abundance）。
    if len(seqs) == 0:
        return torch.empty(0, dtype=torch.float32)
    max_len = max(s.numel() for s in seqs)
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.float32)
    for i, s in enumerate(seqs):
        L = s.numel()
        out[i, :L] = s
    return out

class MiCoCollator:

    def __init__(
        self,
        *,
        pad_taxon_id: int,
        pad_bin_id: int,
        mask_bin_id: int,
        mask_prob: float = 0.15,
        ensure_one_mask_per_nonempty: bool = True,
    ):
        self.pad_taxon_id = pad_taxon_id
        self.pad_bin_id = pad_bin_id
        self.mask_bin_id = mask_bin_id
        self.mask_prob = mask_prob

        # 下面这个参数不接受从外部传入
        # 是否保证“每个非空样本至少有 1 个被 mask 的位置”
        self.ensure_one_mask_per_nonempty = ensure_one_mask_per_nonempty

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        # 将样本中的 taxon_ids 与 abund_bins 转为 Tensor
        taxon_seqs = [torch.as_tensor(b["taxon_ids"], dtype=torch.long) for b in batch]
        abund_seqs = [torch.as_tensor(b["abund_bins"], dtype=torch.long) for b in batch]
        # V5：连续 abund_values 是 encoder 输入；target_abund_values 是 Huber MLM 目标。
        # target 缺失时回退到 abund_values，兼容旧 dataset / 历史产物。
        abund_value_seqs = [
            torch.as_tensor(b.get("abund_values", torch.zeros(len(b["taxon_ids"]))), dtype=torch.float32)
            for b in batch
        ]
        target_abund_value_seqs = [
            torch.as_tensor(
                b.get("target_abund_values", b.get("abund_values", torch.zeros(len(b["taxon_ids"])))),
                dtype=torch.float32,
            )
            for b in batch
        ]
        # AnnDataDataset 始终返回 taxon_path_ids，无需条件检查
        taxon_path_seqs = [torch.as_tensor(b["taxon_path_ids"], dtype=torch.long) for b in batch]
        # var_indices：每个 token 在 adata.var 中的行号，pad 位置填 0
        # PAD 位置不会参与 attention（attention_mask 已屏蔽），具体填什么不重要
        var_index_seqs = [torch.as_tensor(b["var_indices"], dtype=torch.long) for b in batch]

        # Padding：将序列补齐到当前 Batch 的最大长度
        token_ids = pad_sequences(taxon_seqs, self.pad_taxon_id)
        abund_bins = pad_sequences(abund_seqs, self.pad_bin_id)
        abund_values = pad_float_sequences(abund_value_seqs, pad_value=0.0)  # V5
        target_abund_values = pad_float_sequences(target_abund_value_seqs, pad_value=0.0)
        taxon_path_ids = pad_matrix_sequences(taxon_path_seqs, pad_value=0)
        var_indices = pad_sequences(var_index_seqs, pad_value=0)

        # 构建 Attention Mask
        attention_mask = (token_ids != self.pad_taxon_id).to(torch.bool)

        # 只对“真实的物种位置”进行 Mask，避开 Padding
        B, L = token_ids.shape            # B:Batch Size; L:Length;
        # 候选 Mask 区域
        valid = attention_mask.clone()  # 先复制 attention_mask (排除 Pad)
        # 在 valid 为 True 的位置，且随机数 < mask_prob 时，才 Mask
        if self.mask_prob > 0:
            rand = torch.rand(B, L)  # 在 (B, L) 大小矩阵中生成 [0, 1) 随机数
            mask_positions = (rand < self.mask_prob) & valid
        else:                        # 只在特殊测试时才会完全不 mask 任何位置
            mask_positions = torch.zeros(B, L, dtype=torch.bool)

        # 可选：保证每个非空样本至少有一个监督信号，避免整条样本无 loss
        if self.ensure_one_mask_per_nonempty:
            for i in range(B):
                valid_i = torch.where(valid[i])[0]
                if valid_i.numel() == 0:
                    # 空样本（全 pad），跳过
                    continue
                if not mask_positions[i].any():
                    # 若该样本没采到 mask，则强制随机补 1 个位置
                    j = valid_i[torch.randint(0, valid_i.numel(), (1,)).item()]
                    mask_positions[i, j] = True

        # 复制一份原始的 abund_bins 作为标签
        labels_abund = abund_bins.clone()

        # 应用 Mask，将被选中的位置的 abund_bins 替换为特殊的 mask_bin_id
        abund_bins = abund_bins.masked_fill(mask_positions, self.mask_bin_id)
        # V5：abund_values 不在 collator 替换 MASK，encoder 内部用 abund_mask_token 替换。
        # labels_abund_values 来自 target_abund_values，允许只改预测目标、不改 encoder 输入。
        labels_abund_values = target_abund_values.clone()

        # 组装输出
        batch_out = {
            "token_ids": token_ids,                 # [B, L]: taxon ID 序列（含 Pad）
            "abund_bins": abund_bins,                # [B, L]: 丰度 bin 序列（含 MASK 和 Pad）— bin 路径用
            "abund_values": abund_values,            # [B, L] float32: 连续丰度（pad=0.0）— mlp 路径用
            "attention_mask": attention_mask,        # [B, L]: 注意力掩码 (True=有效, False=Pad)
            "labels_abund": labels_abund,            # [B, L]: bin 标签 (bin_ce loss 用)
            "labels_abund_values": labels_abund_values,  # [B, L] float32: 回归标签 (huber loss 用)
            "mask_positions": mask_positions,        # [B, L]: 布尔矩阵，指示哪些位置被 Mask 了
            "taxon_path_ids": taxon_path_ids,        # [B, L, 6]
            "var_indices": var_indices,              # [B, L]: var 行号（0~n_vars-1），PAD 位置为 0
        }

        # V5: metadata 多任务标签
        # 要求 batch 内"全有"或"全无":部分含 env_label 会触发 silent partial fail
        has_env = ["env_label" in b for b in batch]
        if any(has_env):
            if not all(has_env):
                missing = sum(1 for x in has_env if not x)
                raise RuntimeError(
                    f"MiCoCollator: partial env_label batch ({missing}/{len(batch)} samples missing). "
                    "All samples must consistently include env_label, or none of them. "
                    "Check DataModule wraps Subset with _EnvLabelWrappedSubset for every split."
                )
            env_labels = torch.as_tensor([int(b["env_label"]) for b in batch], dtype=torch.long)
            batch_out["env_label"] = env_labels  # [B]

        # 去批次:study_id(条件 MLM / study-balanced 对比用)。同 env_label 规则:batch 内全有或全无。
        has_study = ["study_id" in b for b in batch]
        if any(has_study):
            if not all(has_study):
                missing = sum(1 for x in has_study if not x)
                raise RuntimeError(
                    f"MiCoCollator: partial study_id batch ({missing}/{len(batch)} samples missing). "
                    "All samples must consistently include study_id, or none. "
                    "Check DataModule wraps Subset with study_ids for every split."
                )
            batch_out["study_id"] = torch.as_tensor(
                [int(b["study_id"]) for b in batch], dtype=torch.long
            )  # [B]

        return batch_out
