"""Distance-preservation 辅助损失(tree loss),用于让 token 表征显式编码 phylo 距离。

机制(文献依据):
  - Phyla(bioRxiv 2025):pair-wise tree loss——sequence embedding 两两距离 ≈ phylo tree 距离
  - Micro16S(bioRxiv 2026):专做 16S 微生物组,triplet + pair loss 组合
  - NeuroSEED(NeurIPS 2021):distance-preservation 在生物序列上普遍有效
  - JEPA auxiliary tasks(NeurIPS 2024):理论证明 auxiliary regression 让表征"非等价输入映到不同表征"

为什么 MLM 单独不够:MLM 让"用上下文猜被遮 abundance",
  ① 共现近亲 abundance 同步时模型不需要 phylo(从上下文猜得到)
  ② 罕见菌/未见菌时模型没被教过"用 phylo 坐标推断",任务设计上 phylo 没用武之地
  → PE.scale 学到 ~0.02,phylo 路径被门控关掉。
tree loss 提供绕不开的 phylo 信号 reward:loss 表面强制包含 phylo 距离,
  forcing PE.scale 必须起来。

实测(tmp/20260528_phylo_task_redesign/results/weights_variant_C.csv,1500 step):
  - PE.scale +8.5% net(vs A 基线 -1.3%)、全程 16/16 step 高于基线
  - L_pair 0.67→0.55(-18%)真在降
  - L_mlm 稳定 0.12-0.15,主任务未崩
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeLossHelper(nn.Module):
    """Distance-preservation 辅助损失(pair + triplet)。

    跟普通 nn.Module 类似但**不持有参数**——只持有 phylo_dist 引用(非 Parameter)+ 几个超参。
    放成 nn.Module 是为了 .to(device) 自动迁移,不是为了 state_dict 持久化。

    Args:
        n_pairs:      每 forward 抽多少 pair 算 L_pair (随机 token 对,跨 sample)
        n_triplets:   每 forward 抽多少 triplet (anchor + phylo 近邻 + phylo 远端)
        margin:       triplet margin (||h_a - h_p|| + margin < ||h_a - h_n||)
        log_max:      log1p(phylo_dist.max()) 归一化常数;由 set_phylo_dist 注入
    """

    def __init__(
        self,
        n_pairs: int = 256,
        n_triplets: int = 128,
        margin: float = 0.5,
    ) -> None:
        super().__init__()
        self.n_pairs = int(n_pairs)
        self.n_triplets = int(n_triplets)
        self.margin = float(margin)
        # phylo_dist 注册成 non-persistent buffer:
        #   - non-persistent → 不进 ckpt(避免 263MB 双份)
        #   - 仍受 nn.Module .to(device) 自动迁移 → 跟 helper 一起上 GPU
        # 占位初始化:set_phylo_dist 调用前 _phylo_dist_loaded=False, forward raise
        self.register_buffer(
            "_phylo_dist",
            torch.zeros(1, 1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_log_max",
            torch.zeros((), dtype=torch.float32),
            persistent=False,
        )
        self._phylo_dist_loaded = False

    def set_phylo_dist(self, phylo_dist: torch.Tensor) -> None:
        """注入 phylo 距离矩阵 [V, V] float32。

        workflow 在 inject_var_buffers 之后调用本方法;helper 持有的是
        non-persistent buffer,model.to(device) 时会跟着移到 GPU。
        """
        if phylo_dist.dtype != torch.float32:
            phylo_dist = phylo_dist.float()
        # 用 register_buffer 重新注册替换占位,保持 persistent=False
        self.register_buffer("_phylo_dist", phylo_dist, persistent=False)
        self.register_buffer("_log_max", torch.log1p(phylo_dist.max()), persistent=False)
        self._phylo_dist_loaded = True

    def forward(
        self,
        h: torch.Tensor,                   # [B, L, d_model] encoder token 表征
        var_indices: torch.Tensor,         # [B, L] long, var 行号(pad 位置 = 0)
        attention_mask: torch.Tensor,      # [B, L] bool, True = valid
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict:
            loss_pair, loss_triplet:可加到主 loss 的标量
            triplet_violation_rate, d_ap_mean, d_an_mean:诊断数值(detached)
        """
        if not self._phylo_dist_loaded:
            raise RuntimeError(
                "TreeLossHelper.set_phylo_dist() has not been called. "
                "Workflow must inject phylo_dist before training."
            )

        B, L, D = h.shape
        device = h.device

        # 摊平到 valid 位置
        valid_flat = attention_mask.reshape(-1)              # [B*L]
        h_flat = h.reshape(B * L, D)
        var_flat = var_indices.reshape(-1)

        valid_idx = valid_flat.nonzero(as_tuple=True)[0]     # [N_valid]
        N_valid = valid_idx.size(0)
        if N_valid < 16:
            zero = torch.zeros((), device=device)
            return {
                "loss_pair": zero,
                "loss_triplet": zero,
                "triplet_violation_rate": zero.detach(),
                "d_ap_mean": zero.detach(),
                "d_an_mean": zero.detach(),
            }

        h_v = h_flat[valid_idx]                              # [N_valid, D]
        var_v = var_flat[valid_idx]                          # [N_valid]

        # ====== Pair loss(cosine distance vs normalized log1p(phylo dist)) ======
        idx1 = torch.randint(0, N_valid, (self.n_pairs,), device=device)
        idx2 = torch.randint(0, N_valid, (self.n_pairs,), device=device)
        keep = idx1 != idx2
        idx1, idx2 = idx1[keep], idx2[keep]
        h_i = h_v[idx1]
        h_j = h_v[idx2]
        var_i = var_v[idx1]
        var_j = var_v[idx2]

        # 表征 cosine 距离 [0, 2]
        h_i_n = F.normalize(h_i, dim=-1)
        h_j_n = F.normalize(h_j, dim=-1)
        repr_dist = 1.0 - (h_i_n * h_j_n).sum(dim=-1)

        # phylo 距离归一化到 [0, 2](与 cosine 范围对齐)
        phylo_d = self._phylo_dist[var_i, var_j]
        phylo_target = (torch.log1p(phylo_d) / self._log_max) * 2.0

        loss_pair = F.mse_loss(repr_dist, phylo_target)

        # ====== Triplet loss ======
        anc_idx = torch.randint(0, N_valid, (self.n_triplets,), device=device)
        anc_var = var_v[anc_idx]
        anc_h = h_v[anc_idx]

        # 对每个 anchor,算它与 batch 内所有 valid var 的 phylo 距离
        # [n_triplets, N_valid] —— 显存压力:n_triplets × N_valid × 4 bytes,
        # n_triplets=128, N_valid~1k-10k → 512KB-5MB,可控
        all_d = self._phylo_dist[anc_var[:, None], var_v[None, :]]
        self_mask = anc_var[:, None] == var_v[None, :]
        # positive = 最近(排除 self)
        d_for_pos = all_d.masked_fill(self_mask, float("inf"))
        pos_idx = d_for_pos.argmin(dim=1)
        # negative = 最远
        d_for_neg = all_d.masked_fill(self_mask, 0.0)
        neg_idx = d_for_neg.argmax(dim=1)

        pos_h = h_v[pos_idx]
        neg_h = h_v[neg_idx]

        d_ap = (anc_h - pos_h).norm(dim=-1)
        d_an = (anc_h - neg_h).norm(dim=-1)

        violations = (d_ap + self.margin) > d_an
        loss_triplet = F.relu(d_ap - d_an + self.margin).mean()

        return {
            "loss_pair": loss_pair,
            "loss_triplet": loss_triplet,
            "triplet_violation_rate": violations.float().mean().detach(),
            "d_ap_mean": d_ap.mean().detach(),
            "d_an_mean": d_an.mean().detach(),
        }
