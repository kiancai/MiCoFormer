from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import anndata as ad
import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from micoformer.data.datasets import AnnDataDataset, build_taxon_path_ids, normalize_sample_view_names
from micoformer.data.pretrain_collate import MiCoCollator
from lightning.pytorch.utilities import rank_zero_info


TAG = "[datamodule]"


# ----------------------------------------------------------------------------
# V5: EnvCategory 派生(单标签 6 类)
#
# 派生规则（design 文档 §6.2.1，单标签优先级合并）：
#   Human   = (Database == 'ResMicroDb') OR (MA_IsHuman ∈ {Human, HumanMix})
#   Animal  = MA_Env_Animal == True
#   Soil    = MA_Env_Soil   == True
#   Aquatic = MA_Env_Aquatic== True
#   Plant   = MA_Env_Plant  == True
#   Other   = 都不满足
# 优先级：Human > Animal > Soil > Aquatic > Plant > Other
# ----------------------------------------------------------------------------
ENV_CATEGORY_NAMES: List[str] = ["Human", "Animal", "Soil", "Aquatic", "Plant", "Other"]
ENV_CATEGORY_NUM_CLASSES: int = len(ENV_CATEGORY_NAMES)


def _coerce_bool_series(series: pd.Series) -> np.ndarray:
    """把可能含 NaN/string/bool 的列转成 numpy bool 数组（NaN 视为 False）。"""
    # nullable boolean → fillna(False) → numpy bool
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).to_numpy().astype(bool)
    # 尝试转 bool（True/'True'/'true'/1）
    out = np.zeros(len(series), dtype=bool)
    for i, v in enumerate(series.to_numpy()):
        if v is True or v == 1:
            out[i] = True
        elif isinstance(v, str) and v.strip().lower() == "true":
            out[i] = True
    return out


def derive_env_category(obs: pd.DataFrame) -> np.ndarray:
    """按 design §6.2.1 派生 EnvCategory 单标签（int64 数组,长度 n_obs,值 ∈ [0, 6)）。

    若 obs 缺少某些字段(如 MA_Env_* 全缺)则视为 False。
    """
    n = len(obs)
    labels = np.full(n, ENV_CATEGORY_NAMES.index("Other"), dtype=np.int64)  # 默认 Other

    database = obs["Database"].astype(str).to_numpy() if "Database" in obs.columns else np.full(n, "", dtype=object)

    def _safe_col(name: str) -> np.ndarray:
        if name in obs.columns:
            return _coerce_bool_series(obs[name])
        return np.zeros(n, dtype=bool)

    is_human_text = obs["MA_IsHuman"].astype(str).str.strip() if "MA_IsHuman" in obs.columns else pd.Series([""] * n)
    is_human_ma = is_human_text.isin(["Human", "HumanMix"]).to_numpy()
    is_rm = database == "ResMicroDb"

    is_human = is_human_ma | is_rm
    is_animal = _safe_col("MA_Env_Animal")
    is_soil = _safe_col("MA_Env_Soil")
    is_aquatic = _safe_col("MA_Env_Aquatic")
    is_plant = _safe_col("MA_Env_Plant")

    # 按优先级覆盖（从低到高最后覆盖,确保高优先级生效）
    labels[is_plant]   = ENV_CATEGORY_NAMES.index("Plant")
    labels[is_aquatic] = ENV_CATEGORY_NAMES.index("Aquatic")
    labels[is_soil]    = ENV_CATEGORY_NAMES.index("Soil")
    labels[is_animal]  = ENV_CATEGORY_NAMES.index("Animal")
    labels[is_human]   = ENV_CATEGORY_NAMES.index("Human")
    return labels


def compute_env_class_weights(env_labels: np.ndarray, num_classes: int = ENV_CATEGORY_NUM_CLASSES) -> np.ndarray:
    """sqrt 平滑的 class weight: w_c = (1/freq_c)^0.5, 归一化到均值 1。

    频率为 0 的类(数据集内缺失)权重设为 0,避免 inf。
    """
    freq = np.bincount(env_labels, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = freq > 0
    weights[nonzero] = (1.0 / freq[nonzero]) ** 0.5
    if weights.sum() > 0:
        weights = (weights / weights[nonzero].mean()).astype(np.float32)
    return weights


class _EnvLabelWrappedSubset(torch.utils.data.Dataset):
    """包装 Subset，在 __getitem__ 中附加 env_label，避免改 AnnDataDataset 本身。"""

    def __init__(self, base_subset: Subset, env_labels: np.ndarray) -> None:
        # base_subset.indices 是相对 base_dataset 的全局 obs 行号
        self.base = base_subset
        self.env_labels = env_labels

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.base[i]
        global_idx = int(self.base.indices[i])
        item["env_label"] = int(self.env_labels[global_idx])
        return item


def derive_study_id(project_id_series: pd.Series, min_size: int) -> Tuple[np.ndarray, int]:
    """Project_ID → 整数 study_id(去批次用)。

    样本数 >= min_size 的 study 各占一个 id(1..K);缺失 / 小尾巴 study → 0=UNK。
    (实测全语料 Project_ID 0% 缺失、36k study、>=64 的 6k study 覆盖 80%;小尾巴并 UNK,
     既缩小条件 MLM 的 study 表、又避免给 <min_size 的 study 学不可靠的 embedding。)
    返回 (study_ids[N] int64, n_studies = K+1)。
    """
    s = project_id_series.astype(str)
    bad = s.isin(["nan", "None", "", "NA", "<NA>"])
    vc = s[~bad].value_counts()
    big = sorted(vc[vc >= int(min_size)].index.tolist())
    mapping = {pid: i + 1 for i, pid in enumerate(big)}        # 1..K;0 留给 UNK
    ids = s.map(mapping).to_numpy()
    ids = np.where(np.isnan(ids.astype(np.float64)), 0, ids).astype(np.int64)
    return ids, len(big) + 1


class StudyBalancedBatchSampler(torch.utils.data.Sampler):
    """CONCORD 式 study-balanced 批采样:每个 batch 主要来自同一个 study(study_id>0),
    逼对比损失只能靠生物差异区分、把 study 当 nuisance 对比掉;study_id==0(UNK/小 study)
    的位置走随机混批(它们 <min_size、无法干净同-study,占比小)。每次 __iter__ 重洗。

    study_per_pos: [N_train] 每个 train 位置(0..N_train-1)的 study_id。
    """

    def __init__(self, study_per_pos, batch_size: int, seed: int = 0, min_batch: int = 2):
        self.study = np.asarray(study_per_pos, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.min_batch = int(min_batch)
        self._epoch = 0
        self._groups = [np.where(self.study == s)[0] for s in np.unique(self.study)]
        self._nbatch = sum(
            1
            for pos in self._groups
            for j in range(0, len(pos), self.batch_size)
            if len(pos[j:j + self.batch_size]) >= self.min_batch
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self._nbatch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1                                        # 每次 __iter__ 自动换种子重洗(兼容无 set_epoch)
        batches = []
        for pos in self._groups:
            p = pos.copy()
            rng.shuffle(p)
            for j in range(0, len(p), self.batch_size):
                b = p[j:j + self.batch_size]
                if len(b) >= self.min_batch:
                    batches.append(b.tolist())
        for k in rng.permutation(len(batches)):
            yield batches[k]


class _LabelWrappedSubset(torch.utils.data.Dataset):
    """包装 Subset,在 __getitem__ 按需附加 env_label / study_id,不改 AnnDataDataset。"""

    def __init__(self, base_subset: Subset, env_labels=None, study_ids=None) -> None:
        self.base_subset = base_subset
        self.env_labels = env_labels
        self.study_ids = study_ids

    def __len__(self) -> int:
        return len(self.base_subset)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        item = self.base_subset[i]
        global_idx = self.base_subset.indices[i]
        if self.env_labels is not None:
            item["env_label"] = int(self.env_labels[global_idx])
        if self.study_ids is not None:
            item["study_id"] = int(self.study_ids[global_idx])
        return item


class MiCoDataModule(L.LightningDataModule):

    def __init__(
        self,
        *,
        h5ad_path: str,
        train_indices: Optional[Sequence[int]] = None,
        val_indices: Optional[Sequence[int]] = None,
        test_indices: Optional[Sequence[int]] = None,
        batch_size: int = 32,
        num_workers: int = 4,                  # 数据加载线程数
        max_seq_len: Optional[int] = 512,
        mask_prob: float = 0.15,
        num_abundance_bins: int = 40,          # 丰度分箱数量 (不含 PAD/MASK)
        min_abundance: float = 4e-6,           # 最小丰度阈值 (低于此值归入第一箱)
        abundance_mode: str = "abs_log_bins",  # 丰度编码方式："abs_log_bins" 或 "rank_bins"
        # V5 新增 ↓
        abundance_encoding: str = "mlp",       # "mlp"(默认) | "bin"
        abundance_value_transform: str = "rclr_sigma",  # 旧参数：未显式解耦时同时控制输入和目标
        abundance_input_transform: Optional[str] = None,
        abundance_target_transform: Optional[str] = None,
        sample_view_heads: Optional[Sequence[str] | str] = None,
        sample_view_protein_feat_path: Optional[str] = None,
        shuffle_sample_targets: bool = False,
        sample_target_shuffle_seed: int = 0,
        sample_target_shuffle_manifest_path: Optional[str] = None,
        use_metadata_task: bool = True,         # 是否派生 EnvCategory 并暴露 class_weights
        metadata_cache_dir: Optional[str] = None,  # 默认与 h5ad 同目录，文件名含 h5ad fingerprint
        # 去批次(2026-06-08;默认全关=与现状等价)。study=Project_ID(唯一全覆盖批次粒度)。
        #   study_balanced         : train 用 StudyBalancedBatchSampler(每 batch 同一 study,CONCORD 对比用)
        #   use_study_conditioning : 派生 study_ids + 暴露 n_studies(条件 MLM 头用 study_embed)
        #   study_min_size         : >= 此样本数的 study 各占一 id,小尾巴并 UNK(0)
        study_balanced: bool = False,
        use_study_conditioning: bool = False,
        study_min_size: int = 64,
    ) -> None:

        super().__init__()
        self.h5ad_path = h5ad_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_seq_len = max_seq_len
        self.mask_prob = mask_prob

        # 丰度分箱配置
        self.num_abundance_bins = num_abundance_bins    # 指定的真实 bin 数（不含 PAD/MASK）
        self.min_abundance = min_abundance
        self.abundance_mode = abundance_mode
        self.abundance_encoding = abundance_encoding
        self.abundance_value_transform = abundance_value_transform
        self.abundance_input_transform = abundance_input_transform
        self.abundance_target_transform = abundance_target_transform
        self.sample_view_heads = normalize_sample_view_names(sample_view_heads)
        self.sample_view_protein_feat_path = sample_view_protein_feat_path
        self.shuffle_sample_targets = bool(shuffle_sample_targets)
        self.sample_target_shuffle_seed = int(sample_target_shuffle_seed)
        self.sample_target_shuffle_manifest_path = sample_target_shuffle_manifest_path
        self.sample_view_target_index_map: Optional[np.ndarray] = None

        self.use_metadata_task = use_metadata_task
        self.metadata_cache_dir = metadata_cache_dir

        # 去批次(study=Project_ID)。study_ids/n_studies 在 _peek_dataset_meta 里按需派生。
        self.study_balanced = study_balanced
        self.use_study_conditioning = use_study_conditioning
        self.study_min_size = study_min_size
        self.study_ids: Optional[np.ndarray] = None   # [N_obs] int64,0=UNK
        self.n_studies: int = 0

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # 下面这两个参数是不接受从外部传入的，但是保持现有就行
        self.persistent_workers = True  # 保持 worker 进程 alive，避免重复加载数据
        self.pin_memory = True          # 启用内存锁页，加速数据传输到 GPU

        self.special_ids = {
            "pad_taxon_id": 0,   # 0=PAD
            # taxon id 1 表示 UNK Taxon，但不算 special id
            "pad_bin_id": 0,     # 0=PAD
            "mask_bin_id": 1,    # 1=MASK
        }
        self.total_abundance_bins = self.num_abundance_bins + 2   # 含 PAD/MASK 的总数，传入模型

        # 距离矩阵 / PE coords：从 varp / varm 加载到 datamodule 持有
        # 模型层使用时通过 datamodule 拿,再由 workflow 注入 encoder 作为 buffer
        self.phylo_dist_matrix: Optional[torch.Tensor] = None  # [V, V] float32（patristic 连续距离）
        self.taxo_dist_matrix: Optional[torch.Tensor] = None   # [V, V] int8  （LCA 离散 hop 距离，0~6）
        # protein_w loss 用的蛋白距离矩阵(2026-05-30,镜像 phylo_dist_matrix)。
        # 不在金标准语料 varp 里,而是 workflow 从外部 .npy 路径加载后挂上来(见 workflows/pretrain.py)。
        self.protein_dist_matrix: Optional[torch.Tensor] = None  # [V_real, V_real] float32
        self.phylo_pe_coords_raw: Optional[torch.Tensor] = None  # [V_real, pe_dim] float32 (V5)
        self.pe_dim: Optional[int] = None
        # X2 phase 2:蛋白 PE coords(等 bacformer_prior 出 varm['protein_pe'])
        # 默认 None,workflow 自行判断 use_protein_pe 是否要求其存在
        self.protein_pe_coords_raw: Optional[torch.Tensor] = None  # [V_real, protein_pe_dim] float32
        self.n_obs: int = 0
        self.n_vars: int = 0

        # EnvCategory 派生结果(V5)
        self.env_labels: Optional[np.ndarray] = None        # [N_obs] int64
        self.env_class_weights: Optional[torch.Tensor] = None  # [6] float32
        self.env_class_names: List[str] = list(ENV_CATEGORY_NAMES)

        self.genus_vocab_size, self.rank_vocab_sizes = self._peek_dataset_meta()

        # 数据集占位符 (在 setup 阶段初始化)
        self.train_dataset: Optional[torch.utils.data.Dataset] = None
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.test_dataset: Optional[torch.utils.data.Dataset] = None

    def _peek_dataset_meta(self) -> Tuple[int, Dict[str, int]]:
        # 只读取 h5ad 的必要元信息，避免为了拿配置提前构建完整 dataset。
        # 同时一次性把 varp / varm / obs 关键字段 materialize 到内存
        adata = ad.read_h5ad(self.h5ad_path, backed="r")
        try:
            self.n_obs = int(adata.n_obs)
            self.n_vars = int(adata.n_vars)
            # 构建 rank 词表，两种 embedding 模式均需要；rank_mappings 此处不需要
            _, rank_vocab_sizes, _ = build_taxon_path_ids(adata.var)

            # 加载 varp 距离矩阵
            varp_keys = set(getattr(adata, "varp", {}).keys()) if hasattr(adata, "varp") else set()
            if "phylo_dist" in varp_keys:
                self.phylo_dist_matrix = torch.from_numpy(
                    np.asarray(adata.varp["phylo_dist"], dtype=np.float32)
                )
                rank_zero_info(
                    f"{TAG} Loaded varp['phylo_dist']: {tuple(self.phylo_dist_matrix.shape)} float32"
                )
            if "taxo_dist" in varp_keys:
                self.taxo_dist_matrix = torch.from_numpy(
                    np.asarray(adata.varp["taxo_dist"], dtype=np.int8)
                )
                rank_zero_info(
                    f"{TAG} Loaded varp['taxo_dist']: {tuple(self.taxo_dist_matrix.shape)} int8"
                )

            # V5: 加载 varm['position_encoding']
            varm_keys = set(getattr(adata, "varm", {}).keys()) if hasattr(adata, "varm") else set()
            if "position_encoding" in varm_keys:
                pe_arr = np.asarray(adata.varm["position_encoding"], dtype=np.float32)
                self.phylo_pe_coords_raw = torch.from_numpy(pe_arr)
                self.pe_dim = int(pe_arr.shape[1])
                rank_zero_info(
                    f"{TAG} Loaded varm['position_encoding']: {tuple(self.phylo_pe_coords_raw.shape)} float32"
                )
            # X2 phase 2: 加载 varm['protein_pe'](由 bacformer_prior pipeline 写入,可选)
            if "protein_pe" in varm_keys:
                ppe_arr = np.asarray(adata.varm["protein_pe"], dtype=np.float32)
                self.protein_pe_coords_raw = torch.from_numpy(ppe_arr)
                rank_zero_info(
                    f"{TAG} Loaded varm['protein_pe']: {tuple(self.protein_pe_coords_raw.shape)} float32"
                )

            # V5: 派生 EnvCategory + class weight(若启用)
            if self.use_metadata_task:
                # 优先从磁盘缓存读取（指纹基于 h5ad 路径与 obs.shape）
                cache_path = self._env_cache_path(adata)
                cached = self._try_load_env_cache(cache_path, expected_n=adata.n_obs)
                if cached is not None:
                    self.env_labels, weights_np = cached
                    rank_zero_info(f"{TAG} Loaded EnvCategory cache from {cache_path}")
                else:
                    # 把 backed obs 拉到内存做派生(只需要少量列)
                    needed_cols = [
                        c for c in (
                            "Database", "MA_IsHuman",
                            "MA_Env_Animal", "MA_Env_Soil", "MA_Env_Aquatic", "MA_Env_Plant",
                        ) if c in adata.obs.columns
                    ]
                    obs_subset = adata.obs[needed_cols].copy()
                    self.env_labels = derive_env_category(obs_subset)
                    weights_np = compute_env_class_weights(self.env_labels)
                    self._save_env_cache(cache_path, self.env_labels, weights_np)
                    rank_zero_info(
                        f"{TAG} Derived EnvCategory: counts={np.bincount(self.env_labels, minlength=ENV_CATEGORY_NUM_CLASSES).tolist()}, "
                        f"cached to {cache_path}"
                    )
                self.env_class_weights = torch.from_numpy(weights_np)

            # 去批次:派生 study_id(Project_ID),供条件 MLM / study-balanced 采样
            if self.study_balanced or self.use_study_conditioning:
                if "Project_ID" not in adata.obs.columns:
                    raise RuntimeError(
                        "study_balanced/use_study_conditioning=True 需要 obs['Project_ID'],但语料缺此列。"
                    )
                pid = adata.obs["Project_ID"].copy()
                self.study_ids, self.n_studies = derive_study_id(pid, self.study_min_size)
                n_unk = int((self.study_ids == 0).sum())
                rank_zero_info(
                    f"{TAG} Derived study_id (Project_ID, min_size={self.study_min_size}): "
                    f"n_studies={self.n_studies} (含 UNK=0), UNK 样本={n_unk}/{len(self.study_ids)} "
                    f"({100.0 * n_unk / max(1, len(self.study_ids)):.1f}%)"
                )
        finally:
            # 及时关闭 backed 文件句柄，避免占用文件资源
            if getattr(adata, "file", None) is not None:
                adata.file.close()
        # genus_vocab_size：Genus 词表大小（0=PAD, 1=UNK, 2~=真实 genus）
        return rank_vocab_sizes["Genus"], rank_vocab_sizes

    def _build_sample_target_index_map(self) -> Optional[np.ndarray]:
        if not (self.sample_view_heads and self.shuffle_sample_targets):
            return None
        if self.n_obs <= 0:
            raise RuntimeError("n_obs is not initialized; _peek_dataset_meta must run before shuffle map construction.")
        mapping = np.arange(self.n_obs, dtype=np.int64)
        rng = np.random.default_rng(self.sample_target_shuffle_seed)
        for split_indices in (self.train_indices, self.val_indices, self.test_indices):
            if split_indices is None:
                continue
            idx = np.asarray(split_indices, dtype=np.int64)
            if idx.size <= 1:
                continue
            perm = idx.copy()
            rng.shuffle(perm)
            fixed = np.flatnonzero(perm == idx)
            if fixed.size == idx.size:
                perm = np.roll(perm, 1)
            elif fixed.size == 1:
                swap_j = 0 if fixed[0] != 0 else 1
                perm[fixed[0]], perm[swap_j] = perm[swap_j], perm[fixed[0]]
            elif fixed.size > 1:
                perm[fixed] = np.roll(perm[fixed], 1)
            mapping[idx] = perm
        return mapping

    @staticmethod
    def _sha256_int64(values: np.ndarray) -> str:
        arr = np.asarray(values, dtype=np.int64)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def _write_sample_target_shuffle_manifest(self) -> None:
        path = self.sample_target_shuffle_manifest_path
        mapping = self.sample_view_target_index_map
        if not path or mapping is None:
            return
        if int(os.environ.get("RANK", "0")) != 0:
            return

        def _split_manifest(name: str, split_indices: Optional[Sequence[int]]) -> Dict[str, Any]:
            if split_indices is None:
                return {
                    "name": name,
                    "n": 0,
                    "present": False,
                }
            idx = np.asarray(split_indices, dtype=np.int64)
            mapped = mapping[idx].astype(np.int64, copy=False)
            idx_set = set(int(x) for x in idx.tolist())
            mapped_set = set(int(x) for x in mapped.tolist())
            preview_n = min(10, int(idx.shape[0]))
            return {
                "name": name,
                "present": True,
                "n": int(idx.shape[0]),
                "indices_sha256": self._sha256_int64(idx),
                "mapped_indices_sha256": self._sha256_int64(mapped),
                "split_internal_permutation": idx_set == mapped_set,
                "fixed_points": int(np.sum(idx == mapped)),
                "preview_pairs": [
                    [int(idx[i]), int(mapped[i])]
                    for i in range(preview_n)
                ],
            }

        manifest = {
            "shuffle_sample_targets": True,
            "sample_target_shuffle_seed": self.sample_target_shuffle_seed,
            "sample_view_heads": list(self.sample_view_heads),
            "n_obs": int(self.n_obs),
            "mapping_sha256": self._sha256_int64(mapping),
            "shuffle_axis": "sample",
            "feature_shuffle": False,
            "batch_local_shuffle": False,
            "splits": {
                "train": _split_manifest("train", self.train_indices),
                "val": _split_manifest("val", self.val_indices),
                "test": _split_manifest("test", self.test_indices),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        rank_zero_info(f"{TAG} Wrote sample-target shuffle manifest to {path}")

    def _env_cache_path(self, adata: ad.AnnData) -> str:
        # 缓存文件名含 h5ad 路径 + obs 形状的指纹,避免不同数据混用
        base_dir = self.metadata_cache_dir or os.path.dirname(os.path.abspath(self.h5ad_path)) or "."
        os.makedirs(base_dir, exist_ok=True)
        # 简单指纹:h5ad 绝对路径 + n_obs + n_vars(快速,不读 obs 内容)
        fp = hashlib.md5(
            f"{os.path.abspath(self.h5ad_path)}|{adata.n_obs}|{adata.n_vars}".encode("utf-8")
        ).hexdigest()[:12]
        return os.path.join(base_dir, f"_envcategory_cache_{fp}.npz")

    @staticmethod
    def _try_load_env_cache(path: str, expected_n: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if not os.path.exists(path):
            return None
        try:
            data = np.load(path)
            labels = data["labels"].astype(np.int64)
            weights = data["weights"].astype(np.float32)
            if labels.shape[0] != expected_n:
                return None
            return labels, weights
        except Exception:
            return None

    @staticmethod
    def _save_env_cache(path: str, labels: np.ndarray, weights: np.ndarray) -> None:
        try:
            np.savez(path, labels=labels.astype(np.int64), weights=weights.astype(np.float32))
        except Exception as e:
            rank_zero_info(f"{TAG} Failed to save EnvCategory cache {path}: {e}")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        # setup 方法在每个进程上都会被调用
        self.sample_view_target_index_map = self._build_sample_target_index_map()
        self._write_sample_target_shuffle_manifest()
        base_dataset = AnnDataDataset(
            h5ad_path=self.h5ad_path,
            max_seq_len=self.max_seq_len,
            num_abundance_bins=self.num_abundance_bins,
            min_abundance=self.min_abundance,
            abundance_mode=self.abundance_mode,
            abundance_encoding=self.abundance_encoding,
            abundance_value_transform=self.abundance_value_transform,
            abundance_input_transform=self.abundance_input_transform,
            abundance_target_transform=self.abundance_target_transform,
            sample_view_heads=self.sample_view_heads,
            protein_feat_path=self.sample_view_protein_feat_path,
            sample_view_target_index_map=self.sample_view_target_index_map,
        )

        # Subset：直接使用初始化时传入的索引划分数据集
        train_subset = Subset(base_dataset, list(self.train_indices)) if self.train_indices is not None else None
        val_subset   = Subset(base_dataset, list(self.val_indices))   if self.val_indices   is not None else None
        test_subset  = Subset(base_dataset, list(self.test_indices))  if self.test_indices  is not None else None

        # 在 __getitem__ 注入 env_label(metadata 任务)和/或 study_id(去批次)。两者按 flag 各自启用。
        _env = self.env_labels if (self.use_metadata_task and self.env_labels is not None) else None
        _study = self.study_ids if ((self.study_balanced or self.use_study_conditioning) and self.study_ids is not None) else None

        def _wrap(sub):
            if sub is None:
                return None
            if _env is None and _study is None:
                return sub
            return _LabelWrappedSubset(sub, env_labels=_env, study_ids=_study)

        self.train_dataset = _wrap(train_subset)
        self.val_dataset   = _wrap(val_subset)
        self.test_dataset  = _wrap(test_subset)

        # 打印统计信息
        stats = []
        if self.train_dataset is not None: stats.append(f"Train={len(self.train_dataset)}")
        if self.val_dataset   is not None: stats.append(f"Val={len(self.val_dataset)}")
        if self.test_dataset  is not None: stats.append(f"Test={len(self.test_dataset)}")
        if stats:
            rank_zero_info(f"{TAG} Split stats: {', '.join(stats)}")

    # DataLoaders 构建
    def _create_dataloader(self, dataset, shuffle: bool, batch_sampler=None) -> DataLoader:
        collate_function = MiCoCollator(
            pad_taxon_id=self.special_ids["pad_taxon_id"],
            pad_bin_id=self.special_ids["pad_bin_id"],
            mask_bin_id=self.special_ids["mask_bin_id"],
            mask_prob=self.mask_prob,
        )
        common = dict(
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            collate_fn=collate_function,
        )
        if batch_sampler is not None:
            # batch_sampler 与 batch_size/shuffle/sampler/drop_last 互斥
            return DataLoader(dataset, batch_sampler=batch_sampler, **common)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, **common)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not loaded (train_indices is None).")
        # 去批次:study-balanced 批采样(每 batch 同一 study);否则普通 shuffle
        if self.study_balanced and self.study_ids is not None and self.train_indices is not None:
            study_per_pos = self.study_ids[np.asarray(self.train_indices, dtype=np.int64)]
            sampler = StudyBalancedBatchSampler(study_per_pos, self.batch_size, seed=0)
            return self._create_dataloader(self.train_dataset, shuffle=False, batch_sampler=sampler)
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset is not loaded (val_indices is None).")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not loaded (test_indices is None).")
        return self._create_dataloader(self.test_dataset, shuffle=False)
