"""Generate all splits needed for V5 overnight pipeline test.

Reads `data/gg2/MCFCorpusV2.gg2.labeled.h5ad` (产出于 _prepare_labels.py),
按 finetune_plan.md 的设计生成所有 .npy 索引文件.

Outputs(默认 --out_dir = data/gg2/splits/):
  pretrain_ma_train.npy / pretrain_ma_val.npy           ← Stage 1
  pretrain_rm_train.npy / pretrain_rm_val.npy           ← Stage 2
  broad_train.npy / broad_val.npy / broad_test.npy      ← Stage 3
  cc_loo/{disease}/fold_{i}_{train,test}.npy            ← Stage 4 (每 fold 一对)
  cc_loo/{disease}/manifest.json                        ← fold 元信息

按 finetune_plan.md:
  pretrain_ma: MA 子集按 Project_ID group split 95/5(val 全是没见过的 study,避免泄漏)
  pretrain_rm: RM 子集(排除 external control)随机 95/5
  broad     : BroadFinetune_eligible=True,按 Project_ID group split 80/10/10
  cc_loo    : 每个 disease 每个 CC study 一个 fold;test=该 study,train=其他 CC studies

Usage:
  python MiCoFormer/scripts/_prepare_finetune_splits.py
  python MiCoFormer/scripts/_prepare_finetune_splits.py --diseases COPD Asthma TB RSV --seed 42
  # 只重新生成某些阶段(避免覆盖已就绪的 broad/cc_loo):
  python MiCoFormer/scripts/_prepare_finetune_splits.py --stages pretrain_ma
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# 默认产出 4 个 disease 的 CC LOO splits;其他 disease 用 --diseases 加
DEFAULT_DISEASES = ["COPD", "Asthma", "TB", "RSV"]


def random_split(indices: np.ndarray, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """随机 split 一个索引数组为 train / val."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(indices))
    n_val = int(round(len(indices) * val_frac))
    val_idx = indices[perm[:n_val]]
    train_idx = indices[perm[n_val:]]
    return np.sort(train_idx), np.sort(val_idx)


def group_split(
    indices: np.ndarray,
    groups: np.ndarray,
    val_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """按 group 切 train/val(two-way),group 不重叠 → val 全是没见过的 study。

    用于 pretrain_ma:val/loss 反映跨 study 泛化,避免 train/val 同 study 泄漏。
    返回两个 sorted np.ndarray。
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr_idx, vl_idx = next(gss.split(indices, groups=groups))
    return np.sort(indices[tr_idx]), np.sort(indices[vl_idx])


def group_split_three(
    indices: np.ndarray,
    groups: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按 group 切 train/val/test,group 不重叠.
    返回三个 sorted np.ndarray.
    """
    # 先切 test 出来
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    trv_idx, tst_idx = next(gss1.split(indices, groups=groups))
    trv_indices = indices[trv_idx]
    trv_groups = groups[trv_idx]
    test_indices = indices[tst_idx]

    # 再从 trv 切 val
    val_frac_in_trv = val_frac / (1.0 - test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_in_trv, random_state=seed + 1)
    tr_idx, vl_idx = next(gss2.split(trv_indices, groups=trv_groups))
    train_indices = trv_indices[tr_idx]
    val_indices = trv_indices[vl_idx]

    return np.sort(train_indices), np.sort(val_indices), np.sort(test_indices)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--h5ad", default="data/gg2/MCFCorpusV2.gg2.labeled.h5ad")
    p.add_argument("--out_dir", default="data/gg2/splits")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diseases", nargs="+", default=DEFAULT_DISEASES,
                   help="哪些 disease 生成 CC LOO splits")
    p.add_argument("--stages", nargs="+",
                   default=["pretrain_ma", "pretrain_rm", "broad", "cc_loo"],
                   choices=["pretrain_ma", "pretrain_rm", "broad", "cc_loo"],
                   help="只生成指定阶段的 splits(默认全跑)。传单个阶段可避免覆盖已就绪的其他阶段文件。")
    p.add_argument("--pretrain_val_frac", type=float, default=0.05)
    p.add_argument("--broad_val_frac",    type=float, default=0.10)
    p.add_argument("--broad_test_frac",   type=float, default=0.10)
    args = p.parse_args()

    in_path = Path(args.h5ad)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[1/5] Reading {in_path} ...")
    adata = ad.read_h5ad(in_path, backed="r")
    obs = adata.obs.copy()
    adata.file.close()
    n_total = len(obs)
    print(f"   Total samples: {n_total:,}  ({time.time() - t0:.1f}s)")

    rows = np.arange(n_total, dtype=np.int64)
    stages = set(args.stages)
    print(f"   Stages to generate: {sorted(stages)}")

    # ============= Stage 1: pretrain_ma =============
    if "pretrain_ma" in stages:
        print(f"\n[2/5] pretrain_ma splits (group by Project_ID) ...")
        ma_mask = (obs["Database"] == "MicrobeAtlas").values
        ma_idx = rows[ma_mask]
        ma_groups = obs["Project_ID"].values[ma_mask]
        # NaN study 审计(group split 要求每样本有 study;NaN 会被 astype(str) 并成单一 "nan" 组)
        n_null = int(pd.isna(ma_groups).sum())
        ma_groups = ma_groups.astype(str)
        tr, vl = group_split(ma_idx, ma_groups, args.pretrain_val_frac, args.seed)
        np.save(out_dir / "pretrain_ma_train.npy", tr)
        np.save(out_dir / "pretrain_ma_val.npy", vl)
        tr_studies = set(obs.iloc[tr]["Project_ID"].astype(str))
        vl_studies = set(obs.iloc[vl]["Project_ID"].astype(str))
        overlap = tr_studies & vl_studies
        print(f"   train={len(tr):,}  val={len(vl):,}  (val_frac={args.pretrain_val_frac})")
        print(f"   #studies: train={len(tr_studies)}  val={len(vl_studies)}  "
              f"OVERLAP={len(overlap)}  (Project_ID nulls in MA: {n_null})")
        if overlap:
            print(f"   [WARN] {len(overlap)} studies overlap train/val (expected 0 for group split)")
    else:
        print(f"\n[2/5] pretrain_ma: SKIP (not in --stages)")

    # ============= Stage 2: pretrain_rm =============
    if "pretrain_rm" in stages:
        print(f"\n[3/5] pretrain_rm splits ...")
        rm_mask = (
            (obs["Database"] == "ResMicroDb").values
            & ~obs["IsExternalControl"].values
        )
        rm_idx = rows[rm_mask]
        tr, vl = random_split(rm_idx, args.pretrain_val_frac, args.seed)
        np.save(out_dir / "pretrain_rm_train.npy", tr)
        np.save(out_dir / "pretrain_rm_val.npy", vl)
        print(f"   train={len(tr):,}  val={len(vl):,}  (val_frac={args.pretrain_val_frac})")
    else:
        print(f"\n[3/5] pretrain_rm: SKIP (not in --stages)")

    # ============= Stage 3: broad finetune =============
    if "broad" not in stages:
        print(f"\n[4/5] broad: SKIP (not in --stages)")
    else:
        _run_broad_stage(obs, rows, out_dir, args)

    # ============= Stage 4: CC LOO per disease =============
    if "cc_loo" not in stages:
        print(f"\n[5/5] cc_loo: SKIP (not in --stages)")
    else:
        _run_cc_loo_stage(obs, rows, out_dir, args)

    print(f"\nDone. {time.time() - t0:.1f}s   ->   {out_dir}")


def _run_broad_stage(obs, rows, out_dir, args) -> None:
    """Stage 3: broad finetune splits (group by Project_ID)."""
    print(f"\n[4/5] broad finetune splits (group by Project_ID) ...")
    broad_mask = obs["BroadFinetune_eligible"].values
    broad_idx = rows[broad_mask]
    broad_groups = obs["Project_ID"].values[broad_mask].astype(str)
    tr, vl, ts = group_split_three(
        broad_idx, broad_groups,
        val_frac=args.broad_val_frac,
        test_frac=args.broad_test_frac,
        seed=args.seed,
    )
    np.save(out_dir / "broad_train.npy", tr)
    np.save(out_dir / "broad_val.npy", vl)
    np.save(out_dir / "broad_test.npy", ts)
    print(f"   train={len(tr):,}  val={len(vl):,}  test={len(ts):,}")
    # 报告 train/val/test 各自的 study 数
    n_tr_studies = len(set(obs.iloc[tr]["Project_ID"]))
    n_vl_studies = len(set(obs.iloc[vl]["Project_ID"]))
    n_ts_studies = len(set(obs.iloc[ts]["Project_ID"]))
    print(f"   #studies: train={n_tr_studies}  val={n_vl_studies}  test={n_ts_studies}")
    # train/val/test 各自的 healthy/diseased 平衡
    for name, ix in [("train", tr), ("val", vl), ("test", ts)]:
        sub = obs.iloc[ix]["RM_Is_Healthy"].fillna(False).astype(bool)
        print(f"   {name:5s}: healthy={sub.sum():>6,}  diseased={(~sub).sum():>6,}")


def _run_cc_loo_stage(obs, rows, out_dir, args) -> None:
    """Stage 4: CC LOO splits per disease."""
    print(f"\n[5/5] CC LOO splits for {len(args.diseases)} disease(s) ...")
    cc_dir = out_dir / "cc_loo"
    cc_dir.mkdir(exist_ok=True)
    for disease in args.diseases:
        col = f"Role_{disease}"
        if col not in obs.columns:
            print(f"   [skip] {col} not in obs")
            continue
        # CC study list = study with case OR control role
        cc_studies = sorted(set(obs["Project_ID"][obs[col] != "none"]))
        d_dir = cc_dir / disease
        d_dir.mkdir(exist_ok=True)
        manifest = {"disease": disease, "n_cc_studies": len(cc_studies), "folds": []}

        # 每个 study 一个 fold:test=该 study, train=其他 study
        for i, test_study in enumerate(cc_studies):
            in_disease = obs[col] != "none"
            test_mask = in_disease & (obs["Project_ID"] == test_study)
            train_mask = in_disease & (obs["Project_ID"] != test_study)
            test_idx = rows[test_mask.values]
            train_idx = rows[train_mask.values]

            # 该 fold 的 case/control 平衡
            tr_case = (obs.iloc[train_idx][col] == "case").sum()
            tr_ctrl = (obs.iloc[train_idx][col] == "control").sum()
            ts_case = (obs.iloc[test_idx][col] == "case").sum()
            ts_ctrl = (obs.iloc[test_idx][col] == "control").sum()

            np.save(d_dir / f"fold_{i:02d}_train.npy", train_idx)
            np.save(d_dir / f"fold_{i:02d}_test.npy", test_idx)
            manifest["folds"].append({
                "fold": i,
                "test_study": str(test_study),
                "train_size": int(len(train_idx)),
                "test_size": int(len(test_idx)),
                "train_case": int(tr_case), "train_control": int(tr_ctrl),
                "test_case":  int(ts_case),  "test_control":  int(ts_ctrl),
            })

        (d_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"   {disease}: {len(cc_studies)} folds → {d_dir}")
        # 简明 per-fold 表
        for f in manifest["folds"]:
            print(f"     fold{f['fold']:>2d}  test={f['test_study']:<15s}  "
                  f"train(case/ctrl)={f['train_case']:>4d}/{f['train_control']:>4d}  "
                  f"test(case/ctrl)={f['test_case']:>4d}/{f['test_control']:>4d}")


if __name__ == "__main__":
    main()
