"""Generate derived label fields for V5 finetune stages.

Reads V2 corpus, derives 12 fields:
  - IsExternalControl              (bool)
  - BroadFinetune_eligible         (bool)
  - Role_<disease>                 (str: 'case' / 'control' / 'none') × 10

Writes a new labeled.h5ad. Original h5ad untouched.

See .claude/rules/micoformer/current/finetune_plan.md §6 for full spec.

Usage:
    # 先 dry-run 验证 case/control 数对得上 §3.2
    python MiCoFormer/scripts/_prepare_labels.py --dry_run

    # 正式产出
    python MiCoFormer/scripts/_prepare_labels.py
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import anndata as ad
import numpy as np
import pandas as pd


# §3.2 disease merge rules (exact RM_Phenotype strings as they appear in V2 corpus)
DISEASE_MERGE: Dict[str, List[str]] = {
    "COVID":     ["COVID-19"],
    "Asthma":    ["Asthma", "Atopic Asthma"],
    "CRS":       [
        "Chronic Rhinosinusitis",
        "Chronic Rhinosinusitis with Nasal Polyps",
        "Chronic Rhinosinusitis without Nasal Polyps",
    ],
    "COPD":      ["Chronic Obstructive Pulmonary Disease"],
    "RSV":       ["Respiratory Syncytial Virus Infection"],
    "TB":        ["Tuberculosis"],            # §3.3: 不合并 Latent TB / Primary Pulmonary TB
    "HIV":       ["HIV Infection"],
    "Influenza": ["Influenza A Virus", "Influenza B Virus", "Influenza"],
    "RTI":       ["Respiratory Tract Infectious Disorder"],  # §3.3: 不合并 LRTI / RRI
    "LatentTB":  ["Latent Tuberculosis Infection"],
}


def derive_labels(
    obs: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Set[str]], Set[str]]:
    """按 finetune_plan.md §6.2 步骤派生 12 个新字段。

    返回:
      new_obs: 含 12 个派生字段的 obs 副本
      cc_studies_per_disease: {disease: set of CC Project_IDs}
      broad_excluded: 全部 10 个 disease CC studies 的并集
    """
    obs = obs.copy()

    # ---- Step 1: IsExternalControl ----
    ext_site = obs["RM_Sample_Site"].isin(["Negative Control", "Positive Control"])
    ext_type = obs["RM_Sample_Type"].isin(["Negative Control", "Positive Control"])
    obs["IsExternalControl"] = (ext_site | ext_type).fillna(False).astype(bool)

    # ---- Step 2: ambiguous Control 中间量 ----
    # 注: 宽松 control 定义下,Pheno='Control' 且在 CC study 内的样本仍作为 control 使用
    # 只有"不属于任何 CC study"的 Pheno='Control' 样本才会被真正排除(见 Step 5)
    is_pheno_control = (
        (obs["RM_Phenotype"] == "Control")
        & obs["RM_Is_Healthy"].isna()
        & ~obs["IsExternalControl"]
    )

    # 严格 control: Is_Healthy=True(Int64/BooleanDtype → 干净的 bool Series)
    healthy_true = (
        obs["RM_Is_Healthy"].fillna(False).astype(bool)
        & ~obs["IsExternalControl"]
    )

    # ---- Step 3: per-disease Role_<d>(宽松 control 定义) ----
    # 见 finetune_plan.md §6.2 + §2.1: control = (Is_Healthy=True) OR (Pheno='Control' 且在该疾病 CC study 内)
    cc_studies_per_disease: Dict[str, Set[str]] = {}
    role_cols: Dict[str, pd.Series] = {}

    is_any_control = healthy_true | is_pheno_control

    for disease, merge_set in DISEASE_MERGE.items():
        # 3.1: 该疾病的候选 case 行
        is_case = (
            obs["RM_Phenotype"].isin(merge_set)
            & (obs["RM_Case_Or_Control"] == "case")
            & ~obs["IsExternalControl"]
        )

        # 3.2: 找 CC studies = "既有该疾病 case 又有任意 control(严格 OR 宽松)"的 study
        studies_with_case = set(
            obs.loc[is_case, "Project_ID"].dropna().unique()
        )
        studies_with_any_control = set(
            obs.loc[is_any_control, "Project_ID"].dropna().unique()
        )
        cc_studies = studies_with_case & studies_with_any_control
        cc_studies_per_disease[disease] = cc_studies

        # 3.3 + 3.4: 在 CC studies 内分配 case / control
        in_cc_study = obs["Project_ID"].isin(cc_studies)
        role = pd.Series("none", index=obs.index, dtype="object")
        role[in_cc_study & is_case] = "case"
        role[in_cc_study & is_any_control] = "control"
        role_cols[f"Role_{disease}"] = role

    for col, val in role_cols.items():
        obs[col] = val

    # ---- Step 4: broad_excluded_studies = 10 disease CC studies 的并集 ----
    broad_excluded: Set[str] = set().union(*cc_studies_per_disease.values())

    # ---- Step 5: BroadFinetune_eligible ----
    # 注: 二分类预测 RM_Is_Healthy ∈ {True, False},Is_Healthy=NaN 的 Pheno='Control' 样本本就
    # 进不来(.notna() 过滤),所以无须显式排除"歧义 Control"
    obs["BroadFinetune_eligible"] = (
        obs["RM_Is_Healthy"].notna()
        & ~obs["IsExternalControl"]
        & ~obs["Project_ID"].isin(broad_excluded)
    ).astype(bool)

    return obs, cc_studies_per_disease, broad_excluded


def sanity_print(
    obs: pd.DataFrame,
    cc_studies_per_disease: Dict[str, Set[str]],
    broad_excluded: Set[str],
) -> None:
    """打印各疾病 CC studies + case/control 数,与 finetune_plan.md §3.2 对照。"""
    print()
    print("=" * 80)
    print("Sanity Check (对照 finetune_plan.md §3.2)")
    print("=" * 80)
    print(f"Total samples:                  {len(obs):>10,}")
    print(f"IsExternalControl:              {obs['IsExternalControl'].sum():>10,}  (expected: 2,026)")
    print(f"BroadFinetune_eligible:         {obs['BroadFinetune_eligible'].sum():>10,}")
    print(f"broad_excluded_studies (count): {len(broad_excluded):>10}  (expected: ~52)")
    print()
    print("--- per-disease (expected from §3.2) ---")
    expected = {
        "COVID":     ("13", "1,890", "640"),
        "Asthma":    ("12", "721",   "350"),
        "CRS":       ("9",  "502",   "225"),
        "COPD":      ("9",  "765",   "555"),
        "RSV":       ("5",  "900",   "1,164"),
        "TB":        ("5",  "290",   "172"),
        "HIV":       ("4",  "141",   "218"),
        "Influenza": ("3",  "574",   "398"),
        "RTI":       ("3",  "896",   "1,788"),
        "LatentTB":  ("3",  "89",    "85"),
    }
    print(f"{'disease':<11s}  {'CC':>4s}  {'case':>8s}  {'ctrl':>8s}    {'CC_exp':>6s}  {'case_exp':>9s}  {'ctrl_exp':>9s}")
    for d in DISEASE_MERGE.keys():
        col = f"Role_{d}"
        n_case = int((obs[col] == "case").sum())
        n_ctrl = int((obs[col] == "control").sum())
        n_cc = len(cc_studies_per_disease[d])
        cc_e, case_e, ctrl_e = expected[d]
        print(f"{d:<11s}  {n_cc:>4d}  {n_case:>8,}  {n_ctrl:>8,}    {cc_e:>6s}  {case_e:>9s}  {ctrl_e:>9s}")

    print()
    print("--- broad pool breakdown ---")
    elig = obs[obs["BroadFinetune_eligible"]]
    n_healthy = (elig["RM_Is_Healthy"].fillna(False).astype(bool)).sum()
    n_diseased = (~elig["RM_Is_Healthy"].fillna(True).astype(bool)).sum()
    print(f"Total eligible:                 {len(elig):>10,}  (expected: ~63,800)")
    print(f"  RM_Is_Healthy=True:           {n_healthy:>10,}  (expected: ~29,000)")
    print(f"  RM_Is_Healthy=False:          {n_diseased:>10,}  (expected: ~34,000)")
    print()
    print("--- MA samples eligibility ---")
    ma = obs[obs["Database"] == "MicrobeAtlas"]
    n_ma_elig = ma["BroadFinetune_eligible"].sum()
    print(f"MA samples eligible:            {n_ma_elig:>10,}  (expected: 0 — MA has Is_Healthy=NaN)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input",  default="data/gg2/MCFCorpusV2.gg2.h5ad")
    p.add_argument("--output", default="data/gg2/MCFCorpusV2.gg2.labeled.h5ad")
    p.add_argument("--dry_run", action="store_true",
                   help="Only compute and print sanity check, do not write output h5ad")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    t0 = time.time()
    print(f"[1/3] Reading {in_path} ...")
    # dry-run 只读 obs(快);正式跑读全 anndata 才能写回 X
    if args.dry_run:
        adata = ad.read_h5ad(in_path, backed="r")
        obs_df = adata.obs.copy()
        adata.file.close()
    else:
        adata = ad.read_h5ad(in_path)
        obs_df = adata.obs
    print(f"   Shape: {adata.shape if not args.dry_run else (len(obs_df), '-')}  "
          f"({time.time() - t0:.1f}s)")

    t1 = time.time()
    print(f"[2/3] Deriving labels ...")
    new_obs, cc_studies, broad_excluded = derive_labels(obs_df)
    print(f"   Done ({time.time() - t1:.1f}s)")

    sanity_print(new_obs, cc_studies, broad_excluded)

    if args.dry_run:
        print(f"\n[dry_run] Output not written. Total: {time.time() - t0:.1f}s")
        return

    print(f"\n[3/3] Writing {out_path} ...")
    adata.obs = new_obs
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t2 = time.time()
    adata.write_h5ad(out_path)
    print(f"   Done ({time.time() - t2:.1f}s)")
    sz_gb = out_path.stat().st_size / 1e9
    print(f"   Size: {sz_gb:.2f} GB")
    print(f"\nTotal: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
