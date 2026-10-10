"""Prepare fixed model features from a versioned prior archive.
从带版本号的原始先验文件生成模型固定特征。
"""

from pathlib import Path
import json

import numpy as np

from micoformer.utils.io import sha256_file


def build_prior(
    *,
    prior: Path,
    output: Path,
    expected_version: str,
    expected_sha256: str,
) -> Path:
    """Check the source, normalize features, and save the model prior.
    核对原始文件，归一化特征并保存模型先验。
    """
    prior, output = Path(prior).resolve(), Path(output).resolve()
    metadata_path = output.with_suffix(".json")
    if output.exists() or metadata_path.exists():
        raise FileExistsError(f"prior output already exists: {output}")

    with np.load(prior, allow_pickle=False) as source:
        declared = str(source["version"].item())
        if declared != expected_version:
            raise ValueError(f"expected {expected_version}, found {declared}")
        digest = sha256_file(prior)
        if digest != expected_sha256:
            raise ValueError("prior SHA256 mismatch")
        phy = source["phylogenetic_coords"].astype(np.float32)
        feat = source["protein_features"].astype(np.float32)
        valid = source["protein_valid"]
        names = source["var_names_utf8"]
        observed = source["observed_genus_ids"]

    # Center and scale PHY globally to preserve relative distances.
    # 系统发育坐标整体中心化和缩放，保留相对距离。
    centered = phy - phy.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    scale = float(np.median(norms[norms > 0]))
    phy_unit = centered / scale

    # Normalize valid protein rows; missing features stay zero.
    # 有效蛋白向量逐行归一化；缺失特征保持为零。
    prot_unit = np.zeros_like(feat)
    norms = np.linalg.norm(feat[valid], axis=1)
    prot_unit[valid] = feat[valid] / norms[:, None]

    # Keep the existing block scales and two reserved token rows.
    # 沿用原有两部分尺度，前两行留给特殊编号。
    table = np.zeros((8116, 512), dtype=np.float32)
    table[2:, :32] = phy_unit * np.sqrt(512 / 2)
    table[2:, 32:] = prot_unit * np.sqrt(512 / 2)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, pp_table=table, var_names_utf8=names, observed_genus_ids=observed
    )
    record = {
        "source": {"path": str(prior), "version": declared, "sha256": digest},
        "prior_sha256": sha256_file(output),
        "phy_scale": scale,
        "protein_valid_count": int(valid.sum()),
        "observed_genus_count": int(observed.size),
    }
    metadata_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return output
