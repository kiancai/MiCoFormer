"""把 mock V2 anndata 写到 tmp/mock_v2.h5ad — 供 V5 sanity check / CLI 烟雾测试用。

Usage:
    python MiCoFormer/scripts/_make_mock_h5ad.py
    # 产出: tmp/mock_v2.h5ad + tmp/mock_train_indices.npy + tmp/mock_val_indices.npy
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 让脚本能 import tests 包
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
sys.path.insert(0, _REPO_ROOT)

from tests._fixtures import build_mock_v2_anndata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_obs", type=int, default=12)
    parser.add_argument("--n_vars", type=int, default=20)
    parser.add_argument("--pe_dim", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="tmp")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    h5ad_path = os.path.join(args.output_dir, "mock_v2.h5ad")
    train_path = os.path.join(args.output_dir, "mock_train_indices.npy")
    val_path = os.path.join(args.output_dir, "mock_val_indices.npy")

    adata = build_mock_v2_anndata(
        n_obs=args.n_obs,
        n_vars=args.n_vars,
        pe_dim=args.pe_dim,
        seed=args.seed,
    )
    adata.write_h5ad(h5ad_path)
    print(f"[mock] wrote {h5ad_path}: n_obs={adata.n_obs}, n_vars={adata.n_vars}")

    # 简单切分(前 70% train, 后 30% val)
    n_train = max(1, int(args.n_obs * 0.7))
    train_idx = np.arange(n_train, dtype=np.int64)
    val_idx = np.arange(n_train, args.n_obs, dtype=np.int64)
    np.save(train_path, train_idx)
    np.save(val_path, val_idx)
    print(f"[mock] wrote train={train_path} ({n_train}) / val={val_path} ({args.n_obs - n_train})")


if __name__ == "__main__":
    main()
