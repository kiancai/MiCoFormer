#!/usr/bin/env python3
"""Prepare, smoke-test, and train fixed-PP MPC with explicit UniFrac strength.

This research entrypoint retains the historical V3.1, three-GPU, B128 contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def prepare(config) -> Path:
    """Bind inputs, fixed probes, initialization, and explicit loss coefficients.
    绑定输入、固定探针、初始化权重和显式损失系数。
    """
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    from anndata.io import read_elem
    from micoformer.mpc_unifrac import workflow as w
    from micoformer.mpc_unifrac.data import FullSupportRowDataset, DualCorruptionCollator, SharedEpoch
    from micoformer.mpc_unifrac.teacher import ASSETS

    config = config.normalized()
    output = config.output_root
    if (output / "audit/sealed_run_manifest.json").exists():
        raise FileExistsError(f"choose a new preparation directory: {output}")

    # Match tree columns to the corpus before constructing branch masses.
    # 计算树枝质量前，核对系统发育树与语料的菌属顺序。
    feature_map = pd.read_csv(ASSETS / "feature_map.tsv", sep="\t")
    with h5py.File(config.corpus, "r") as handle:
        names = read_elem(handle["var"]).index.tolist()
    if feature_map.var_name.tolist() != names or feature_map.var_index.tolist() != list(range(8114)):
        raise ValueError("UniFrac tree columns do not match the corpus")
    tree = json.loads((ASSETS / "manifest.json").read_text())
    for name, record in tree["artifacts"].items():
        if w.sha256_file(ASSETS / name) != record["sha256"]:
            raise ValueError(f"UniFrac tree asset changed: {name}")

    prior, observed = w._load_prior(config)
    rows = np.random.default_rng(20260922).choice(np.load(config.train_rows), size=768, replace=False)
    dataset = FullSupportRowDataset(config.corpus, rows)
    collator = DualCorruptionCollator(observed_genus_ids=observed, seed=config.seed, stream=91, shared_epoch=SharedEpoch(0))
    probes = output / "audit/probe"
    probes.mkdir(parents=True, exist_ok=True)
    with dataset._h5():
        # Retain the two global B384 probes, split into three B128 ranks.
        # 沿用两个全局 B384 探针，每批分给三个进程，各取 B128。
        for rank in range(3):
            for batch in range(2):
                start = batch * 384 + rank * 128
                values = collator([dataset[i] for i in range(start, start + 128)])
                np.savez(probes / f"rank{rank}_batch{batch}.npz", **values)
    np.save(probes / "rows.npy", rows)

    manifest_path = w.prepare_run(config)
    sealed = json.loads(manifest_path.read_text())
    model = w.MPCPretrainingModel(prior)
    initial = torch.load(config.initialization_checkpoint, map_location="cpu", weights_only=False)
    if initial["checkpoint_kind"] != "mpc_epoch0" or initial["model_config"] != model.config.to_dict():
        raise ValueError("initialization must be a matching fixed-PP epoch-zero checkpoint")
    model.load_state_dict(initial["model_state_dict"], strict=True)
    if w._model_state_sha256(model) != initial["model_state_sha256"]:
        raise ValueError("initialization state hash mismatch")
    initial["source_epoch0"] = {"path": str(config.initialization_checkpoint), "sha256": w.sha256_file(config.initialization_checkpoint)}
    initial["manifest_sha256"] = sealed["manifest_sha256"]
    epoch0 = output / "run/epoch0.pt"
    w._atomic_torch(epoch0, initial)

    # Fixed coefficients are recorded directly; preparation performs no updates.
    # 直接记录固定系数；准备阶段不进行参数更新或重新校准。
    weights = {
        "mode": "EXPLICIT_UNIFRAC_WEIGHT",
        "manifest_sha256": sealed["manifest_sha256"],
        "lambda": config.loss_weights,
        "epoch0": {"path": str(epoch0), "sha256": w.sha256_file(epoch0)},
        "new_calibration_optimizer_steps": 0,
    }
    weights["calibration_sha256"] = hashlib.sha256(json.dumps(weights, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    w._atomic_json(output / "audit/loss_weights.json", weights)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "smoke", "train"))
    parser.add_argument("--corpus", type=Path, required=True, help="Historical V3.1 H5AD corpus.")
    parser.add_argument("--train-rows", type=Path, required=True, help="Canonical V3.1 training row indices.")
    parser.add_argument("--val-rows", type=Path, required=True, help="Canonical V3.1 validation row indices.")
    parser.add_argument("--prior-assets", type=Path, required=True, help="Fixed PHY32+PROT480 prior archive.")
    parser.add_argument("--output-root", type=Path, required=True, help="Independent run output directory.")
    parser.add_argument("--initialization-checkpoint", type=Path, required=True, help="Matching fixed-PP epoch-zero checkpoint.")
    parser.add_argument("--unifrac-weight", type=float, required=True, help="Explicit UniFrac coefficient; historical runs used 0.00005 and 0.0015.")
    parser.add_argument("--resume", type=Path, help="Complete training checkpoint; valid only with train.")
    args = parser.parse_args()

    import torch
    from micoformer.mpc_unifrac import workflow as w

    if args.resume is not None and args.command != "train":
        parser.error("--resume is valid only with train")
    config = w.UniFracRunConfig(
        corpus=args.corpus, train_rows=args.train_rows, val_rows=args.val_rows,
        prior_assets=args.prior_assets, output_root=args.output_root,
        initialization_checkpoint=args.initialization_checkpoint, unifrac_weight=args.unifrac_weight,
    )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.command == "prepare":
        print(prepare(config))
    elif args.command == "smoke":
        print(w.run_ddp_smoke(config))
    else:
        print(w.run_full_pretraining(config, resume=args.resume))


if __name__ == "__main__":
    main()
