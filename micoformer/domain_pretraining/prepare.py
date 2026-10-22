"""Freeze V3.3 domain rows, train-only vocabulary, initialization and provenance."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import socket
import subprocess
from datetime import datetime

import h5py
import numpy as np
import pandas as pd
import torch
from anndata.io import read_elem

from .data import DomainModelConfig, ExactBatchSampler
from micoformer.mpc_unifrac import workflow as common
from micoformer.mpc_unifrac.model import UniFracModel
from micoformer.mpc_unifrac.teacher import ASSETS, TEACHER_T, STUDENT_T

BASE_COMMIT = "28de0ee4f2b363cb4130f86064a542c3e3666fea"
CORPUS_SHA256 = "4373cace4204621e4438ee2e05531c3677a0eaa437289bbdbf0805f581a8d887"
PRIOR_SHA256 = "6a2c08d216efb90499720eec5aa872df5275a62d5e8052760a5105765262262c"
INITIAL_12_SHA256 = "011f34e4e98652ae60cfb8c7e4a821061e2e8a39fc2132312921035fce1157a5"
WEIGHTS = dict(mlm=1., continuous=.534399128791011,
               presence=.24296011504612833, unifrac=.00005)
GUT_SITES = ("Large intestine", "Small intestine", "Lower gastrointestinal tract")
REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
GIT_BINARY = "/home/cml_lab/anaconda3/bin/git"  # Verified 2.42; system Git lacks worktree support.


def git(*args: str) -> str:
    return subprocess.check_output([GIT_BINARY, "-C", str(REPO), *args], text=True).strip()


def file_record(path: Path) -> dict:
    return dict(path=str(path.resolve()), sha256=common.sha256_file(path), bytes=path.stat().st_size)


def verify_file(record: dict) -> None:
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or common.sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"bound asset changed: {path}")


def source_identity() -> dict:
    if git("status", "--porcelain"):
        raise RuntimeError("commit preparation code before binding or running it")
    subprocess.run([GIT_BINARY, "-C", str(REPO), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], check=True)
    paths = subprocess.check_output([GIT_BINARY, "-C", str(REPO), "ls-files", "-z"]).decode().split("\0")
    return dict(base_commit=BASE_COMMIT, commit=git("rev-parse", "HEAD"),
                branch=git("branch", "--show-current"), worktree=str(REPO),
                files={name: common.sha256_file(REPO / name) for name in paths if name})


def validate_manifest(path: Path, expected_sha256: str) -> dict:
    if common.sha256_file(path) != expected_sha256:
        raise RuntimeError("submitted manifest SHA256 changed")
    manifest = json.loads(path.read_text())
    if manifest["torch"] != torch.__version__ or manifest["cuda"] != torch.version.cuda:
        raise RuntimeError("prepared PyTorch/CUDA environment changed")
    if manifest["source"] != source_identity():
        raise RuntimeError("run source/commit changed after preparation")
    for record in manifest["assets"].values():
        verify_file(record)
    return manifest


def rebind_source(path: Path, expected_sha256: str, destination: Path) -> Path:
    """Preserve all scientific inputs when repairing a failed job launcher.

    The old manifest remains immutable. This explicit command only binds the
    clean new source and retains the previous manifest/hash as provenance.
    """
    if common.sha256_file(path) != expected_sha256 or destination.exists():
        raise RuntimeError("old manifest changed or new manifest already exists")
    manifest = json.loads(path.read_text())
    for record in manifest["assets"].values():
        verify_file(record)
    manifest["source"] = source_identity()
    manifest["previous_manifest"] = file_record(path)
    manifest["source_rebound_at"] = datetime.now().astimezone().isoformat()
    manifest["source_rebind_reason"] = "repair GPU batch driver environment; unchanged scientific inputs and method"
    common._atomic_json(destination, manifest)
    return destination


def select_domains(primary, l2, l3, l4) -> dict[str, np.ndarray]:
    human = np.asarray(primary, dtype=bool) & (np.asarray(l2) == "Mammals: Human")
    return {
        "gut": np.flatnonzero(human & (np.asarray(l3) == "Digestive system") & np.isin(l4, GUT_SITES)),
        "respiratory": np.flatnonzero(human & (np.asarray(l3) == "Respiratory system")),
    }


def normalize_run_ids(values) -> np.ndarray:
    """Preserve missing Run identities without adding a categorical value."""
    return pd.Series(values).astype("string").fillna("").to_numpy(dtype=str)


def prepare_experiment(output: Path) -> Path:
    output = output.resolve()
    manifest_path = output / "audit/domain_run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("existing preparation is immutable; validate or choose a new directory")
    source = source_identity()
    corpus = PROJECT / "data/MCFCorpusV3.3/MCFCorpusV3.3.gg2.h5ad"
    prepared = corpus.parent / "prepared"
    prior = PROJECT / "tmp/20260813_fixed_prior_query_architecture_gate/cache/prior_assets.npz"
    initial = PROJECT / "tmp/20260817_mpc_fixed_context_full_data_pretraining/run/epoch0.pt"
    assets = {"corpus": file_record(corpus), "prior": file_record(prior),
              "reference_epoch0": file_record(initial),
              "contract": file_record(output / "PLAN.md")}
    for key, expected in [("corpus", CORPUS_SHA256), ("prior", PRIOR_SHA256), ("reference_epoch0", INITIAL_12_SHA256)]:
        if assets[key]["sha256"] != expected:
            raise ValueError(f"unexpected {key} identity")
    prep = json.loads((prepared / "preparation.json").read_text())
    assets["canonical_preparation"] = file_record(prepared / "preparation.json")
    canonical = {}
    for split in ("all", "train", "val"):
        record = prep["splits"][split]
        path = prepared / record["file"]
        assets["canonical_" + split] = file_record(path)
        if assets["canonical_" + split]["sha256"] != record["sha256"]:
            raise ValueError("canonical V3.3 split changed")
        canonical[split] = np.load(path, allow_pickle=False)
        if len(canonical[split]) != record["count"] or len(np.unique(canonical[split])) != len(canonical[split]):
            raise ValueError("invalid canonical split")
    if np.intersect1d(canonical["train"], canonical["val"]).size or not np.array_equal(
            np.sort(np.r_[canonical["train"], canonical["val"]]), np.sort(canonical["all"])):
        raise ValueError("canonical splits overlap or omit rows")
    tree = json.loads((ASSETS / "manifest.json").read_text())
    assets["tree_manifest"] = file_record(ASSETS / "manifest.json")
    for name, record in tree["artifacts"].items():
        assets["tree_" + name] = file_record(ASSETS / name)
        if assets["tree_" + name]["sha256"] != record["sha256"]:
            raise ValueError(f"tree asset changed: {name}")
    with np.load(prior, allow_pickle=False) as z:
        table = torch.from_numpy(z["pp_table"].copy()).float()
        prior_names = z["var_names_utf8"].astype(str)
        if table.shape != (8116, 512) or not torch.isfinite(table).all():
            raise ValueError("invalid fixed prior")
        joint = np.concatenate([z["phyproj_table"], z["protproj_table"]], axis=1) * 16.
        if not np.allclose(z["pp_table"], joint, rtol=2e-7, atol=1e-7):
            raise ValueError("prior is not direct PHY32+PROT480")
    feature_map = pd.read_csv(ASSETS / "feature_map.tsv", sep="\t")
    output.joinpath("data").mkdir(parents=True, exist_ok=True)
    domains = {}
    with h5py.File(corpus, "r") as h:
        if tuple(h["X"].attrs["shape"]) != (3334541, 8114):
            raise ValueError("unexpected corpus dimensions")
        names = np.asarray(read_elem(h["var"]).index).astype(str)
        if not np.array_equal(names, prior_names) or feature_map.var_name.tolist() != names.tolist():
            raise ValueError("corpus/prior/UniFrac genus order mismatch")
        if feature_map.var_index.tolist() != list(range(8114)):
            raise ValueError("UniFrac feature indices changed")
        meta = {key: read_elem(h["obs"][key]) for key in (
            "analysis_primary", "gold_l2", "gold_l3", "gold_l4", "Run", "Database", "sample_id")}
        primary = np.flatnonzero(np.asarray(meta["analysis_primary"], dtype=bool))
        if not np.array_equal(primary, np.sort(canonical["all"])):
            raise ValueError("canonical all rows differ from current analysis_primary")
        masks = select_domains(meta["analysis_primary"], meta["gold_l2"], meta["gold_l3"], meta["gold_l4"])
        train_masks = {}
        observed = {}
        indptr = h["X/indptr"][:]
        richness = np.diff(indptr)
        runs = normalize_run_ids(meta["Run"])
        for name, rows in masks.items():
            train = np.intersect1d(rows, canonical["train"])
            val = np.intersect1d(rows, canonical["val"])
            if len(train) < 1024 or len(val) < 2 or len(train) + len(val) != len(rows):
                raise ValueError("domain split too small or incomplete")
            if np.any(richness[rows] < 5):
                raise ValueError("domain violates >=5 genera input contract")
            paths = {}
            for split, array in [("all", rows), ("train", train), ("val", val)]:
                path = output / "data" / f"{name}_{split}.npy"
                np.save(path, array.astype(np.int64), allow_pickle=False)
                assets[f"{name}_{split}"] = file_record(path)
                paths[split] = str(path)
            domain_frame = pd.DataFrame({"row_id": rows,
                "sample_id": np.asarray(meta["sample_id"])[rows].astype(str),
                "Run": runs[rows], "Database": np.asarray(meta["Database"])[rows].astype(str),
                "gold_l4": np.asarray(meta["gold_l4"])[rows].astype(str),
                "split": np.where(np.isin(rows, val), "val", "train")})
            row_table = output / "data" / f"{name}_rows.tsv.gz"
            domain_frame.to_csv(row_table, sep="\t", index=False, compression="gzip")
            assets[name + "_row_table"] = file_record(row_table)
            # The hardest real B128 batch exercises the domain's maximum length.
            order = np.lexsort((train, -richness[train]))
            smoke_rows = train[order[:128]]
            path = output / "data" / f"{name}_smoke_rows.npy"
            np.save(path, smoke_rows, allow_pickle=False)
            assets[name + "_smoke_rows"] = file_record(path)
            shared = np.intersect1d(runs[train][runs[train] != ""], runs[val][runs[val] != ""])
            domains[name] = dict(rows=len(rows), train=len(train), val=len(val), paths=paths,
                smoke_rows=str(path), maximum_richness=int(richness[rows].max()),
                source_counts=domain_frame.Database.value_counts().to_dict(),
                site_counts=domain_frame.gold_l4.value_counts().to_dict(),
                cross_split_shared_nonmissing_Run_ids=int(len(shared)),
                cross_split_shared_Run_val_rows=int(np.isin(runs[val], shared).sum()))
            train_masks[name] = np.zeros(len(richness), dtype=bool)
            train_masks[name][train] = True
            observed[name] = np.zeros(8114, dtype=bool)
        # One bounded sequential CSR scan, not hundreds of thousands of H5 reads.
        for start in range(0, len(richness), 8192):
            stop = min(start + 8192, len(richness))
            if not any(mask[start:stop].any() for mask in train_masks.values()):
                continue
            indices = h["X/indices"][int(indptr[start]):int(indptr[stop])]
            for name, mask in train_masks.items():
                keep = np.repeat(mask[start:stop], richness[start:stop])
                observed[name][indices[keep]] = True
    for name, seen in observed.items():
        path = output / "data" / f"{name}_observed_genus_ids.npy"
        values = np.flatnonzero(seen).astype(np.int64) + 2
        np.save(path, values, allow_pickle=False)
        assets[name + "_observed"] = file_record(path)
        domains[name]["observed_genus_ids"] = str(path)
        domains[name]["observed_genus_count"] = len(values)

    initializations = {}
    for layers in (12, 6):
        common._seed_all(42)
        config = DomainModelConfig(num_layers=layers)
        model = UniFracModel(table, config)
        if layers == 12:
            reference = torch.load(initial, map_location="cpu", weights_only=False)
            if reference["checkpoint_kind"] != "mpc_epoch0" or reference["model_config"] != config.to_dict():
                raise ValueError("reference initialization is not matching untrained fixed-PP")
            model.load_state_dict(reference["model_state_dict"], strict=True)
            if common._model_state_sha256(model) != reference["model_state_sha256"]:
                raise ValueError("reference epoch0 strict restore changed parameters")
        path = output / "audit" / f"initialization_{layers}l.pt"
        state_hash = common._model_state_sha256(model)
        common._atomic_torch(path, dict(checkpoint_kind="mpc_epoch0", model_config=config.to_dict(),
            model_state_dict=model.state_dict(), model_state_sha256=state_hash, seed=42,
            initialization_policy="reference_untrained_epoch0" if layers == 12 else "native_6l_seed42"))
        assets[f"initialization_{layers}l"] = file_record(path)
        initializations[str(layers)] = dict(path=str(path), model_state_sha256=state_hash,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
        del model
    arms = {}
    for domain, data in domains.items():
        batches = len(ExactBatchSampler(data["train"], seed=42, epoch=0, shuffle=True))
        for layers in (12, 6):
            name = f"{domain}_{layers}l"
            arms[name] = dict(name=name, domain=domain, num_layers=layers, d_model=512,
                seed=42, epochs=50, microbatch=128, accumulate=3, workers=2,
                learning_rate=.0003, weight_decay=.01, gradient_clip=1., warmup_ratio=.02,
                minimum_lr=.000001, weights=WEIGHTS, teacher_temperature=TEACHER_T,
                student_temperature=STUDENT_T, initialization=initializations[str(layers)],
                output=str(output / "runs" / name), data=data,
                steps_per_epoch=math.ceil(batches / 3), total_steps=50 * math.ceil(batches / 3))
    manifest = dict(schema_version=1, status="SEALED", machine="CmlLab", node=socket.gethostname(),
        prepared_at=datetime.now().astimezone().isoformat(), source=source, assets=assets,
        corpus_version="MCFCorpusV3.3", domains=domains, arms=arms,
        domain_policy=dict(human_gold_l2="Mammals: Human", primary_only=True,
            gut_gold_l3="Digestive system", gut_gold_l4=list(GUT_SITES),
            respiratory_gold_l3="Respiratory system", split="intersection_with_V3.3_published_splits",
            split_limit="row-random; not Study/subject held out; retain source duplicate policy"),
        torch=torch.__version__, cuda=torch.version.cuda)
    common._atomic_json(manifest_path, manifest)
    return manifest_path
