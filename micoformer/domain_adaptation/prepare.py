"""Seal two LoRA domains against the accepted V3.4 roster and trained origin."""
from __future__ import annotations

import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import socket
import subprocess

import numpy as np
import torch

from micoformer.mpc_fixed_pretraining.model import MPCFixedModelConfig
from micoformer.mpc_unifrac import workflow as common
from micoformer.mpc_unifrac.model import UniFracModel
from .lora import LoRAConfig, inject_lora, frozen_sha256

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
GIT_BINARY = "/home/cml_lab/anaconda3/bin/git"
BASE_COMMIT = "28de0ee4f2b363cb4130f86064a542c3e3666fea"
PARENT = PROJECT / "tmp/20261022_domain_corpus_pretraining/v34/audit/domain_run_manifest_v2.json"
PARENT_SHA = "1489d372acf8fbc0e879fe0f681a23b12bd8d5f00b05b28702907442da140255"
ORIGIN = PROJECT / "tmp/20260923_mpc_fixedpp_v31_weak_unifrac/run/milestones/epoch_050.pt"
ORIGIN_SHA = "0eeef0cc83fae83217a5d6808629b840c919bc77de7824a4fe580479fced289c"
SUBMISSION_ORDER = ("respiratory_lora", "gut_lora")
WEIGHTS = dict(mlm=1., continuous=.534399128791011,
               presence=.24296011504612833, unifrac=.00005)


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=common.sha256_file(path))


def verify_file(record):
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"] or common.sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"bound asset changed: {path}")


def git(*args):
    return subprocess.check_output([GIT_BINARY, "-C", str(REPO), *args], text=True).strip()


def source_identity():
    if git("status", "--porcelain"):
        raise RuntimeError("commit preparation code before sealing or running")
    subprocess.run([GIT_BINARY, "-C", str(REPO), "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], check=True)
    names = subprocess.check_output([GIT_BINARY, "-C", str(REPO), "ls-files", "-z"]).decode().split("\0")
    return dict(base_commit=BASE_COMMIT, commit=git("rev-parse", "HEAD"),
                branch=git("branch", "--show-current"), worktree=str(REPO),
                files={name: common.sha256_file(REPO / name) for name in names if name})


def load_origin(prior_path, origin_path=ORIGIN):
    with np.load(prior_path, allow_pickle=False) as data:
        prior = torch.from_numpy(data["pp_table"].copy()).float()
    payload = torch.load(origin_path, map_location="cpu", weights_only=False)
    if (payload["checkpoint_kind"] != "mpc_model_milestone" or payload["epoch"] != 49
            or payload["global_step"] != 263500 or payload["lambda"] != WEIGHTS):
        raise RuntimeError("origin must be the approved trained weak-UniFrac epoch50")
    config = MPCFixedModelConfig(**payload["model_config"])
    model = UniFracModel(prior, config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def validate_contract(manifest):
    if (set(manifest["arms"]) != set(SUBMISSION_ORDER)
            or manifest["submission_order"] != list(SUBMISSION_ORDER)
            or manifest["method"] != "unlabelled_lora_plus_existing_self_supervised_heads"
            or manifest["assets"]["origin"]["sha256"] != ORIGIN_SHA
            or manifest["assets"]["parent_manifest"]["sha256"] != PARENT_SHA):
        raise RuntimeError("invalid LoRA-only experiment identity")
    for name in SUBMISSION_ORDER:
        arm = manifest["arms"][name]
        domain = name.removesuffix("_lora")
        expected_steps = {"respiratory": 289, "gut": 768}[domain]
        if (arm["name"] != name or arm["domain"] != domain or arm["num_layers"] != 12
                or arm["epochs"] != 10 or arm["seed"] != 42
                or arm["microbatch"] != 128 or arm["accumulate"] != 3
                or arm["weights"] != WEIGHTS or arm["lora"] != LoRAConfig().to_dict()
                or arm["steps_per_epoch"] != expected_steps or arm["total_steps"] != expected_steps * 10
                or arm["learning_rate"] != 1e-4 or arm["head_learning_rate"] != 1e-5
                or arm["minimum_lr"] != 1e-6 or arm["warmup_ratio"] != .02
                or arm["milestones"] != [0, 1, 3, 5, 10]):
            raise RuntimeError(f"LoRA training contract changed: {name}")
        for split in ("all", "train", "val"):
            if arm["data"]["paths"][split] != manifest["assets"][f"{domain}_{split}"]["path"]:
                raise RuntimeError("unbound domain rows")
        for key, asset_key in (("observed_genus_ids", "observed"), ("smoke_rows", "smoke_rows")):
            if arm["data"][key] != manifest["assets"][f"{domain}_{asset_key}"]["path"]:
                raise RuntimeError("unbound domain auxiliary data")


def validate_manifest(path, expected_sha):
    path = Path(path)
    if common.sha256_file(path) != expected_sha:
        raise RuntimeError("manifest hash changed")
    manifest = json.loads(path.read_text())
    validate_contract(manifest)
    if manifest["source"] != source_identity():
        raise RuntimeError("source changed after preparation")
    if manifest["torch"] != torch.__version__ or manifest["cuda"] != torch.version.cuda:
        raise RuntimeError("prepared environment changed")
    for record in manifest["assets"].values():
        verify_file(record)
    return manifest


def prepare_experiment(output):
    output = Path(output).resolve()
    destination = output / "audit/lora_run_manifest.json"
    if destination.exists():
        raise FileExistsError("preparation is immutable")
    source = source_identity()
    if common.sha256_file(PARENT) != PARENT_SHA or common.sha256_file(ORIGIN) != ORIGIN_SHA:
        raise RuntimeError("domain roster or trained origin identity changed")
    parent = json.loads(PARENT.read_text())
    # Verify the existing asset contract before copying its exact domain rosters.
    # This does not require editing/rebinding the running experiment's worktree.
    for record in parent["assets"].values():
        verify_file(record)
    keys = [name for name in parent["assets"] if name.startswith(("tree_", "canonical_", "corpus"))]
    assets = {name: copy.deepcopy(parent["assets"][name]) for name in keys + ["prior"]}
    assets.update(origin=file_record(ORIGIN), parent_manifest=file_record(PARENT),
                  contract=file_record(output / "PLAN.md"))
    (output / "data").mkdir(parents=True, exist_ok=True)
    arms = {}
    for name in SUBMISSION_ORDER:
        domain = name.removesuffix("_lora")
        arm = copy.deepcopy(parent["arms"][domain + "_12l"])
        arm.pop("initialization")
        for suffix in ("all", "train", "val", "observed", "smoke_rows", "row_table"):
            key = domain + "_" + suffix
            src = Path(parent["assets"][key]["path"])
            dst = output / "data" / src.name
            if dst.exists():
                raise FileExistsError(f"refusing to overwrite prepared data: {dst}")
            shutil.copyfile(src, dst)
            assets[key] = file_record(dst)
            if assets[key]["sha256"] != parent["assets"][key]["sha256"]:
                raise RuntimeError("domain data copy changed content")
            if suffix in ("all", "train", "val"):
                arm["data"]["paths"][suffix] = str(dst)
            elif suffix == "observed":
                arm["data"]["observed_genus_ids"] = str(dst)
            elif suffix == "smoke_rows":
                arm["data"]["smoke_rows"] = str(dst)
        arm.update(name=name, epochs=10, total_steps=10 * arm["steps_per_epoch"],
                   learning_rate=1e-4, head_learning_rate=1e-5, minimum_lr=1e-6,
                   lora=LoRAConfig().to_dict(), milestones=[0, 1, 3, 5, 10],
                   output=str(output / "runs" / name))
        arms[name] = arm
    common._seed_all(42)
    model = load_origin(assets["prior"]["path"])
    origin_state = common._model_state_sha256(model)
    # Adapter initialization uses a separate seed after restoring trained weights.
    common._seed_all(42)
    inventory = inject_lora(model, LoRAConfig())
    initialization = dict(origin_model_state_sha256=origin_state,
                          adapted_model_state_sha256=common._model_state_sha256(model),
                          frozen_parameter_sha256=frozen_sha256(model),
                          optimizer_policy="fresh AdamW and schedule; do not restore origin optimizer")
    manifest = dict(schema_version=1, status="SEALED", method="unlabelled_lora_plus_existing_self_supervised_heads",
                    prepared_at=datetime.now().astimezone().isoformat(), machine="CmlLab", node=socket.gethostname(),
                    corpus_version="MCFCorpusV3.4", domain_policy=parent["domain_policy"],
                    source=source, assets=assets, arms=arms, submission_order=list(SUBMISSION_ORDER),
                    initialization=initialization, trainable_inventory=inventory,
                    torch=torch.__version__, cuda=torch.version.cuda)
    validate_contract(manifest)
    common._atomic_json(destination, manifest)
    return destination
