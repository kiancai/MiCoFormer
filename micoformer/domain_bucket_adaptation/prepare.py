"""封存已批准的两臂、原始资产、共同初始化及已提交源码。"""
from __future__ import annotations

import copy
import json
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from micoformer.domain_adaptation.prepare import load_origin, verify_file, file_record, WEIGHTS
from micoformer.domain_adaptation.lora import LoRAConfig, inject_lora, frozen_sha256
from micoformer.mpc_unifrac import workflow as common
from micoformer.mpc_unifrac.teacher import ASSETS

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO.parent
RUN = PROJECT / "tmp/20261024_respiratory_peft_mechanism_pilot/a100_bucket"
REFERENCE = PROJECT / "tmp/20261022_domain_adaptive_pretraining/strengthened_lora/audit/lora_run_manifest.json"
REFERENCE_SHA = "5ef206194b06d294f50e6c9222b77797d1f6989b3c4c5562ec2ce6dbf1528f13"
BASE = "28de0ee4f2b363cb4130f86064a542c3e3666fea"
GIT_BINARY = "/home/cml_lab/anaconda3/bin/git"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def git(*args):
    return subprocess.check_output([GIT_BINARY, "-C", str(REPO), *args], text=True).strip()


def source_record():
    if git("status", "--porcelain"):
        raise RuntimeError("commit the complete experiment implementation before sealing")
    subprocess.run([GIT_BINARY, "-C", str(REPO), "merge-base", "--is-ancestor", BASE, "HEAD"], check=True)
    names = subprocess.check_output([GIT_BINARY, "-C", str(REPO), "ls-files", "-z"]).decode().split("\0")
    return dict(base_commit=BASE, inherited_domain_implementation="5eac4cf48562309501b0663bf9494ea601162f81",
                commit=git("rev-parse", "HEAD"), worktree=str(REPO),
                files={n: common.sha256_file(REPO/n) for n in names if n})


def prepare():
    destination = RUN / "audit/manifest.json"
    if destination.exists():
        raise FileExistsError("sealed manifest already exists")
    if common.sha256_file(REFERENCE) != REFERENCE_SHA:
        raise RuntimeError("previous accepted reference changed")
    reference = json.loads(REFERENCE.read_text())
    assets = {name: copy.deepcopy(reference["assets"][name]) for name in
              ("corpus", "origin", "prior", "respiratory_train", "respiratory_val",
               "respiratory_observed", "respiratory_smoke_rows")}
    assets["reference"] = file_record(REFERENCE)
    for name in ("genus_to_edge.npz", "branch_lengths.npy"):
        assets["tree_"+name] = file_record(ASSETS/name)
    for record in assets.values():
        verify_file(record)
    train = np.load(assets["respiratory_train"]["path"], allow_pickle=False)
    val = np.load(assets["respiratory_val"]["path"], allow_pickle=False)
    if (len(train) != 110868 or len(val) != 571 or len(np.unique(train)) != len(train)
            or len(np.unique(val)) != len(val) or len(np.intersect1d(train, val))):
        raise RuntimeError("domain roster mismatch")
    common._seed_all(42)
    model = load_origin(assets["prior"]["path"], Path(assets["origin"]["path"]))
    origin_sha = common._model_state_sha256(model)
    common._seed_all(42)
    inventory = inject_lora(model, LoRAConfig(128, 256., .05))
    initialization = dict(origin_model_state_sha256=origin_sha,
        adapted_model_state_sha256=common._model_state_sha256(model),
        frozen_parameter_sha256=frozen_sha256(model))
    if (origin_sha != reference["arms"]["respiratory_rank64"]["initialization"]["origin_model_state_sha256"]
            or initialization["frozen_parameter_sha256"] != reference["arms"]["respiratory_rank64"]["initialization"]["frozen_parameter_sha256"]
            or inventory["counts"]["lora"] != 12582912):
        raise RuntimeError("initialization does not match accepted origin")
    arms = {}
    for name in ("random", "window4096"):
        arms[name] = dict(name=name, output=str(RUN/"runs"/name), seed=42,
            lora=LoRAConfig(128, 256., .05).to_dict(), epochs=10, microbatch=128,
            accumulate=1, steps_per_epoch=867, total_steps=8670, window=4096,
            learning_rate=1e-4, head_learning_rate=0., minimum_lr=1e-6,
            warmup_ratio=.02, weight_decay=.01, gradient_clip=1., workers=4,
            milestones=[0,1,3,5,10], weights=WEIGHTS,
            initialization=initialization, trainable_inventory=inventory,
            data=dict(paths=dict(train=assets["respiratory_train"]["path"], val=assets["respiratory_val"]["path"]),
                observed_genus_ids=assets["respiratory_observed"]["path"],
                smoke_rows=assets["respiratory_smoke_rows"]["path"]))
    contract = PROJECT/"tmp/20261024_respiratory_peft_mechanism_pilot/design/A100_BUCKET_PILOT.md"
    manifest = dict(schema_version=1, method="respiratory_lora_random_vs_window4096",
        prepared_at=now(), machine="CmlLab", node=socket.gethostname(), status="SEALED",
        source=source_record(), assets=assets, contract=file_record(contract), arms=arms,
        execution=dict(activation_recomputation=False, precision="BF16 autocast; FP32 parameters",
            optimizer="fused AdamW", forward="MLM then Query backward; one clip and optimizer step",
            allocation=673251, world_size=1, relation_pool_limit=128,
            validation="same original saved row order; stream2 epoch0; no length grouping"),
        torch=torch.__version__, cuda=torch.version.cuda)
    common._atomic_json(destination, manifest)
    sha = common.sha256_file(destination)
    (RUN/"audit/manifest.sha256").write_text(sha+"\n")
    print(json.dumps(dict(path=str(destination), sha256=sha, source=manifest["source"]["commit"])))
    return manifest


def verify(path, expected_sha):
    path = Path(path)
    if common.sha256_file(path) != expected_sha:
        raise RuntimeError("manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest["method"] != "respiratory_lora_random_vs_window4096" or set(manifest["arms"]) != {"random", "window4096"}:
        raise RuntimeError("wrong experiment")
    if manifest["source"]["commit"] != git("rev-parse", "HEAD") or git("status", "--porcelain"):
        raise RuntimeError("running code must match sealed clean commit")
    for name, sha in manifest["source"]["files"].items():
        if common.sha256_file(REPO/name) != sha:
            raise RuntimeError("source changed: "+name)
    for record in list(manifest["assets"].values())+[manifest["contract"]]:
        verify_file(record)
    return manifest
