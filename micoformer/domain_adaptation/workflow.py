"""Single allocated GPU: zero-delta/frozen/restore gates, then 10-epoch LoRA DAPT."""
from __future__ import annotations

import gc
import json
import math
import os
import random
from pathlib import Path
import signal
import socket
import subprocess
from time import perf_counter
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from micoformer.domain_pretraining.data import ExactBatchSampler, accumulation_groups, accumulation_loss, target_counts
from .prepare import validate_manifest, load_origin
from .lora import LoRAConfig, inject_lora, frozen_sha256, parameter_groups, trainable_inventory
from micoformer.domain_pretraining.input import InMemoryFullSupportRowDataset as FullSupportRowDataset
from micoformer.mpc_unifrac import workflow as common
from micoformer.mpc_unifrac.data import DualCorruptionCollator, SharedEpoch, batch_to_device
from micoformer.mpc_unifrac.model import UniFracModel
from micoformer.mpc_unifrac.teacher import weighted_unifrac, neighborhood_details
from micoformer.mpc_unifrac.support import _state_close


def state_equal(left, right):
    """Exact replay check, with no implicit floating-point tolerance."""
    if torch.is_tensor(left):
        return (torch.is_tensor(right) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return left == right


def capture_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        if state["torch_cuda"] is None:
            raise RuntimeError("CPU audit state cannot resume a CUDA run")
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)
    elif state["torch_cuda"] is not None:
        raise RuntimeError("CUDA run cannot silently resume on CPU")


def stamp() -> str:
    return datetime.now().astimezone().isoformat()


def gpu_preflight(output: Path) -> torch.device:
    job = os.environ.get("SLURM_JOB_ID", "")
    cgroup = Path("/proc/self/cgroup").read_text()
    if not job or f"job_{job}" not in cgroup:
        raise RuntimeError("training must run inside its own Slurm allocation")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one Slurm-assigned visible CUDA device is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    info = torch.cuda.get_device_properties(device)
    if "A100" not in info.name or info.total_memory < 38 * 1024**3 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("B128 contract requires an allocated A100 >=38GiB with BF16")
    # Actual CUDA work before a formal update; no silent CPU fallback.
    x = torch.ones((64, 64), device=device)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        result = x @ x
    torch.cuda.synchronize()
    if not torch.all(result == 64):
        raise RuntimeError("CUDA preflight arithmetic failed")
    details = subprocess.check_output(["scontrol", "show", "job", "-dd", job], text=True)
    smi = subprocess.check_output(["nvidia-smi"], text=True)
    common._atomic_json(output / "audit/gpu_preflight.json", dict(
        machine="CmlLab", node=socket.gethostname(), checked_at=stamp(), slurm_job_id=job,
        cuda_visible_devices=visible, slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
        cuda_device_count=1, local_device="cuda:0", name=info.name,
        total_memory_bytes=info.total_memory, cgroup=cgroup,
        scontrol=details, nvidia_smi=smi, torch=torch.__version__, cuda=torch.version.cuda))
    del x, result
    return device


def load_model(arm: dict, manifest: dict, device: torch.device) -> UniFracModel:
    model = load_origin(manifest["assets"]["prior"]["path"], manifest["assets"]["origin"]["path"])
    if common._model_state_sha256(model) != manifest["initialization"]["origin_model_state_sha256"]:
        raise RuntimeError("trained origin state changed")
    common._seed_all(arm["seed"])
    inventory = inject_lora(model, LoRAConfig(**arm["lora"]))
    if inventory != manifest["trainable_inventory"]:
        raise RuntimeError("trainable parameter scope changed")
    if common._model_state_sha256(model) != manifest["initialization"]["adapted_model_state_sha256"]:
        raise RuntimeError("LoRA initialization changed")
    model._dapt_prior_record = manifest["assets"]["prior"]
    model._dapt_origin_record = manifest["assets"]["origin"]
    return model.to(device)


def make_optimizer(model, arm):
    groups = parameter_groups(model, lora_lr=arm["learning_rate"], head_lr=arm["head_learning_rate"])
    optimizer = torch.optim.AdamW(groups, weight_decay=arm["weight_decay"])
    scheduler = common._make_scheduler(optimizer, total_steps=arm["total_steps"],
        warmup_ratio=arm["warmup_ratio"], base_lr=arm["learning_rate"], minimum_lr=arm["minimum_lr"])
    return optimizer, scheduler


def make_loader(arm, corpus, split: str, *, epoch: int, start_batch: int = 0, workers: int | None = None):
    rows = np.load(arm["data"]["paths"][split], allow_pickle=False)
    dataset = FullSupportRowDataset(corpus, rows)
    sampler = ExactBatchSampler(len(rows), seed=arm["seed"], epoch=epoch,
        shuffle=split == "train", batch_size=arm["microbatch"], start_batch=start_batch)
    collator = DualCorruptionCollator(
        observed_genus_ids=np.load(arm["data"]["observed_genus_ids"], allow_pickle=False),
        seed=arm["seed"], stream=1 if split == "train" else 2,
        shared_epoch=SharedEpoch(epoch if split == "train" else 0))
    worker_count = arm["workers"] if workers is None else workers
    # An independent generator prevents worker startup/prefetch from consuming
    # the model dropout RNG, which makes cursor-based resume reproducible.
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator,
        num_workers=worker_count, pin_memory=True,
        multiprocessing_context="fork" if worker_count else None,
        prefetch_factor=2 if worker_count else None,
        generator=torch.Generator().manual_seed(arm["seed"] + epoch + (0 if split == "train" else 10000)))
    return loader, sampler, rows


def update(model, optimizer, scheduler, group, arm, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = np.stack([target_counts(batch) for batch in group]).sum(0)
    statistics = torch.zeros(12, device=device, dtype=torch.float64)
    objective = 0.
    for batch in group:
        arguments = batch_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**arguments)
            loss = accumulation_loss(output, arm["weights"], totals)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        loss.backward()
        objective += float(loss.detach())
        statistics += common._loss_statistics_tensor(output, arm["weights"], device=device)
        del output, loss, arguments
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], arm["gradient_clip"], error_if_nonfinite=True)
    # clip_grad_norm checks every gradient; do not step on a failed gradient.
    lr_used = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return statistics, dict(objective=objective, gradient_norm_preclip=float(norm),
        lr_used=lr_used, lr_next=float(optimizer.param_groups[0]["lr"]),
        head_lr_next=float(optimizer.param_groups[1]["lr"]),
        target_counts=totals.tolist(), samples=int(totals[3]))


@torch.inference_mode()
def validate(model, arm, corpus, device, *, max_batches=None, workers=None):
    model.eval()
    loader, _, expected_rows = make_loader(arm, corpus, "val", epoch=0, workers=workers)
    statistics = torch.zeros(12, device=device, dtype=torch.float64)
    clean = torch.zeros(4, device=device, dtype=torch.float64)
    sum_h = torch.zeros(512, device=device, dtype=torch.float64)
    sum_outer = torch.zeros((512, 512), device=device, dtype=torch.float64)
    seen = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        seen.append(batch["row_ids"])
        args = batch_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(**args)
            _, mean = model.unmasked_representations(args["genus_ids"], args["rclr"], args["padding_mask"])
        ce, entropy, kl = neighborhood_details(mean, weighted_unifrac(args["edge_mass"], model.unifrac_lengths))
        statistics += common._loss_statistics_tensor(output, arm["weights"], device=device)
        count = len(mean)
        clean += torch.stack([ce.double() * count, entropy.double() * count,
                              kl.double() * count, torch.as_tensor(count, device=device)])
        h = mean.float().double()
        sum_h += h.sum(0)
        sum_outer += h.T @ h
    rows = np.concatenate(seen)
    if max_batches is None and not np.array_equal(rows, expected_rows):
        raise RuntimeError("validation omitted, repeated or reordered rows")
    return dict(loss=common._unpack_loss_statistics(statistics, arm["weights"]),
        clean_unifrac_diagnostics=dict(zip(["ce", "teacher_entropy", "kl"], (clean[:3] / clean[3]).cpu().tolist())),
        representation=common._spectral_summary(len(rows), sum_h.cpu(), sum_outer.cpu()),
        row_count=len(rows), row_unique_count=len(np.unique(rows)), exact_row_coverage=max_batches is None)


def checkpoint(model, optimizer, scheduler, arm, manifest_sha, device, *, epoch, cursor,
               global_step, best_score, statistics):
    return dict(schema_version=1, checkpoint_kind="domain_lora_full_resume",
        lora_config=arm["lora"], prior=model._dapt_prior_record, origin=model._dapt_origin_record,
        frozen_parameter_sha256=frozen_sha256(model),
        manifest_sha256=manifest_sha, arm=arm["name"], model_config=model.config.to_dict(),
        model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(), rng_state=capture_rng(device),
        epoch=epoch, next_batch=cursor, global_step=global_step, best_score=best_score,
        epoch_statistics=statistics.detach().cpu())


def restore(path, model, optimizer, scheduler, arm, manifest_sha, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["checkpoint_kind"] != "domain_lora_full_resume" or state["arm"] != arm["name"]
            or state["manifest_sha256"] != manifest_sha or state["model_config"] != model.config.to_dict()
            or state["lora_config"] != arm["lora"] or state["prior"] != model._dapt_prior_record
            or state["origin"] != model._dapt_origin_record):
        raise RuntimeError("resume belongs to a different arm/manifest/architecture")
    expected_frozen = frozen_sha256(model)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if frozen_sha256(model) != expected_frozen or state["frozen_parameter_sha256"] != expected_frozen:
        raise RuntimeError("frozen origin weights changed in checkpoint")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng(state["rng_state"], device)
    return state


def smoke(arm, manifest, manifest_sha, device):
    output = Path(arm["output"])
    corpus = manifest["assets"]["corpus"]["path"]
    rows = np.load(arm["data"]["smoke_rows"], allow_pickle=False)
    dataset = FullSupportRowDataset(corpus, rows)
    collator = DualCorruptionCollator(
        observed_genus_ids=np.load(arm["data"]["observed_genus_ids"], allow_pickle=False),
        seed=42, stream=91, shared_epoch=SharedEpoch(0))
    samples = [dataset[i] for i in range(len(rows))]
    full = collator(samples)
    # Three real B128 forwards test peak activation memory and accumulation.
    # An uneven 127+2 group separately tests tail-count weighting and restore.
    group = [full, full, full]
    tail = [collator(samples[:127]), collator(samples[126:128])]
    model = load_model(arm, manifest, device)
    original = load_origin(manifest["assets"]["prior"]["path"], manifest["assets"]["origin"]["path"]).to(device)
    args = batch_to_device(full, device)
    model.eval()
    original.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = original.unmasked_representations(args["genus_ids"], args["rclr"], args["padding_mask"])[1]
        adapted = model.unmasked_representations(args["genus_ids"], args["rclr"], args["padding_mask"])[1]
    if not torch.equal(before, adapted):
        raise RuntimeError("zero-delta LoRA changed original embeddings")
    frozen_before = frozen_sha256(model)
    del original, args, before, adapted
    optimizer, scheduler = make_optimizer(model, arm)
    torch.cuda.reset_peak_memory_stats(device)
    common._seed_all(42)
    first_stats, first = update(model, optimizer, scheduler, group, arm, device)
    cp = output / "audit/smoke_resume.pt"
    common._atomic_torch(cp, checkpoint(model, optimizer, scheduler, arm, manifest_sha, device,
        epoch=0, cursor=3, global_step=1, best_score=math.inf, statistics=first_stats))
    _, next_update = update(model, optimizer, scheduler, tail, arm, device)
    expected_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    expected_opt = optimizer.state_dict()
    # Move the expected optimizer tensors to CPU before releasing the first run.
    for state in expected_opt["state"].values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.detach().cpu().clone()
    expected_sched = scheduler.state_dict()
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    model = load_model(arm, manifest, device)
    optimizer, scheduler = make_optimizer(model, arm)
    restored = restore(cp, model, optimizer, scheduler, arm, manifest_sha, device)
    if restored["next_batch"] != 3 or restored["global_step"] != 1:
        raise RuntimeError("smoke resume cursor changed")
    _, replay = update(model, optimizer, scheduler, tail, arm, device)
    if not state_equal(expected_model, model.state_dict()) or not state_equal(expected_opt, optimizer.state_dict()):
        raise RuntimeError("restored next update differs from uninterrupted model/optimizer")
    if expected_sched != scheduler.state_dict() or replay != next_update:
        raise RuntimeError("restored LR schedule or next update diagnostics drifted")
    validation = validate(model, arm, corpus, device, workers=0)
    if frozen_sha256(model) != frozen_before:
        raise RuntimeError("frozen parameters changed during smoke")
    if not any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if n.endswith(".lora_B")):
        raise RuntimeError("LoRA received no effective update")
    result = dict(zero_delta_embedding_exact=True, frozen_weights_unchanged=True,
        lora_updated=True, status="PASS", checked_at=stamp(), manifest_sha256=manifest_sha,
        arm=arm["name"], actual_microbatch=128, accumulation=3,
        max_smoke_sequence_length=int(full["genus_ids"].shape[1]), tail_microbatches=[127, 2],
        next_update_restore_match=True, checkpoint_strict_load=True,
        first_update=first, replay_update=replay, validation=validation,
        peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3)
    common._atomic_json(output / "audit/cuda_smoke.json", result)
    del model, optimizer, scheduler, restored, expected_model, expected_opt, first_stats
    gc.collect()
    torch.cuda.empty_cache()
    # The small smoke checkpoint is diagnostic, not a formal initialization.
    return result


def train(arm, manifest, manifest_sha, device, *, resume: Path | None):
    output = Path(arm["output"])
    corpus = manifest["assets"]["corpus"]["path"]
    model = load_model(arm, manifest, device)
    optimizer, scheduler = make_optimizer(model, arm)
    common._seed_all(arm["seed"])
    epoch, cursor, global_step, best_score = 0, 0, 0, math.inf
    statistics = torch.zeros(12, device=device, dtype=torch.float64)
    if resume:
        state = restore(resume, model, optimizer, scheduler, arm, manifest_sha, device)
        epoch, cursor, global_step = state["epoch"], state["next_batch"], state["global_step"]
        best_score = state["best_score"]
        statistics.copy_(state["epoch_statistics"].to(device))
        del state
    stop = []
    def request_stop(signum, frame):
        stop.append(signum)
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, request_stop)
    def save(epoch, cursor, statistics, path=None):
        path = path or output / "run/last.pt"
        common._atomic_torch(path, checkpoint(model, optimizer, scheduler, arm, manifest_sha, device,
            epoch=epoch, cursor=cursor, global_step=global_step, best_score=best_score, statistics=statistics))
    common._atomic_json(output / "audit/formal_launch.json", dict(
        status="RUNNING", launched_at=stamp(), arm=arm["name"], manifest_sha256=manifest_sha,
        source_commit=manifest["source"]["commit"], slurm_job_id=os.environ["SLURM_JOB_ID"],
        machine="CmlLab", node=socket.gethostname(), world_size=1,
        microbatch=128, accumulation=3, usual_samples_per_update=384,
        steps_per_epoch=arm["steps_per_epoch"], resume=str(resume) if resume else None))
    frozen_origin = manifest["initialization"]["frozen_parameter_sha256"]
    if frozen_sha256(model) != frozen_origin:
        raise RuntimeError("initial or resumed frozen parameters changed")
    if not resume:
        save(0, 0, statistics, output / "run/epoch00.pt")
        common._atomic_json(output / "results/epoch00.json", dict(
            epoch=0, global_step=0, validation=validate(model, arm, corpus, device),
            role="zero_adaptation_reference"))
    while epoch < arm["epochs"]:
        started = perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        loader, sampler, expected_rows = make_loader(arm, corpus, "train", epoch=epoch, start_batch=cursor)
        seen = [expected_rows[np.concatenate(sampler.batches[:cursor])]] if cursor else []
        for group in accumulation_groups(loader, arm["accumulate"]):
            if any(int(batch["corruption_epoch"]) != epoch for batch in group):
                raise RuntimeError("wrong training corruption epoch")
            batch_stats, record = update(model, optimizer, scheduler, group, arm, device)
            statistics += batch_stats
            seen.extend(np.asarray(batch["row_ids"]) for batch in group)
            cursor += len(group)
            global_step += 1
            if global_step % 10 == 0 or global_step == 1 or cursor == len(sampler.batches):
                common._append_jsonl(output / "logs/progress.jsonl", dict(
                    time=stamp(), epoch=epoch + 1, next_batch=cursor, global_step=global_step,
                    loss=common._unpack_loss_statistics(batch_stats, arm["weights"]), **record))
                print(f"{stamp()} {arm['name']} epoch={epoch+1}/{arm['epochs']} step={global_step} "
                      f"batch={cursor}/{len(sampler.batches)} loss={record['objective']:.6f}", flush=True)
            if global_step % 100 == 0 or stop:
                save(epoch, cursor, statistics)
            if stop:
                common._atomic_json(output / "results/training_status.json", dict(
                    status="INTERRUPTED", time=stamp(), signal=stop[-1], epoch=epoch,
                    next_batch=cursor, global_step=global_step, checkpoint=str(output / "run/last.pt")))
                return
        order = np.concatenate(sampler.batches)
        if not np.array_equal(np.concatenate(seen), expected_rows[order]):
            raise RuntimeError("training duplicated or omitted rows")
        if global_step != (epoch + 1) * arm["steps_per_epoch"]:
            raise RuntimeError("optimizer step count drifted")
        # Save the end-of-training-batches state before validation as well. A
        # resume at cursor=len(batches) reruns validation without another update.
        save(epoch, cursor, statistics)
        validation = validate(model, arm, corpus, device)
        if frozen_sha256(model) != frozen_origin:
            raise RuntimeError("frozen parameters changed during domain adaptation")
        trainable_inventory(model, LoRAConfig(**arm["lora"]))
        score = validation["loss"]["weighted_total"]
        is_best = score < best_score
        best_score = min(best_score, score)
        result = dict(epoch=epoch + 1, global_step=global_step, time=stamp(),
            seconds=perf_counter() - started, train=common._unpack_loss_statistics(statistics, arm["weights"]),
            validation=validation, exact_train_row_coverage=True, train_rows=len(expected_rows),
            peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 1024**3)
        common._atomic_json(output / f"results/epoch{epoch+1:02d}.json", result)
        common._append_jsonl(output / "results/epochs.jsonl", result)
        epoch += 1
        cursor = 0
        statistics.zero_()
        save(epoch, cursor, statistics)
        if is_best:
            save(epoch, cursor, statistics, output / "run/best.pt")
        if epoch in arm["milestones"]:
            save(epoch, cursor, statistics, output / f"run/epoch{epoch:02d}.pt")
        print(f"{stamp()} {arm['name']} completed epoch={epoch}/{arm['epochs']} val={score:.6f}", flush=True)
    common._atomic_json(output / "results/training_status.json", dict(
        status="COMPLETED", time=stamp(), epochs=epoch, global_step=global_step,
        manifest_sha256=manifest_sha, checkpoint=str(output / "run/epoch10.pt"),
        selection="fixed_epoch10; best checkpoint is diagnostic only"))


def run_job(manifest_path: Path, expected_sha: str, arm_name: str, *, resume=False):
    manifest = validate_manifest(manifest_path, expected_sha)
    arm = manifest["arms"][arm_name]
    output = Path(arm["output"])
    output.mkdir(parents=True, exist_ok=True)
    # Advisory lock catches duplicate submissions before any CUDA work.
    import fcntl
    with (output / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completion = output / "results/training_status.json"
        if completion.exists() and json.loads(completion.read_text())["status"] == "COMPLETED":
            raise RuntimeError("this arm already completed; refusing a second training")
        last = output / "run/last.pt"
        if resume and not last.exists():
            raise RuntimeError("requested resume checkpoint is missing")
        if last.exists() and not resume:
            raise RuntimeError("existing training requires explicit --resume")
        try:
            device = gpu_preflight(output)
            smoke(arm, manifest, expected_sha, device)
            train(arm, manifest, expected_sha, device, resume=last if resume and last.exists() else None)
        except BaseException as error:
            common._atomic_json(output / "audit/failure.json", dict(
                status="FAILED", time=stamp(), stage="smoke_or_training", error=repr(error),
                machine="CmlLab", node=socket.gethostname(), slurm_job_id=os.environ.get("SLURM_JOB_ID")))
            raise
