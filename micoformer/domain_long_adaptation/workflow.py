"""单H100长程无标签适配；完整训练预算、固定验证、冻结头与严格恢复。"""
from __future__ import annotations

import gc
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from micoformer.domain_adaptation.workflow import (
    capture_rng, restore_rng, state_equal,
    validate as _legacy_fixed_validation,
)
from micoformer.domain_method_adaptation.modules import frozen_sha256, inventory, inject, parameter_groups
from micoformer.domain_adaptation.prepare import load_origin
from micoformer.domain_pretraining.input import InMemoryFullSupportRowDataset
from micoformer.mpc_unifrac.data import DualCorruptionCollator, SharedEpoch
from micoformer.mpc_unifrac.model import Output
from micoformer.mpc_unifrac.teacher import weighted_unifrac, neighborhood_details
from micoformer.mpc_unifrac import workflow as common
from micoformer.domain_bucket_adaptation.sampling import BucketSampler, TensorCollator
from .prepare import now, RUN, configure_tree

NAMES = ("mlm", "continuous", "presence", "unifrac")
ARGUMENTS = ("genus_ids", "rclr", "padding_mask", "query_mask", "mlm_mask", "negative_genus_ids", "edge_mass")
EXPECTED_UUID = os.environ.get("MCF_TARGET_GPU_UUID", "")


def fixed_validation(model, arm, corpus, device):
    # 跨seed也固定原seed42验证题目；训练mask/组批仍使用各自训练seed。
    validation_arm = copy.deepcopy(arm)
    validation_arm["seed"] = 42
    return _legacy_fixed_validation(model, validation_arm, corpus, device)


def preflight(destination):
    """Direct H100 use, one explicitly inspected UUID, never a Slurm allocation."""
    if socket.gethostname() != "NVH1003":
        raise RuntimeError("expected the approved H100 host NVH1003")
    allowed={"GPU-3d8a1781-fc8e-1855-bff0-6bda65635a07", "GPU-a576c223-bd1d-ae30-dc26-ab90d9053d61"}
    if EXPECTED_UUID not in allowed or os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_UUID:
        raise RuntimeError("one explicitly approved H100 UUID required")
    query=subprocess.check_output(["nvidia-smi","-i",EXPECTED_UUID,"--query-gpu=uuid,name,memory.total","--format=csv,noheader"],text=True)
    apps=subprocess.check_output(["nvidia-smi","-i",EXPECTED_UUID,"--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    if any(v.strip() and v.strip()!=str(os.getpid()) for v in apps.splitlines()):
        raise RuntimeError("GPU occupied; preserve all existing processes")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("one actual BF16 CUDA device required; no fallback")
    device=torch.device("cuda:0");torch.cuda.set_device(device)
    info=torch.cuda.get_device_properties(device)
    if "H100" not in info.name or info.total_memory<78*1024**3:
        raise RuntimeError("expected an 80GB H100")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.90)
    value=torch.ones((64,64),device=device)
    with torch.autocast("cuda",dtype=torch.bfloat16): result=value@value
    if not bool(torch.all(result==64)):raise RuntimeError("CUDA arithmetic failed")
    common._atomic_json(destination,dict(status="PASS",time=now(),machine="H100",node=socket.gethostname(),
        pid=os.getpid(),uuid=EXPECTED_UUID,gpu_query=query,torch=torch.__version__,cuda=torch.version.cuda,
        cpu_affinity=sorted(os.sched_getaffinity(0)),device_count=1))
    return device


def load_model(arm, manifest, device):
    configure_tree()
    model = load_origin(manifest["assets"]["prior"]["path"], Path(manifest["assets"]["origin"]["path"]))
    if common._model_state_sha256(model) != arm["initialization"]["origin_model_state_sha256"]:
        raise RuntimeError("pretrained origin changed")
    common._seed_all(arm["seed"])
    actual = inject(model, arm["adaptation"])
    if actual != arm["trainable_inventory"] or common._model_state_sha256(model) != arm["initialization"]["adapted_model_state_sha256"]:
        raise RuntimeError("method initialization/scope changed")
    model._dapt_prior_record = manifest["assets"]["prior"]
    model._dapt_origin_record = manifest["assets"]["origin"]
    return model.to(device)


def checkpoint(model, optimizer, scheduler, arm, manifest_sha, device, *, epoch, cursor, global_step, best_score, statistics):
    return dict(schema_version=1, checkpoint_kind="domain_method_full_resume", adaptation=arm["adaptation"],
        prior=model._dapt_prior_record, origin=model._dapt_origin_record,
        frozen_parameter_sha256=frozen_sha256(model), manifest_sha256=manifest_sha, arm=arm["name"],
        model_config=model.config.to_dict(), model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(), rng_state=capture_rng(device),
        epoch=epoch, next_batch=cursor, global_step=global_step, best_score=best_score, epoch_statistics=statistics.detach().cpu())


def restore(path, model, optimizer, scheduler, arm, manifest_sha, device):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["checkpoint_kind"] != "domain_method_full_resume" or state["arm"] != arm["name"]
            or state["manifest_sha256"] != manifest_sha or state["adaptation"] != arm["adaptation"]
            or state["model_config"] != model.config.to_dict() or state["prior"] != model._dapt_prior_record
            or state["origin"] != model._dapt_origin_record):
        raise RuntimeError("method checkpoint identity mismatch")
    frozen = frozen_sha256(model)
    model.load_state_dict(state["model_state_dict"], strict=True)
    if frozen_sha256(model) != frozen or state["frozen_parameter_sha256"] != frozen:
        raise RuntimeError("frozen origin changed")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng(state["rng_state"], device)
    return state


def make_optimizer(model, arm):
    optimizer = torch.optim.AdamW(parameter_groups(model, arm),
        lr=arm["learning_rate"], weight_decay=arm["weight_decay"], fused=True)
    scheduler = common._make_scheduler(optimizer, total_steps=arm["total_steps"],
        warmup_ratio=arm["warmup_ratio"], base_lr=arm["learning_rate"], minimum_lr=arm["minimum_lr"])
    return optimizer, scheduler


def make_loader(arm, corpus, epoch, *, cursor=0, workers=None):
    rows = np.load(arm["data"]["paths"]["train"], allow_pickle=False)
    dataset = InMemoryFullSupportRowDataset(corpus, rows)
    ptr = dataset._h5()["X"]["indptr"]
    lengths = np.minimum(ptr[rows+1]-ptr[rows], 512)
    sampler = BucketSampler(lengths, mode=arm.get("sampler", "window4096"), seed=arm["seed"], epoch=epoch,
        batch_size=arm["microbatch"], window=arm["window"], cursor=cursor)
    collator = TensorCollator(DualCorruptionCollator(
        observed_genus_ids=np.load(arm["data"]["observed_genus_ids"], allow_pickle=False),
        seed=arm["seed"], stream=1, shared_epoch=SharedEpoch(epoch)))
    workers = arm["workers"] if workers is None else workers
    kwargs = dict(batch_sampler=sampler, collate_fn=collator, num_workers=workers,
                  pin_memory=True, generator=torch.Generator().manual_seed(7000+epoch))
    if workers:
        kwargs.update(persistent_workers=True, multiprocessing_context="fork", prefetch_factor=2)
    return DataLoader(dataset, **kwargs), sampler, rows


def device_batch(batch, device):
    return {key: torch.as_tensor(batch[key]).to(device, non_blocking=True) for key in ARGUMENTS}


def split_backward(model, arguments, weights, *, bf16=True):
    """MLM图先释放；Query只前向一次，三个损失共享Query图；不在分支之间step。"""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        mlm, mc = model._mlm_loss(arguments["genus_ids"], arguments["rclr"],
            arguments["padding_mask"], arguments["mlm_mask"])
        weighted = weights["mlm"]*mlm
    weighted.backward()
    m = mlm.detach()
    del mlm, weighted
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        q = model._query_output(arguments["genus_ids"], arguments["rclr"],
            arguments["padding_mask"], arguments["query_mask"], arguments["negative_genus_ids"])
        c,p,cc,pc = model._query_losses(q)
        u,entropy,kl = neighborhood_details(q.sample_h, weighted_unifrac(arguments["edge_mass"], model.unifrac_lengths))
        loss = weights["continuous"]*c + weights["presence"]*p + weights["unifrac"]*u
    loss.backward()
    return Output(m,c.detach(),p.detach(),u.detach(),mc,cc,pc,len(arguments["genus_ids"]),
                  entropy.detach(),kl.detach())


def update(model, optimizer, scheduler, batch, arm, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = split_backward(model, device_batch(batch,device), arm["weights"])
    norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                         arm["gradient_clip"], error_if_nonfinite=True)
    if float(norm)>1e4:
        raise RuntimeError("gradient norm exceeds the predeclared numerical stop threshold")
    objective = sum(arm["weights"][n]*getattr(output,n+"_loss") for n in NAMES)
    if not bool(torch.isfinite(objective)):
        raise RuntimeError("nonfinite loss")
    lr = float(optimizer.param_groups[0]["lr"])
    optimizer.step()
    scheduler.step()
    stats = common._loss_statistics_tensor(output, arm["weights"], device=device)
    return stats, dict(objective=float(objective), gradient_norm_preclip=float(norm),
                      lr_used=lr, lr_next=float(optimizer.param_groups[0]["lr"]),
                      group_lrs={g["group_name"]:float(g["lr"]) for g in optimizer.param_groups})


def save_checkpoint(model, optimizer, scheduler, arm, sha, device, epoch, cursor,
                    global_step, best_score, stats, path, counters):
    payload = checkpoint(model,optimizer,scheduler,arm,sha,device,epoch=epoch,cursor=cursor,
                         global_step=global_step,best_score=best_score,statistics=stats)
    payload["bucket_counters"] = counters
    common._atomic_torch(path,payload)


def train(arm, manifest, sha, device, *, resume=False):
    destination = Path(arm["output"])
    destination.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (destination/"run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_file = destination/"results/training_status.json"
        if status_file.exists() and json.loads(status_file.read_text())["status"] == "COMPLETED":
            if resume:
                return
            raise RuntimeError("refusing to overwrite completed arm")
        last = destination/"run/last.pt"
        if last.exists() and not resume:
            raise RuntimeError("existing state requires explicit resume")
        model = load_model(arm, manifest, device)
        optimizer,scheduler = make_optimizer(model,arm)
        common._seed_all(arm["seed"])
        epoch=cursor=step=0
        best=math.inf
        stats=torch.zeros(12,device=device,dtype=torch.float64)
        counters=dict(samples=0,tokens=0,padded_positions=0)
        if resume and last.exists():
            state=restore(last,model,optimizer,scheduler,arm,sha,device)
            epoch,cursor,step=state["epoch"],state["next_batch"],state["global_step"]
            best=state["best_score"]
            stats.copy_(state["epoch_statistics"].to(device))
            counters=state["bucket_counters"]
            del state
        signals=[]
        for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,lambda signum,frame:signals.append(signum))
        def save(path=None):
            save_checkpoint(model,optimizer,scheduler,arm,sha,device,epoch,cursor,step,best,
                stats,path or last,counters.copy())
        corpus=manifest["assets"]["corpus"]["path"]
        common._atomic_json(destination/"audit/formal_launch.json",dict(
            status="RUNNING",time=now(),manifest_sha256=sha,source_commit=manifest["source"]["commit"],
            arm=arm["name"],slurm_job_id=None,
            slurm_step_id=os.environ.get("SLURM_STEP_ID"),pid=os.getpid(),uuid=EXPECTED_UUID,
            machine="H100",world_size=1,batch_size=arm["microbatch"],accumulate=1,epochs=arm["epochs"],total_steps=arm["total_steps"]))
        if not last.exists():
            save(destination/"run/epoch00.pt")
            common._atomic_json(destination/"results/epoch00.json",dict(epoch=0,global_step=0,
                validation=fixed_validation(model,arm,corpus,device),role="zero_adaptation"))
        while epoch < arm["epochs"]:
            epoch_start=perf_counter()
            loader,sampler,rows=make_loader(arm,corpus,epoch,cursor=cursor)
            order=np.concatenate(sampler.batches)
            seen=[rows[np.concatenate(sampler.batches[:cursor])]] if cursor else []
            torch.cuda.synchronize()
            train_start=perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            start_cursor=cursor
            for batch in loader:
                if int(batch["corruption_epoch"]) != epoch:
                    raise RuntimeError("wrong corruption epoch")
                batch_stats,record=update(model,optimizer,scheduler,batch,arm,device)
                stats+=batch_stats
                batch_rows=batch["row_ids"].numpy()
                seen.append(batch_rows)
                counters["samples"]+=len(batch_rows)
                counters["tokens"]+=int((~batch["padding_mask"]).sum())
                counters["padded_positions"]+=batch["padding_mask"].numel()
                cursor+=1
                step+=1
                if step%20==0 or cursor==len(sampler.batches) or step==1:
                    common._append_jsonl(destination/"logs/progress.jsonl",dict(
                        time=now(),epoch=epoch+1,global_step=step,next_batch=cursor,
                        loss=common._unpack_loss_statistics(batch_stats,arm["weights"]),**record))
                    print(f"{now()} {arm['name']} epoch={epoch+1}/{arm['epochs']} step={step}/{arm['total_steps']} batch={cursor}/{arm['steps_per_epoch']} loss={record['objective']:.6f}",flush=True)
                if step%500==0 or signals:
                    save()
                if signals:
                    common._atomic_json(status_file,dict(status="INTERRUPTED",time=now(),epoch=epoch,
                        next_batch=cursor,global_step=step,signal=signals[-1]))
                    return False
            torch.cuda.synchronize()
            train_seconds=perf_counter()-train_start
            if (not np.array_equal(np.concatenate(seen),rows[order]) or counters["samples"]!=len(rows)
                    or step!=(epoch+1)*arm["steps_per_epoch"]):
                raise RuntimeError("row coverage or update budget drifted")
            validation=fixed_validation(model,arm,corpus,device)
            if frozen_sha256(model)!=arm["initialization"]["frozen_parameter_sha256"]:
                raise RuntimeError("frozen weights changed")
            inventory(model,arm["adaptation"])
            score=validation["loss"]["weighted_total"]
            improved=score<best
            best=min(best,score)
            result=dict(epoch=epoch+1,global_step=step,time=now(),train_seconds=train_seconds,
                resumed_partial_epoch=start_cursor>0,measured_batches=cursor-start_cursor,
                samples=len(rows),tokens=counters["tokens"],padding_fraction=1-counters["tokens"]/counters["padded_positions"],
                samples_per_second=(len(rows)/train_seconds) if start_cursor==0 else None,
                train=common._unpack_loss_statistics(stats,arm["weights"]),validation=validation,
                sampler_order_sha256=sampler.order_sha256,actual_rows_sha256=hashlib.sha256(np.concatenate(seen).tobytes()).hexdigest(),
                exact_train_row_coverage=True,frozen_weights_unchanged=True,
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/1024**3)
            epoch+=1
            cursor=0
            stats.zero_()
            counters=dict(samples=0,tokens=0,padded_positions=0)
            save()
            if epoch in arm["milestones"]:
                save(destination/f"run/epoch{epoch:02d}.pt")
            if improved:
                save(destination/"run/best.pt")
            result["epoch_wall_seconds"]=perf_counter()-epoch_start
            common._atomic_json(destination/f"results/epoch{epoch:02d}.json",result)
            common._append_jsonl(destination/"results/epochs.jsonl",result)
            print(f"{now()} {arm['name']} completed epoch={epoch}/{arm['epochs']} train={train_seconds:.2f}s wall={result['epoch_wall_seconds']:.2f}s val={score:.6f}",flush=True)
            del loader
            if epoch in [10,20,30,50]:
                import sys
                command=[sys.executable,"-m","micoformer.domain_long_adaptation.runner","extract",
                         "--arm",arm["name"],"--epoch",str(epoch)]
                subprocess.run(command,check=True)
        common._atomic_json(status_file,dict(status="COMPLETED",time=now(),epochs=arm["epochs"],global_step=step,
            manifest_sha256=sha,checkpoint=str(destination/f"run/epoch{arm['epochs']:02d}.pt"),selection="fixed_epoch50"))
        del model,optimizer,scheduler
        gc.collect()
        torch.cuda.empty_cache()
        return True
