"""Three-rank data parallel LoRA in an existing, preserved GPU allocation.

Keep the original global groups of three B128 microbatches, including their
local UniFrac pools. DDP averages gradients, so actual component target counts
are reduced before applying the existing world-size correction.
"""
from __future__ import annotations

import copy
from datetime import timedelta
import fcntl
import gc
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import subprocess
from time import perf_counter

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from . import workflow as single
from .lora import LoRAConfig, frozen_sha256, trainable_inventory
from .prepare import validate_manifest
from micoformer.domain_pretraining.data import ExactBatchSampler
from micoformer.domain_pretraining.input import InMemoryFullSupportRowDataset
from micoformer.mpc_unifrac import workflow as common
from micoformer.mpc_unifrac.data import DualCorruptionCollator, SharedEpoch, batch_to_device
from micoformer.mpc_unifrac.support import distributed_training_loss
from micoformer.mpc_unifrac.teacher import weighted_unifrac, neighborhood_details


class ForkserverEpoch(SharedEpoch):
    """Use a semaphore compatible with the NCCL-safe loader start context."""
    def __init__(self, initial=0):
        self._value = mp.get_context('forkserver').Value('q', int(initial))


def rank_batches(size, *, seed, epoch, shuffle, rank, world=3, start_batch=0):
    sampler = ExactBatchSampler(size, seed=seed, epoch=epoch, shuffle=shuffle,
                                batch_size=128, start_batch=start_batch)
    if shuffle and (len(sampler.batches) % world or start_batch % world):
        raise RuntimeError("training must have complete three-rank groups; no padding or dropping")
    return sampler, sampler.batches[start_batch + rank::world]


def loader(arm, corpus, split, rank, *, epoch, cursor=0, workers=None):
    rows = np.load(arm['data']['paths'][split], allow_pickle=False)
    sampler, batches = rank_batches(len(rows), seed=arm['seed'], epoch=epoch,
        shuffle=split == 'train', rank=rank, start_batch=cursor)
    dataset = InMemoryFullSupportRowDataset(corpus, rows)
    collator = DualCorruptionCollator(observed_genus_ids=np.load(
        arm['data']['observed_genus_ids'], allow_pickle=False), seed=arm['seed'],
        stream=1 if split == 'train' else 2,
        shared_epoch=ForkserverEpoch(epoch if split == 'train' else 0))
    workers = arm['workers'] if workers is None else workers
    result = DataLoader(dataset, batch_sampler=batches, collate_fn=collator,
        num_workers=workers, pin_memory=True,
        # NCCL is not fork-safe; fresh workers lazily load the same bounded CSR.
        multiprocessing_context='forkserver' if workers else None,
        prefetch_factor=2 if workers else None,
        generator=torch.Generator().manual_seed(arm['seed'] + epoch + rank * 1000
            + (0 if split == 'train' else 10000)))
    return result, sampler, rows


def update(ddp, optimizer, scheduler, batch, arm, device):
    ddp.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        output = ddp(**batch_to_device(batch, device))
        loss = distributed_training_loss(output, arm['weights'], dist.get_world_size(), device)
    finite = torch.tensor(int(torch.isfinite(loss)), device=device)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if not finite.item():
        raise RuntimeError('nonfinite distributed loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_([p for p in ddp.parameters() if p.requires_grad],
        arm['gradient_clip'], error_if_nonfinite=True)
    stats = common._loss_statistics_tensor(output, arm['weights'], device=device)
    dist.all_reduce(stats)
    objective = loss.detach().double() / dist.get_world_size()
    dist.all_reduce(objective)
    counts = torch.tensor([getattr(output, name + '_count') for name in
        ('mlm', 'continuous', 'presence', 'unifrac')], device=device, dtype=torch.int64)
    dist.all_reduce(counts)
    lr = float(optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return stats, dict(objective=float(objective), gradient_norm_preclip=float(norm),
        lr_used=lr, lr_next=float(optimizer.param_groups[0]['lr']),
        head_lr_next=0.0,
        target_counts=counts.cpu().tolist(), samples=int(counts[3]))


def all_rows(local, expected):
    gathered = [None] * 3
    dist.all_gather_object(gathered, np.concatenate(local) if local else np.empty(0, np.int64))
    combined = np.concatenate(gathered)
    if not np.array_equal(np.sort(combined), np.sort(expected)):
        raise RuntimeError('distributed coverage omitted or duplicated rows')


@torch.inference_mode()
def validate(model, arm, corpus, device, rank):
    model.eval()
    batches, _, rows = loader(arm, corpus, 'val', rank, epoch=0)
    statistics = torch.zeros(12, device=device, dtype=torch.float64)
    clean = torch.zeros(4, device=device, dtype=torch.float64)
    sum_h = torch.zeros(512, device=device, dtype=torch.float64)
    sum_outer = torch.zeros((512, 512), device=device, dtype=torch.float64)
    seen = []
    for batch in batches:
        seen.append(batch['row_ids'])
        args = batch_to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            result = model(**args)
            _, mean = model.unmasked_representations(args['genus_ids'], args['rclr'], args['padding_mask'])
        ce, entropy, kl = neighborhood_details(mean, weighted_unifrac(args['edge_mass'], model.unifrac_lengths))
        statistics += common._loss_statistics_tensor(result, arm['weights'], device=device)
        n = len(mean)
        clean += torch.stack([ce.double() * n, entropy.double() * n, kl.double() * n,
                              torch.as_tensor(n, device=device)])
        h = mean.float().double()
        sum_h += h.sum(0)
        sum_outer += h.T @ h
    for value in (statistics, clean, sum_h, sum_outer):
        dist.all_reduce(value)
    all_rows(seen, rows)
    return dict(loss=common._unpack_loss_statistics(statistics, arm['weights']),
        clean_unifrac_diagnostics=dict(zip(['ce', 'teacher_entropy', 'kl'], (clean[:3] / clean[3]).cpu().tolist())),
        representation=common._spectral_summary(len(rows), sum_h.cpu(), sum_outer.cpu()),
        row_count=len(rows), row_unique_count=len(np.unique(rows)), exact_row_coverage=True)


def save(path, model, optimizer, scheduler, arm, sha, device, rank, *, epoch, cursor,
         global_step, best_score, statistics):
    rng = [None] * 3
    dist.all_gather_object(rng, single.capture_rng(device))
    if rank == 0:
        state = single.checkpoint(model, optimizer, scheduler, arm, sha, device,
            epoch=epoch, cursor=cursor, global_step=global_step, best_score=best_score, statistics=statistics)
        state.update(world_size=3, rng_states=rng, per_rank_accumulation=1,
                     activation_recomputation=model.backbone.encoder.grad_checkpointing,
                     dropout_seed_policy='seed42+rank; adapter initialization remains seed42')
        common._atomic_torch(path, state)
    dist.barrier()


def restore(path, model, optimizer, scheduler, arm, sha, device, rank):
    state = single.restore(path, model, optimizer, scheduler, arm, sha, device)
    if state.get('world_size') != 3 or len(state.get('rng_states', [])) != 3:
        raise RuntimeError('resume requires the same three-rank topology and RNG states')
    single.restore_rng(state['rng_states'][rank], device)
    return state


def rank_ordered_mean(state, bucket):
    """Keep floating summation order independent of DDP bucket rebuilding.

    Only ~8.5MB of trainable gradients are communicated. Gathering them in rank
    order avoids position-dependent NCCL summation order when a restored DDP
    reducer reconstructs its buckets; the mathematical gradient mean is intact.
    """
    buffer = bucket.buffer()
    values = [torch.empty_like(buffer) for _ in range(dist.get_world_size())]
    dist.all_gather(values, buffer)
    buffer.copy_(values[0])
    for value in values[1:]:
        buffer.add_(value)
    buffer.div_(dist.get_world_size())
    future = torch.futures.Future()
    future.set_result(buffer)
    return future


def wrap(model, device):
    ddp = DDP(model, device_ids=[device.index], output_device=device.index,
              broadcast_buffers=False, gradient_as_bucket_view=True)
    ddp.register_comm_hook(None, rank_ordered_mean)
    return ddp


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_clone(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu_clone(v) for v in value)
    return copy.deepcopy(value)


def numeric_close(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(numeric_close(a[k], b[k]) for k in a)
    return bool(np.isclose(a, b, atol=2e-6, rtol=2e-6))


def smoke(arm, manifest, sha, device, rank):
    # Existing single-device gates independently exercise every assigned GPU.
    local_arm = copy.deepcopy(arm)
    local_arm['output'] = str(Path(arm['output']) / 'audit' / f'rank{rank}_single_gate')
    single.smoke(local_arm, manifest, sha, device)
    dist.barrier()
    corpus = manifest['assets']['corpus']['path']
    batches, _, _ = loader(arm, corpus, 'train', rank, epoch=0, workers=0)
    batch = next(iter(batches))
    model = single.load_model(arm, manifest, device)
    ddp = wrap(model, device)
    optimizer, scheduler = single.make_optimizer(model, arm)
    common._seed_all(arm['seed'] + rank)
    stats, first = update(ddp, optimizer, scheduler, batch, arm, device)
    checkpoint = Path(arm['output']) / 'audit/ddp_smoke_resume.pt'
    save(checkpoint, model, optimizer, scheduler, arm, sha, device, rank,
         epoch=0, cursor=3, global_step=1, best_score=math.inf, statistics=stats)
    # Unequal local counts, including a B2 relation pool, exercise DDP weighting.
    tail = {k: (v[:127 if rank == 0 else 2] if isinstance(v, np.ndarray)
                 and v.ndim and v.shape[0] == len(batch['row_ids']) else v)
            for k, v in batch.items()}
    _, expected_record = update(ddp, optimizer, scheduler, tail, arm, device)
    expected = cpu_clone((model.state_dict(), optimizer.state_dict(), scheduler.state_dict()))
    del ddp, model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    model = single.load_model(arm, manifest, device)
    ddp = wrap(model, device)
    optimizer, scheduler = single.make_optimizer(model, arm)
    state = restore(checkpoint, model, optimizer, scheduler, arm, sha, device, rank)
    assert state['next_batch'] == 3 and state['global_step'] == 1
    del state
    _, replay = update(ddp, optimizer, scheduler, tail, arm, device)
    replay_states = (model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
    checks = dict(model_exact=single.state_equal(expected[0], replay_states[0]),
        optimizer_exact=single.state_equal(expected[1], replay_states[1]),
        scheduler_exact=single.state_equal(expected[2], replay_states[2]),
        update_diagnostics_exact=replay == expected_record)
    common._atomic_json(Path(arm['output']) / f'audit/strict_replay_rank{rank}.json', dict(
        status='PASS' if all(checks.values()) else 'FAILED', checks=checks,
        expected_update=expected_record, replay_update=replay,
        gradient_reduction='rank_ordered_mean'))
    if not all(checks.values()):
        raise RuntimeError('DDP strict next-update replay failed')
    frozen = frozen_sha256(model)
    if frozen != arm['initialization']['frozen_parameter_sha256']:
        raise RuntimeError('DDP updated frozen encoder parameters')
    hashes = [None] * 3
    dist.all_gather_object(hashes, common._model_state_sha256(model))
    if len(set(hashes)) != 1:
        raise RuntimeError('rank model parameters diverged')
    validation = validate(model, arm, corpus, device, rank)
    if rank == 0:
        reference = single.validate(model, arm, corpus, device, workers=0)
        if not numeric_close(validation['loss'], reference['loss']):
            raise RuntimeError('distributed validation changed original loss aggregation')
        common._atomic_json(Path(arm['output']) / 'audit/ddp_cuda_smoke.json', dict(
            status='PASS', time=single.stamp(), manifest_sha256=sha, world_size=3,
            actual_microbatch=128, first_update=first, tail_microbatches=[127, 2, 2],
            strict_next_update_exact=True, ranks_equal=True, frozen_weights_unchanged=True,
            full_validation=validation, serial_validation_matches=True))
    dist.barrier()
    del ddp, model, optimizer, scheduler, expected
    gc.collect()
    torch.cuda.empty_cache()


def preflight(output, execution):
    rank = int(os.environ['LOCAL_RANK'])
    job = os.environ.get('SLURM_JOB_ID')
    cgroup = Path('/proc/self/cgroup').read_text()
    if job != str(execution['placeholder_job_id']) or f'job_{job}' not in cgroup:
        raise RuntimeError('worker must remain inside the approved placeholder cgroup')
    if int(os.environ['WORLD_SIZE']) != 3 or not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError('exactly three allocated CUDA GPUs are required')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != execution['gpu_uuids']:
        raise RuntimeError('GPU UUID binding differs from the approved allocation')
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    info = torch.cuda.get_device_properties(device)
    if 'A100' not in info.name or info.total_memory < 38 * 1024**3 or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A100 >=38GiB and BF16 required')
    x = torch.ones((64, 64), device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        y = x @ x
    torch.cuda.synchronize(device)
    if not bool((y == 64).all()):
        raise RuntimeError('CUDA preflight failed')
    common._atomic_json(output / f'audit/gpu_preflight_rank{rank}.json', dict(
        status='PASS', time=single.stamp(), machine='CmlLab', node=socket.gethostname(),
        job_id=job, rank=rank, gpu_uuid=visible[rank], cgroup=cgroup,
        name=info.name, total_memory_bytes=info.total_memory))
    dist.init_process_group('nccl', timeout=timedelta(minutes=5), device_id=device)
    probe = torch.tensor(float(rank + 1), device=device)
    dist.all_reduce(probe)
    if probe.item() != 6:
        raise RuntimeError('three-GPU NCCL reduction failed')
    return device, rank


def train(arm, manifest, sha, device, rank, *, resume=False):
    output = Path(arm['output'])
    model = single.load_model(arm, manifest, device)
    ddp = wrap(model, device)
    optimizer, scheduler = single.make_optimizer(model, arm)
    common._seed_all(arm['seed'] + rank)
    epoch = cursor = step = 0
    best = math.inf
    stats = torch.zeros(12, device=device, dtype=torch.float64)
    if resume:
        state = restore(output / 'run/last.pt', model, optimizer, scheduler, arm, sha, device, rank)
        epoch, cursor, step = state['epoch'], state['next_batch'], state['global_step']
        best = state['best_score']
        stats.copy_(state['epoch_statistics'].to(device))
        del state
    stop = []
    def request_stop(signum, frame):
        stop.append(signum)
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, request_stop)
    def persist(path=None):
        save(path or output / 'run/last.pt', model, optimizer, scheduler, arm, sha, device, rank,
             epoch=epoch, cursor=cursor, global_step=step, best_score=best, statistics=stats)
    if rank == 0:
        common._atomic_json(output / 'audit/formal_launch.json', dict(
            status='RUNNING', launched_at=single.stamp(), arm=arm['name'], manifest_sha256=sha,
            source_commit=manifest['source']['commit'], placeholder_job_id=os.environ['SLURM_JOB_ID'],
            node=socket.gethostname(), world_size=3, per_rank_microbatch=128,
            activation_recomputation=model.backbone.encoder.grad_checkpointing,
            per_rank_accumulation=1, usual_samples_per_update=384, resume=resume))
    if not resume:
        persist(output / 'run/epoch00.pt')
        validation = validate(model, arm, manifest['assets']['corpus']['path'], device, rank)
        if rank == 0:
            common._atomic_json(output / 'results/epoch00.json', dict(epoch=0,
                global_step=0, validation=validation, role='zero_adaptation_reference'))
    launched = perf_counter()
    while epoch < arm['epochs']:
        started = perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        batches, sampler, rows = loader(arm, manifest['assets']['corpus']['path'], 'train', rank,
                                       epoch=epoch, cursor=cursor)
        seen = [rows[np.concatenate(sampler.batches[rank:cursor:3])]] if cursor > rank else []
        for batch in batches:
            if int(batch['corruption_epoch']) != epoch:
                raise RuntimeError('wrong corruption epoch')
            batch_stats, record = update(ddp, optimizer, scheduler, batch, arm, device)
            stats += batch_stats
            seen.append(np.asarray(batch['row_ids']))
            cursor += 3
            step += 1
            if rank == 0 and (step % 10 == 0 or step == 1 or cursor == len(sampler.batches)):
                common._append_jsonl(output / 'logs/progress.jsonl', dict(
                    time=single.stamp(), epoch=epoch + 1, global_step=step, next_batch=cursor,
                    loss=common._unpack_loss_statistics(batch_stats, arm['weights']), **record))
                print(f"{single.stamp()} {arm['name']} epoch={epoch+1}/10 step={step} loss={record['objective']:.6f}", flush=True)
            flag = torch.tensor(int(bool(stop) or perf_counter() - launched > 24 * 3600), device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            if step % 100 == 0 or flag.item():
                persist()
            if flag.item():
                if rank == 0:
                    common._atomic_json(output / 'results/training_status.json', dict(status='INTERRUPTED',
                        time=single.stamp(), epoch=epoch, next_batch=cursor, global_step=step,
                        checkpoint=str(output / 'run/last.pt'), allocation_preserved=True))
                return
        all_rows(seen, rows)
        if step != (epoch + 1) * arm['steps_per_epoch']:
            raise RuntimeError('global optimizer step count changed')
        persist()
        validation = validate(model, arm, manifest['assets']['corpus']['path'], device, rank)
        if frozen_sha256(model) != arm['initialization']['frozen_parameter_sha256']:
            raise RuntimeError('frozen parameters changed')
        trainable_inventory(model, LoRAConfig(**arm['lora']))
        score = validation['loss']['weighted_total']
        improved = score < best
        best = min(best, score)
        peaks = [None] * 3
        dist.all_gather_object(peaks, torch.cuda.max_memory_allocated(device) / 1024**3)
        result = dict(epoch=epoch + 1, global_step=step, time=single.stamp(),
            seconds=perf_counter() - started, train=common._unpack_loss_statistics(stats, arm['weights']),
            validation=validation, exact_train_row_coverage=True, train_rows=len(rows),
            peak_allocated_gib_by_rank=peaks, world_size=3)
        if rank == 0:
            common._atomic_json(output / f'results/epoch{epoch+1:02d}.json', result)
            common._append_jsonl(output / 'results/epochs.jsonl', result)
        epoch += 1
        cursor = 0
        stats.zero_()
        persist()
        if improved:
            persist(output / 'run/best.pt')
        if epoch in arm['milestones']:
            persist(output / f'run/epoch{epoch:02d}.pt')
        if rank == 0:
            print(f"{single.stamp()} {arm['name']} completed epoch={epoch}/10 val={score:.6f}", flush=True)
    if rank == 0:
        common._atomic_json(output / 'results/training_status.json', dict(status='COMPLETED',
            time=single.stamp(), epochs=10, global_step=step, manifest_sha256=sha,
            checkpoint=str(output / 'run/epoch10.pt'), world_size=3, allocation_preserved=True,
            selection='fixed_epoch10; best is diagnostic only'))


def run(manifest_path, sha, arm_name, *, resume=False):
    manifest = validate_manifest(manifest_path, sha)
    execution = manifest['execution']
    if execution['world_size'] != 3 or execution['per_rank_accumulation'] != 1:
        raise RuntimeError('invalid three-card execution contract')
    arm = manifest['arms'][arm_name]
    output = Path(arm['output'])
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ['LOCAL_RANK'])
    with (output / f'rank{rank}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        last = output / 'run/last.pt'
        status = output / 'results/training_status.json'
        if status.exists() and json.loads(status.read_text())['status'] == 'COMPLETED':
            raise RuntimeError('arm already completed; refusing duplicate training')
        if last.exists() != resume:
            raise RuntimeError('existing checkpoint requires explicit matching --resume')
        try:
            device, rank = preflight(output, execution)
            smoke(arm, manifest, sha, device, rank)
            train(arm, manifest, sha, device, rank, resume=resume)
            dist.barrier()
        except BaseException as error:
            common._atomic_json(output / f'audit/failure_rank{rank}.json', dict(status='FAILED',
                time=single.stamp(), error=repr(error), rank=rank, allocation_preserved=True))
            raise
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
