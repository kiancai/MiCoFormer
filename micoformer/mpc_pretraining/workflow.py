"""Fail-closed three-rank DDP workflow for full-data MPC pretraining."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data import (
    DualCorruptionCollator,
    FullSupportRowDataset,
    SharedEpoch,
    batch_to_device,
)
from .model import MPCModelConfig, MPCPretrainingModel


SCHEMA_VERSION = 1
TRAIN_ROWS = 2_030_925
VAL_ROWS = 10_206
CORPUS_SHAPE = (3_334_541, 8_114)
EXPECTED_CORPUS_SHA256 = "f10ffe14d2e4384a738e84a9d26694a8e46055520cc30fd5e9fb1d8415d6d78c"
EXPECTED_PRIOR_SHA256 = "6a2c08d216efb90499720eec5aa872df5275a62d5e8052760a5105765262262c"
EXPECTED_TRAIN_SHA256 = "7b85c30341c8afc1720c18789514e1aa7adf4171429f25f8c03badca0aa79f11"
EXPECTED_VAL_SHA256 = "03656a2926f9e849b82cd9f82b2bf63a95d1ce5883f177bdeca2949b19ff9124"


@dataclass(frozen=True)
class MPCRunConfig:
    corpus: Path
    train_rows: Path
    val_rows: Path
    prior_assets: Path
    output_root: Path
    seed: int = 42
    batch_size_per_rank: int = 128
    workers_per_rank: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    max_epochs: int = 50
    min_epochs: int = 10
    warmup_ratio: float = 0.02
    minimum_lr: float = 1e-6
    early_stopping_patience: int = 5
    relative_min_improvement: float = 0.001
    calibration_batches: int = 12

    def __post_init__(self) -> None:
        fixed = (
            self.seed,
            self.batch_size_per_rank,
            self.workers_per_rank,
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip,
            self.max_epochs,
            self.min_epochs,
            self.warmup_ratio,
            self.minimum_lr,
            self.early_stopping_patience,
            self.relative_min_improvement,
            self.calibration_batches,
        )
        expected = (42, 128, 2, 3e-4, 1e-2, 1.0, 50, 10, 0.02, 1e-6, 5, 0.001, 12)
        if fixed != expected:
            raise ValueError("the user-approved first full-data training contract drifted")

    def normalized(self) -> "MPCRunConfig":
        return MPCRunConfig(
            corpus=self.corpus.resolve(),
            train_rows=self.train_rows.resolve(),
            val_rows=self.val_rows.resolve(),
            prior_assets=self.prior_assets.resolve(),
            output_root=self.output_root.resolve(),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for name in ("corpus", "train_rows", "val_rows", "prior_assets", "output_root"):
            payload[name] = str(payload[name])
        return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files() -> list[Path]:
    package = Path(__file__).resolve().parent
    repo = package.parents[1]
    experiment_scripts = repo.parent / "tmp/20260814_mpc_full_data_pretraining/scripts"
    files = [
        package / "__init__.py",
        package / "data.py",
        package / "model.py",
        package / "workflow.py",
        repo / "scripts/2.train_mpc.py",
        repo / "micoformer/fullscale_relation_pretraining/model.py",
        repo / "micoformer/models/attn_bias.py",
        repo / "micoformer/models/heads.py",
        experiment_scripts / "run_common.sh",
        experiment_scripts / "run_prepare.sh",
        experiment_scripts / "run_calibrate.sh",
        experiment_scripts / "run_smoke.sh",
        experiment_scripts / "run_train.sh",
    ]
    if not all(path.is_file() for path in files):
        missing = [str(path) for path in files if not path.is_file()]
        raise RuntimeError(f"MPC source files are missing: {missing}")
    return files


def _manifest_payload(config: MPCRunConfig, *, hash_corpus: bool) -> dict[str, Any]:
    config = config.normalized()
    train = np.load(config.train_rows, mmap_mode="r", allow_pickle=False)
    val = np.load(config.val_rows, mmap_mode="r", allow_pickle=False)
    if train.shape != (TRAIN_ROWS,) or val.shape != (VAL_ROWS,):
        raise RuntimeError("canonical train/validation counts drifted")
    if np.intersect1d(train, val).size:
        raise RuntimeError("canonical train and validation splits overlap")
    with h5py.File(config.corpus, "r") as handle:
        corpus_shape = tuple(int(x) for x in np.asarray(handle["X"].attrs["shape"]))
    if corpus_shape != CORPUS_SHAPE:
        raise RuntimeError(f"canonical corpus shape drifted: {corpus_shape}")
    with np.load(config.prior_assets, allow_pickle=False) as archive:
        if archive["pp_table"].shape != (8_116, 512):
            raise RuntimeError("fixed PP candidate table shape drifted")
        observed = np.asarray(archive["observed_genus_ids"])
        if observed.ndim != 1 or observed.dtype != np.int64 or np.any(observed < 2):
            raise RuntimeError("observed training vocabulary drifted")
    sources = {
        str(path): sha256_file(path)
        for path in _source_files()
    }
    assets = {
        "train_rows": {
            "path": str(config.train_rows),
            "sha256": sha256_file(config.train_rows),
            "count": int(train.size),
        },
        "val_rows": {
            "path": str(config.val_rows),
            "sha256": sha256_file(config.val_rows),
            "count": int(val.size),
        },
        "prior_assets": {
            "path": str(config.prior_assets),
            "sha256": sha256_file(config.prior_assets),
            "observed_genus_ids": int(observed.size),
        },
        "corpus": {
            "path": str(config.corpus),
            "sha256": sha256_file(config.corpus) if hash_corpus else EXPECTED_CORPUS_SHA256,
            "shape": list(corpus_shape),
            "size": config.corpus.stat().st_size,
        },
    }
    expected_hashes = {
        "train_rows": EXPECTED_TRAIN_SHA256,
        "val_rows": EXPECTED_VAL_SHA256,
        "prior_assets": EXPECTED_PRIOR_SHA256,
        "corpus": EXPECTED_CORPUS_SHA256,
    }
    for name, expected in expected_hashes.items():
        if assets[name]["sha256"] != expected:
            raise RuntimeError(f"{name} SHA256 drifted")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "SEALED",
        "config": config.to_dict(),
        "model_config": MPCModelConfig().to_dict(),
        "sources": sources,
        "assets": assets,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def prepare_run(config: MPCRunConfig) -> Path:
    config = config.normalized()
    manifest = _manifest_payload(config, hash_corpus=True)
    path = config.output_root / "audit/sealed_run_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("refusing to replace a different sealed run manifest")
    else:
        _atomic_json(path, manifest)
    return path


def validate_sealed_run(config: MPCRunConfig) -> dict[str, Any]:
    config = config.normalized()
    path = config.output_root / "audit/sealed_run_manifest.json"
    if not path.is_file():
        raise RuntimeError("run prepare before calibration, smoke, or training")
    sealed = json.loads(path.read_text(encoding="utf-8"))
    current = _manifest_payload(config, hash_corpus=True)
    if sealed != current:
        raise RuntimeError("sealed source/data/config manifest no longer matches the run")
    return sealed


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_distributed() -> tuple[int, int, int, torch.device]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 3:
        raise RuntimeError(
            f"the formal run requires exactly three visible CUDA devices, got {torch.cuda.device_count()}"
        )
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world != 3 or local_rank not in (0, 1, 2):
        raise RuntimeError(f"expected single-node world=3, got world={world}, local_rank={local_rank}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if "A100" not in torch.cuda.get_device_name(device):
        raise RuntimeError("the formal run may use only the allocated A100 devices")
    return rank, world, local_rank, device


def _load_prior(config: MPCRunConfig) -> tuple[Tensor, np.ndarray]:
    with np.load(config.prior_assets, allow_pickle=False) as archive:
        table = torch.from_numpy(np.asarray(archive["pp_table"])).float()
        observed = np.asarray(archive["observed_genus_ids"], dtype=np.int64)
    return table, observed


@dataclass
class _LoaderBundle:
    train_loader: DataLoader[dict[str, np.ndarray]]
    val_loader: DataLoader[dict[str, np.ndarray]]
    train_sampler: DistributedSampler[int]
    val_sampler: DistributedSampler[int]
    train_epoch: SharedEpoch
    val_epoch: SharedEpoch
    train_rows: np.ndarray
    val_rows: np.ndarray


def _build_loaders(
    config: MPCRunConfig,
    *,
    rank: int,
    world: int,
    observed: np.ndarray,
    batch_size: int | None = None,
    workers: int | None = None,
) -> _LoaderBundle:
    train_rows = np.asarray(np.load(config.train_rows, mmap_mode="r", allow_pickle=False))
    val_rows = np.asarray(np.load(config.val_rows, mmap_mode="r", allow_pickle=False))
    train_dataset = FullSupportRowDataset(config.corpus, train_rows)
    val_dataset = FullSupportRowDataset(config.corpus, val_rows)
    train_sampler: DistributedSampler[int] = DistributedSampler(
        train_dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=False,
    )
    val_sampler: DistributedSampler[int] = DistributedSampler(
        val_dataset,
        num_replicas=world,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    if len(train_sampler) != 676_975 or len(val_sampler) != 3_402:
        raise RuntimeError("distributed split cardinality drifted or would require padding")
    train_epoch = SharedEpoch(0)
    val_epoch = SharedEpoch(0)
    worker_count = config.workers_per_rank if workers is None else int(workers)
    effective_batch = config.batch_size_per_rank if batch_size is None else int(batch_size)
    common: dict[str, Any] = {
        "batch_size": effective_batch,
        "num_workers": worker_count,
        "persistent_workers": worker_count > 0,
        "prefetch_factor": 2 if worker_count > 0 else None,
        "pin_memory": True,
        "drop_last": False,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        collate_fn=DualCorruptionCollator(
            observed_genus_ids=observed,
            seed=config.seed,
            stream=1,
            shared_epoch=train_epoch,
        ),
        **common,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        collate_fn=DualCorruptionCollator(
            observed_genus_ids=observed,
            seed=config.seed,
            stream=2,
            shared_epoch=val_epoch,
        ),
        **common,
    )
    return _LoaderBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        train_epoch=train_epoch,
        val_epoch=val_epoch,
        train_rows=train_rows,
        val_rows=val_rows,
    )


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _averaged_encoder_gradients(
    loss: Tensor,
    parameters: tuple[nn.Parameter, ...],
    world: int,
) -> tuple[Tensor, ...]:
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    averaged: list[Tensor] = []
    for gradient in gradients:
        value = gradient.detach().float()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(world)
        averaged.append(value)
    return tuple(averaged)


def _gradient_norm(gradients: Iterable[Tensor]) -> float:
    values = tuple(gradients)
    if not values:
        raise ValueError("gradient collection is empty")
    total = torch.zeros((), device=values[0].device, dtype=torch.float64)
    for gradient in values:
        total += gradient.double().square().sum()
    return float(torch.sqrt(total).cpu())


def _gradient_cosine(left: Iterable[Tensor], right: Iterable[Tensor]) -> float:
    left_values = tuple(left)
    right_values = tuple(right)
    if not left_values or len(left_values) != len(right_values):
        raise ValueError("gradient collections must be non-empty and aligned")
    dot = torch.zeros((), device=left_values[0].device, dtype=torch.float64)
    left_sq = torch.zeros_like(dot)
    right_sq = torch.zeros_like(dot)
    for first, second in zip(left_values, right_values, strict=True):
        dot += (first.double() * second.double()).sum()
        left_sq += first.double().square().sum()
        right_sq += second.double().square().sum()
    denominator = torch.sqrt(left_sq * right_sq).clamp_min(1e-30)
    return float((dot / denominator).cpu())


def calibrate_loss_weights(config: MPCRunConfig) -> Path:
    config = config.normalized()
    sealed = validate_sealed_run(config)
    rank, world, _, device = _init_distributed()
    try:
        _seed_all(config.seed)
        candidate, observed = _load_prior(config)
        model = MPCPretrainingModel(candidate).to(device)
        initial_sha = _model_state_sha256(model)
        # Keep identical parameters but give each rank an independent dropout stream.
        _seed_all(config.seed + 10_000 * rank)
        loaders = _build_loaders(config, rank=rank, world=world, observed=observed)
        loaders.train_sampler.set_epoch(0)
        loaders.train_epoch.set(0)
        parameters = model.shared_encoder_parameters()
        records: list[dict[str, float | int]] = []
        iterator = iter(loaders.train_loader)
        torch.cuda.reset_peak_memory_stats(device)
        for batch_index in range(config.calibration_batches):
            batch = batch_to_device(next(iterator), device)
            component_gradients: dict[str, tuple[Tensor, ...]] = {}
            component_losses: dict[str, float] = {}
            for component in ("mlm", "continuous", "presence"):
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model.component_loss(component, **batch)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"non-finite {component} calibration loss")
                reduced_loss = loss.detach().float().clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss.div_(world)
                component_losses[component] = float(reduced_loss.cpu())
                component_gradients[component] = _averaged_encoder_gradients(
                    loss, parameters, world
                )
            norms = {
                name: _gradient_norm(gradients)
                for name, gradients in component_gradients.items()
            }
            records.append({
                "batch_index": batch_index,
                **{f"{name}_loss": value for name, value in component_losses.items()},
                **{f"{name}_encoder_grad_norm": value for name, value in norms.items()},
                "continuous_cosine_with_mlm": _gradient_cosine(
                    component_gradients["continuous"], component_gradients["mlm"]
                ),
                "presence_cosine_with_mlm": _gradient_cosine(
                    component_gradients["presence"], component_gradients["mlm"]
                ),
            })
            del component_gradients, batch
        medians = {
            name: float(np.median([
                float(record[f"{name}_encoder_grad_norm"]) for record in records
            ]))
            for name in ("mlm", "continuous", "presence")
        }
        if not all(math.isfinite(value) and value > 0 for value in medians.values()):
            raise RuntimeError("invalid median encoder gradients during calibration")
        lambda_c = 0.30 * medians["mlm"] / medians["continuous"]
        lambda_p = 0.10 * medians["mlm"] / medians["presence"]
        if not all(math.isfinite(value) and value > 0 for value in (lambda_c, lambda_p)):
            raise RuntimeError("calibrated loss coefficient is invalid")
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "CALIBRATED",
            "manifest_sha256": sealed["manifest_sha256"],
            "seed": config.seed,
            "world_size": world,
            "batch_size_per_rank": config.batch_size_per_rank,
            "global_sample_batch": config.batch_size_per_rank * world,
            "batches": config.calibration_batches,
            "target_encoder_gradient_ratio": {
                "mlm": 1.0,
                "continuous": 0.30,
                "presence": 0.10,
            },
            "median_unscaled_encoder_grad_norm": medians,
            "lambda": {"mlm": 1.0, "continuous": lambda_c, "presence": lambda_p},
            "records": records,
            "initial_model_state_sha256": initial_sha,
            "peak_reserved_gib_max_local": torch.cuda.max_memory_reserved(device) / 2**30,
        }
        peak = torch.tensor(payload["peak_reserved_gib_max_local"], device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        payload["peak_reserved_gib_max"] = float(peak.cpu())
        if payload["peak_reserved_gib_max"] >= 36.0:
            raise RuntimeError("calibration exceeded the frozen 36 GiB memory gate")
        if rank == 0:
            epoch0_path = config.output_root / "run/epoch0.pt"
            _atomic_torch(epoch0_path, {
                "schema_version": SCHEMA_VERSION,
                "checkpoint_kind": "mpc_epoch0",
                "manifest_sha256": sealed["manifest_sha256"],
                "model_config": model.config.to_dict(),
                "model_state_dict": model.state_dict(),
                "model_state_sha256": initial_sha,
            })
            payload["epoch0"] = {
                "path": str(epoch0_path),
                "sha256": sha256_file(epoch0_path),
            }
            payload["calibration_sha256"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            path = config.output_root / "audit/calibration.json"
            _atomic_json(path, payload)
            print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return config.output_root / "audit/calibration.json"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _finite_gradients(module: nn.Module) -> bool:
    return all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in module.parameters()
    )


def _maximum_replica_difference(
    named_values: Iterable[tuple[str, Tensor]],
) -> tuple[float, str]:
    """Return the exact cross-rank maximum without flattening the whole model."""
    maximum = -1.0
    maximum_name = ""
    for name, value in named_values:
        reference = value.detach().clone()
        dist.broadcast(reference, src=0)
        difference = (value.detach() - reference).abs().max()
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        current = float(difference.cpu())
        if current > maximum:
            maximum = current
            maximum_name = name
    if maximum < 0:
        raise RuntimeError("replica comparison received no tensors")
    return maximum, maximum_name


def _assert_parameter_sync(
    module: nn.Module,
    world: int,
    *,
    tolerance: float = 1e-6,
) -> float:
    del world  # The initialized process group is the source of truth.
    maximum, name = _maximum_replica_difference(
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    )
    if not math.isfinite(maximum) or maximum > tolerance:
        raise RuntimeError(
            "DDP parameter replicas diverged: "
            f"max_abs={maximum}, parameter={name}, tolerance={tolerance}"
        )
    return maximum


def _assert_gradient_sync(
    module: nn.Module,
    world: int,
    *,
    tolerance: float = 1e-6,
) -> float:
    trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    missing = sum(parameter.grad is None for _, parameter in trainable)
    missing_tensor = torch.tensor(
        [missing], device=next(module.parameters()).device, dtype=torch.int64
    )
    gathered_missing = [torch.empty_like(missing_tensor) for _ in range(world)]
    dist.all_gather(gathered_missing, missing_tensor)
    missing_by_rank = [int(value.item()) for value in gathered_missing]
    if any(missing_by_rank):
        raise RuntimeError(
            f"DDP trainable gradients are missing by rank: {missing_by_rank}"
        )
    maximum, name = _maximum_replica_difference(
        (name, parameter.grad)  # type: ignore[arg-type]
        for name, parameter in trainable
    )
    if not math.isfinite(maximum) or maximum > tolerance:
        raise RuntimeError(
            "DDP gradients were not reduced across ranks: "
            f"max_abs={maximum}, parameter={name}, tolerance={tolerance}"
        )
    return maximum


def _load_calibration(config: MPCRunConfig, sealed: Mapping[str, Any]) -> dict[str, Any]:
    path = config.output_root / "audit/calibration.json"
    if not path.is_file():
        raise RuntimeError("run three-rank B128 calibration before smoke or training")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "CALIBRATED"
        or payload.get("manifest_sha256") != sealed["manifest_sha256"]
        or payload.get("world_size") != 3
        or payload.get("batch_size_per_rank") != 128
        or payload.get("batches") != 12
    ):
        raise RuntimeError("calibration contract does not match the sealed run")
    lambdas = payload.get("lambda", {})
    if lambdas.get("mlm") != 1.0 or not all(
        math.isfinite(float(lambdas.get(name, float("nan"))))
        and float(lambdas[name]) > 0
        for name in ("continuous", "presence")
    ):
        raise RuntimeError("calibration contains invalid lambdas")
    epoch0 = payload.get("epoch0", {})
    path0 = Path(epoch0.get("path", ""))
    if not path0.is_file() or sha256_file(path0) != epoch0.get("sha256"):
        raise RuntimeError("epoch0 artifact is missing or changed")
    expected_calibration_hash = payload.pop("calibration_sha256", None)
    actual_calibration_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload["calibration_sha256"] = expected_calibration_hash
    if actual_calibration_hash != expected_calibration_hash:
        raise RuntimeError("calibration payload hash cannot be reproduced")
    return payload


def _load_epoch0_model(
    config: MPCRunConfig,
    sealed: Mapping[str, Any],
    calibration: Mapping[str, Any],
    device: torch.device,
) -> MPCPretrainingModel:
    candidate, _ = _load_prior(config)
    model = MPCPretrainingModel(candidate).to(device)
    epoch0_path = Path(calibration["epoch0"]["path"])
    payload = torch.load(epoch0_path, map_location=device, weights_only=False)
    if (
        payload.get("checkpoint_kind") != "mpc_epoch0"
        or payload.get("manifest_sha256") != sealed["manifest_sha256"]
        or payload.get("model_config") != model.config.to_dict()
    ):
        raise RuntimeError("epoch0 checkpoint contract mismatch")
    incompatible = model.load_state_dict(payload["model_state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("epoch0 model failed strict restore")
    if _model_state_sha256(model) != payload["model_state_sha256"]:
        raise RuntimeError("epoch0 model state hash changed after strict restore")
    return model


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
    base_lr: float,
    minimum_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(round(total_steps * warmup_ratio))
    if warmup_steps <= 0 or warmup_steps >= total_steps:
        raise ValueError("invalid warmup duration")
    minimum_factor = minimum_lr / base_lr

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_factor + (1.0 - minimum_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _loss_statistics_tensor(
    output: Any,
    lambdas: Mapping[str, float],
    *,
    device: torch.device,
) -> Tensor:
    """Return sums/counts for exact cross-rank epoch aggregation."""
    total = (
        output.mlm_loss
        + float(lambdas["continuous"]) * output.continuous_loss
        + float(lambdas["presence"]) * output.presence_loss
    )
    return torch.tensor([
        float(output.mlm_loss.detach()) * output.mlm_count,
        float(output.mlm_count),
        float(output.continuous_loss.detach()) * output.continuous_count,
        float(output.continuous_count),
        float(output.presence_loss.detach()) * output.presence_count,
        float(output.presence_count),
        float(total.detach()),
        1.0,
    ], device=device, dtype=torch.float64)


def _unpack_loss_statistics(values: Tensor, lambdas: Mapping[str, float]) -> dict[str, float]:
    array = values.detach().cpu().numpy()
    mlm = float(array[0] / array[1])
    continuous = float(array[2] / array[3])
    presence = float(array[4] / array[5])
    return {
        "mlm": mlm,
        "continuous": continuous,
        "presence": presence,
        "weighted_total": (
            mlm
            + float(lambdas["continuous"]) * continuous
            + float(lambdas["presence"]) * presence
        ),
        "mean_batch_weighted_total": float(array[6] / array[7]),
    }


def _spectral_summary(count: int, sum_h: Tensor, sum_outer: Tensor) -> dict[str, Any]:
    if count < 2:
        raise ValueError("spectral summary requires at least two samples")
    mean = sum_h / count
    covariance = sum_outer / count - torch.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = torch.linalg.eigvalsh(covariance.cpu()).double().clamp_min(0.0)
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0:
        raise RuntimeError("validation representation covariance collapsed")
    probabilities = eigenvalues / total
    positive = probabilities > 0
    entropy_rank = float(torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum()))
    participation_ratio = float(1.0 / probabilities.square().sum())
    maximum = float(eigenvalues[-1])
    stable_rank = total / maximum
    descending = torch.flip(eigenvalues, dims=(0,))
    cumulative = torch.cumsum(descending, dim=0) / total

    def k_at(fraction: float) -> int:
        return int(torch.searchsorted(cumulative, torch.tensor(fraction, dtype=cumulative.dtype)).item() + 1)

    return {
        "n": count,
        "dimension": int(eigenvalues.numel()),
        "total_variance": total,
        "maximum_eigenvalue": maximum,
        "effective_rank_covariance_entropy": entropy_rank,
        "participation_ratio": participation_ratio,
        "stable_rank_representation": stable_rank,
        "k50": k_at(0.50),
        "k90": k_at(0.90),
        "k95": k_at(0.95),
        "k99": k_at(0.99),
    }


@torch.inference_mode()
def _validate_epoch(
    ddp: DDP,
    loader: DataLoader[dict[str, np.ndarray]],
    *,
    expected_rows: np.ndarray,
    lambdas: Mapping[str, float],
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any]:
    ddp.eval()
    statistics = torch.zeros(8, device=device, dtype=torch.float64)
    representation_count = torch.zeros((), device=device, dtype=torch.float64)
    sum_h = torch.zeros(512, device=device, dtype=torch.float64)
    sum_outer = torch.zeros((512, 512), device=device, dtype=torch.float64)
    local_rows: list[np.ndarray] = []
    for batch in loader:
        local_rows.append(np.asarray(batch["row_ids"], dtype=np.int64))
        arguments = batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = ddp(**arguments)
            _, mean = ddp.module.unmasked_representations(
                arguments["genus_ids"], arguments["rclr"], arguments["padding_mask"]
            )
        statistics += _loss_statistics_tensor(output, lambdas, device=device)
        mean64 = mean.float().double()
        representation_count += mean64.shape[0]
        sum_h += mean64.sum(0)
        sum_outer += mean64.T @ mean64
    for tensor in (statistics, representation_count, sum_h, sum_outer):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    rows_array = np.concatenate(local_rows)
    gathered_rows: list[np.ndarray] | None = [None] * world if rank == 0 else None  # type: ignore[list-item]
    dist.gather_object(rows_array, gathered_rows, dst=0)
    if rank == 0:
        assert gathered_rows is not None
        combined = np.concatenate(gathered_rows)
        if combined.size != expected_rows.size or not np.array_equal(
            np.sort(combined), np.sort(np.asarray(expected_rows, dtype=np.int64))
        ):
            raise RuntimeError("distributed validation rows contain padding, duplication, or omission")
        payload = {
            "loss": _unpack_loss_statistics(statistics, lambdas),
            "representation": _spectral_summary(
                int(representation_count.item()), sum_h.cpu(), sum_outer.cpu()
            ),
            "row_count": int(combined.size),
            "row_unique_count": int(np.unique(combined).size),
        }
    else:
        payload = None
    objects = [payload]
    dist.broadcast_object_list(objects, src=0)
    assert isinstance(objects[0], dict)
    return objects[0]


def _capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def _restore_rng_state(state: Mapping[str, Any], device: torch.device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"], device)


def _checkpoint_payload(
    *,
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_score: float,
    meaningful_anchor: float,
    stale_epochs: int,
    lambdas: Mapping[str, float],
    sealed: Mapping[str, Any],
    rng_by_rank: list[dict[str, Any]],
    validation: Mapping[str, Any],
    config: MPCRunConfig,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_kind": "mpc_full_resume",
        "manifest_sha256": sealed["manifest_sha256"],
        "config": config.to_dict(),
        "model_config": ddp.module.config.to_dict(),
        "model_state_dict": ddp.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_score": best_score,
        "meaningful_anchor": meaningful_anchor,
        "stale_epochs": stale_epochs,
        "lambda": dict(lambdas),
        "rng_by_rank": rng_by_rank,
        "validation": dict(validation),
    }


def _save_training_checkpoint(
    *,
    config: MPCRunConfig,
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    global_step: int,
    best_score: float,
    meaningful_anchor: float,
    stale_epochs: int,
    lambdas: Mapping[str, float],
    sealed: Mapping[str, Any],
    validation: Mapping[str, Any],
    is_absolute_best: bool,
    rank: int,
    world: int,
    device: torch.device,
) -> dict[str, Any] | None:
    local_rng = _capture_rng_state(device)
    gathered_rng: list[dict[str, Any]] | None = [None] * world if rank == 0 else None  # type: ignore[list-item]
    dist.gather_object(local_rng, gathered_rng, dst=0)
    if rank != 0:
        dist.barrier()
        return None
    assert gathered_rng is not None
    payload = _checkpoint_payload(
        ddp=ddp,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        global_step=global_step,
        best_score=best_score,
        meaningful_anchor=meaningful_anchor,
        stale_epochs=stale_epochs,
        lambdas=lambdas,
        sealed=sealed,
        rng_by_rank=gathered_rng,
        validation=validation,
        config=config,
    )
    last_path = config.output_root / "run/checkpoints/last.pt"
    _atomic_torch(last_path, payload)
    result = {"last": str(last_path), "last_sha256": sha256_file(last_path)}
    if is_absolute_best:
        best_path = config.output_root / "run/checkpoints/best.pt"
        _atomic_torch(best_path, payload)
        result.update({"best": str(best_path), "best_sha256": sha256_file(best_path)})
    if epoch == 0 or (epoch + 1) % 5 == 0:
        milestone_path = config.output_root / f"run/milestones/epoch_{epoch + 1:03d}.pt"
        _atomic_torch(milestone_path, {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_kind": "mpc_model_milestone",
            "manifest_sha256": sealed["manifest_sha256"],
            "epoch": epoch,
            "global_step": global_step,
            "lambda": dict(lambdas),
            "model_config": ddp.module.config.to_dict(),
            "model_state_dict": ddp.module.state_dict(),
            "validation": dict(validation),
        })
        result.update({
            "milestone": str(milestone_path),
            "milestone_sha256": sha256_file(milestone_path),
        })
    _atomic_json(config.output_root / "results/checkpoint_index_latest.json", result)
    dist.barrier()
    return result


def run_ddp_smoke(config: MPCRunConfig) -> Path:
    """Exercise updates, DDP sync and strict optimizer/scheduler restore at B8."""
    config = config.normalized()
    sealed = validate_sealed_run(config)
    calibration = _load_calibration(config, sealed)
    rank, world, _, device = _init_distributed()
    try:
        _seed_all(config.seed)
        _, observed = _load_prior(config)
        model = _load_epoch0_model(config, sealed, calibration, device)
        _seed_all(config.seed + 10_000 * rank)
        loaders = _build_loaders(
            config, rank=rank, world=world, observed=observed, batch_size=8, workers=0
        )
        loaders.train_sampler.set_epoch(0)
        loaders.train_epoch.set(0)
        ddp = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        optimizer = torch.optim.AdamW(
            ddp.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        steps_per_epoch = math.ceil(676_975 / 8)
        scheduler = _make_scheduler(
            optimizer,
            total_steps=config.max_epochs * steps_per_epoch,
            warmup_ratio=config.warmup_ratio,
            base_lr=config.learning_rate,
            minimum_lr=config.minimum_lr,
        )
        lambdas = calibration["lambda"]
        iterator = iter(loaders.train_loader)
        losses: list[float] = []
        gradient_sync_max_abs = 0.0
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(2):
            arguments = batch_to_device(next(iterator), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = ddp(**arguments)
                total = (
                    output.mlm_loss
                    + float(lambdas["continuous"]) * output.continuous_loss
                    + float(lambdas["presence"]) * output.presence_loss
                )
            if not bool(torch.isfinite(total)):
                raise RuntimeError("DDP smoke produced a non-finite loss")
            total.backward()
            if not _finite_gradients(ddp):
                raise RuntimeError("DDP smoke produced a non-finite gradient")
            gradient_sync_max_abs = max(
                gradient_sync_max_abs, _assert_gradient_sync(ddp, world)
            )
            torch.nn.utils.clip_grad_norm_(ddp.parameters(), config.gradient_clip)
            optimizer.step()
            scheduler.step()
            losses.append(float(total.detach().cpu()))
        sync_max_abs = _assert_parameter_sync(ddp, world)
        peak = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        local_rng = _capture_rng_state(device)
        gathered_rng: list[dict[str, Any]] | None = [None] * world if rank == 0 else None  # type: ignore[list-item]
        dist.gather_object(local_rng, gathered_rng, dst=0)
        smoke_checkpoint = config.output_root / "audit/smoke_resume.pt"
        if rank == 0:
            assert gathered_rng is not None
            saved_model_sha256 = _model_state_sha256(ddp.module)
            _atomic_torch(smoke_checkpoint, {
                "model": ddp.module.state_dict(),
                "model_state_sha256": saved_model_sha256,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": gathered_rng,
                "manifest_sha256": sealed["manifest_sha256"],
            })
        dist.barrier()
        restored = _load_epoch0_model(config, sealed, calibration, device)
        restored_ddp = DDP(
            restored,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        restored_optimizer = torch.optim.AdamW(
            restored_ddp.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        restored_scheduler = _make_scheduler(
            restored_optimizer,
            total_steps=config.max_epochs * steps_per_epoch,
            warmup_ratio=config.warmup_ratio,
            base_lr=config.learning_rate,
            minimum_lr=config.minimum_lr,
        )
        checkpoint = torch.load(smoke_checkpoint, map_location=device, weights_only=False)
        restored_ddp.module.load_state_dict(checkpoint["model"], strict=True)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        restored_scheduler.load_state_dict(checkpoint["scheduler"])
        if _model_state_sha256(restored_ddp.module) != checkpoint["model_state_sha256"]:
            raise RuntimeError("smoke strict restore differs from the saved rank0 state")
        if restored_scheduler.state_dict() != checkpoint["scheduler"]:
            raise RuntimeError("smoke strict restore changed scheduler state")
        if len(restored_optimizer.state) != len(checkpoint["optimizer"]["state"]):
            raise RuntimeError("smoke strict restore changed optimizer state cardinality")
        restored_sync_max_abs = _assert_parameter_sync(restored_ddp, world)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "manifest_sha256": sealed["manifest_sha256"],
            "calibration_sha256": calibration["calibration_sha256"],
            "world_size": world,
            "batch_size_per_rank": 8,
            "optimizer_steps": 2,
            "losses": losses,
            "finite": True,
            "gradient_sync_max_abs": gradient_sync_max_abs,
            "parameter_sync_max_abs": sync_max_abs,
            "restored_parameter_sync_max_abs": restored_sync_max_abs,
            "saved_model_state_sha256": checkpoint["model_state_sha256"],
            "strict_model_optimizer_scheduler_restore": True,
            "peak_reserved_gib": float(peak.cpu()),
            "all_a100": True,
        }
        if rank == 0:
            path = config.output_root / "audit/ddp_smoke.json"
            _atomic_json(path, result)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return config.output_root / "audit/ddp_smoke.json"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _validate_smoke(config: MPCRunConfig, sealed: Mapping[str, Any], calibration: Mapping[str, Any]) -> None:
    path = config.output_root / "audit/ddp_smoke.json"
    if not path.is_file():
        raise RuntimeError("run the three-rank DDP smoke before formal training")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("manifest_sha256") != sealed["manifest_sha256"]
        or payload.get("calibration_sha256") != calibration["calibration_sha256"]
        or payload.get("world_size") != 3
        or payload.get("optimizer_steps") != 2
        or not payload.get("strict_model_optimizer_scheduler_restore")
        or not math.isfinite(float(payload.get("gradient_sync_max_abs", float("nan"))))
        or float(payload["gradient_sync_max_abs"]) > 1e-6
        or not math.isfinite(float(payload.get("parameter_sync_max_abs", float("nan"))))
        or float(payload["parameter_sync_max_abs"]) > 1e-6
        or not math.isfinite(float(payload.get("restored_parameter_sync_max_abs", float("nan"))))
        or float(payload["restored_parameter_sync_max_abs"]) > 1e-6
        or len(str(payload.get("saved_model_state_sha256", ""))) != 64
    ):
        raise RuntimeError("DDP smoke does not satisfy the formal-run gate")


def _load_resume(
    path: Path,
    *,
    config: MPCRunConfig,
    sealed: Mapping[str, Any],
    lambdas: Mapping[str, float],
    ddp: DDP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    rank: int,
    device: torch.device,
) -> tuple[int, int, float, float, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("checkpoint_kind") != "mpc_full_resume"
        or payload.get("manifest_sha256") != sealed["manifest_sha256"]
        or payload.get("config") != config.to_dict()
        or payload.get("model_config") != ddp.module.config.to_dict()
        or payload.get("lambda") != dict(lambdas)
        or len(payload.get("rng_by_rank", [])) != 3
    ):
        raise RuntimeError("resume checkpoint contract mismatch")
    ddp.module.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    _restore_rng_state(payload["rng_by_rank"][rank], device)
    return (
        int(payload["epoch"]) + 1,
        int(payload["global_step"]),
        float(payload["best_score"]),
        float(payload["meaningful_anchor"]),
        int(payload["stale_epochs"]),
    )


def run_full_pretraining(config: MPCRunConfig, *, resume: Path | None = None) -> Path:
    config = config.normalized()
    sealed = validate_sealed_run(config)
    calibration = _load_calibration(config, sealed)
    _validate_smoke(config, sealed, calibration)
    rank, world, _, device = _init_distributed()
    try:
        _seed_all(config.seed)
        _, observed = _load_prior(config)
        model = _load_epoch0_model(config, sealed, calibration, device)
        _seed_all(config.seed + 10_000 * rank)
        loaders = _build_loaders(config, rank=rank, world=world, observed=observed)
        if len(loaders.train_loader) != 5_289:
            raise RuntimeError("full-data updates per epoch drifted from 5,289")
        ddp = DDP(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        optimizer = torch.optim.AdamW(
            ddp.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        total_steps = config.max_epochs * len(loaders.train_loader)
        scheduler = _make_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_ratio=config.warmup_ratio,
            base_lr=config.learning_rate,
            minimum_lr=config.minimum_lr,
        )
        lambdas = {
            name: float(value) for name, value in calibration["lambda"].items()
        }
        start_epoch = 0
        global_step = 0
        best_score = math.inf
        meaningful_anchor = math.inf
        stale_epochs = 0
        if resume is not None:
            start_epoch, global_step, best_score, meaningful_anchor, stale_epochs = _load_resume(
                resume.resolve(),
                config=config,
                sealed=sealed,
                lambdas=lambdas,
                ddp=ddp,
                optimizer=optimizer,
                scheduler=scheduler,
                rank=rank,
                device=device,
            )
        _assert_parameter_sync(ddp, world)
        if rank == 0:
            _atomic_json(config.output_root / "audit/formal_launch.json", {
                "schema_version": SCHEMA_VERSION,
                "status": "RUNNING",
                "manifest_sha256": sealed["manifest_sha256"],
                "calibration_sha256": calibration["calibration_sha256"],
                "lambda": lambdas,
                "start_epoch": start_epoch,
                "start_global_step": global_step,
                "resume": str(resume.resolve()) if resume is not None else None,
                "world_size": world,
                "batch_size_per_rank": config.batch_size_per_rank,
                "global_sample_batch": config.batch_size_per_rank * world,
                "encoder_sample_views_per_update": config.batch_size_per_rank * world * 2,
                "steps_per_epoch": len(loaders.train_loader),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            })
        progress_path = config.output_root / "logs/progress.jsonl"
        epoch_path = config.output_root / "results/epochs.jsonl"
        final_reason = "MAX_EPOCHS"
        for epoch in range(start_epoch, config.max_epochs):
            ddp.train()
            loaders.train_sampler.set_epoch(epoch)
            loaders.train_epoch.set(epoch)
            epoch_started = perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            train_statistics = torch.zeros(8, device=device, dtype=torch.float64)
            local_samples = 0
            first_step_gradient_sync_max_abs = math.nan
            first_step_parameter_sync_max_abs = math.nan
            for batch_index, batch in enumerate(loaders.train_loader):
                if int(batch["corruption_epoch"]) != epoch:
                    raise RuntimeError("persistent training worker did not receive the new epoch")
                arguments = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = ddp(**arguments)
                    total = (
                        output.mlm_loss
                        + lambdas["continuous"] * output.continuous_loss
                        + lambdas["presence"] * output.presence_loss
                    )
                if not bool(torch.isfinite(total)):
                    raise RuntimeError(f"non-finite total loss at epoch={epoch}, batch={batch_index}")
                total.backward()
                if not _finite_gradients(ddp):
                    raise RuntimeError(f"non-finite gradient at epoch={epoch}, batch={batch_index}")
                if batch_index == 0:
                    first_step_gradient_sync_max_abs = _assert_gradient_sync(ddp, world)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    ddp.parameters(), config.gradient_clip
                )
                if not bool(torch.isfinite(gradient_norm)):
                    raise RuntimeError("non-finite pre-clip global gradient norm")
                optimizer.step()
                scheduler.step()
                if batch_index == 0:
                    first_step_parameter_sync_max_abs = _assert_parameter_sync(ddp, world)
                train_statistics += _loss_statistics_tensor(output, lambdas, device=device)
                batch_samples = int(arguments["genus_ids"].shape[0])
                local_samples += batch_samples
                global_step += 1
                if rank == 0 and (batch_index == 0 or (batch_index + 1) % 100 == 0):
                    _append_jsonl(progress_path, {
                        "epoch": epoch + 1,
                        "batch": batch_index + 1,
                        "batches_per_epoch": len(loaders.train_loader),
                        "global_step": global_step,
                        "weighted_total_local": float(total.detach().cpu()),
                        "mlm_local": float(output.mlm_loss.detach().cpu()),
                        "continuous_local": float(output.continuous_loss.detach().cpu()),
                        "presence_local": float(output.presence_loss.detach().cpu()),
                        "gradient_norm_preclip_local": float(gradient_norm.detach().cpu()),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": perf_counter() - epoch_started,
                    })
            sample_tensor = torch.tensor(local_samples, device=device, dtype=torch.int64)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
            if int(sample_tensor.item()) != TRAIN_ROWS:
                raise RuntimeError("training epoch omitted, duplicated, or padded rows")
            dist.all_reduce(train_statistics, op=dist.ReduceOp.SUM)
            training_loss = _unpack_loss_statistics(train_statistics, lambdas)
            loaders.val_sampler.set_epoch(0)
            loaders.val_epoch.set(0)
            validation = _validate_epoch(
                ddp,
                loaders.val_loader,
                expected_rows=loaders.val_rows,
                lambdas=lambdas,
                rank=rank,
                world=world,
                device=device,
            )
            sync_max_abs = _assert_parameter_sync(ddp, world)
            peak = torch.tensor(torch.cuda.max_memory_reserved(device) / 2**30, device=device)
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            peak_reserved = float(peak.cpu())
            if peak_reserved >= 36.0:
                raise RuntimeError("formal epoch exceeded the 36 GiB memory gate")
            score = float(validation["loss"]["weighted_total"])
            is_absolute_best = score < best_score
            if is_absolute_best:
                best_score = score
            if not math.isfinite(meaningful_anchor) or score <= meaningful_anchor * (
                1.0 - config.relative_min_improvement
            ):
                meaningful_anchor = score
                stale_epochs = 0
                meaningful_improvement = True
            else:
                stale_epochs += 1
                meaningful_improvement = False
            epoch_seconds = perf_counter() - epoch_started
            epoch_record = {
                "schema_version": SCHEMA_VERSION,
                "epoch": epoch + 1,
                "global_step": global_step,
                "training_loss": training_loss,
                "validation": validation,
                "lambda": lambdas,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": epoch_seconds,
                "global_samples_per_second": TRAIN_ROWS / epoch_seconds,
                "peak_reserved_gib": peak_reserved,
                "first_step_gradient_sync_max_abs": first_step_gradient_sync_max_abs,
                "first_step_parameter_sync_max_abs": first_step_parameter_sync_max_abs,
                "parameter_sync_max_abs": sync_max_abs,
                "absolute_best": is_absolute_best,
                "best_score": best_score,
                "meaningful_improvement": meaningful_improvement,
                "meaningful_anchor": meaningful_anchor,
                "stale_epochs": stale_epochs,
            }
            if rank == 0:
                _append_jsonl(epoch_path, epoch_record)
                print(json.dumps(epoch_record, sort_keys=True), flush=True)
            checkpoint_record = _save_training_checkpoint(
                config=config,
                ddp=ddp,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_score=best_score,
                meaningful_anchor=meaningful_anchor,
                stale_epochs=stale_epochs,
                lambdas=lambdas,
                sealed=sealed,
                validation=validation,
                is_absolute_best=is_absolute_best,
                rank=rank,
                world=world,
                device=device,
            )
            if rank == 0:
                _append_jsonl(config.output_root / "results/checkpoints.jsonl", {
                    "epoch": epoch + 1,
                    **(checkpoint_record or {}),
                })
            if epoch + 1 >= config.min_epochs and stale_epochs >= config.early_stopping_patience:
                final_reason = "LABEL_FREE_SATURATION"
                break
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "TRAINING_COMPLETE",
            "reason": final_reason,
            "last_completed_epoch": epoch + 1,
            "global_step": global_step,
            "best_score": best_score,
            "meaningful_anchor": meaningful_anchor,
            "stale_epochs": stale_epochs,
        }
        if rank == 0:
            path = config.output_root / "results/training_complete.json"
            _atomic_json(path, result)
            launch_path = config.output_root / "audit/formal_launch.json"
            launch = json.loads(launch_path.read_text(encoding="utf-8"))
            launch["status"] = "TRAINING_COMPLETE"
            launch["result"] = result
            _atomic_json(launch_path, launch)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        dist.barrier()
        return config.output_root / "results/training_complete.json"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
