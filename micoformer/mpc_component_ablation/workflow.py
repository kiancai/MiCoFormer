"""Fail-closed three-rank DDP workflow for full-data MPC pretraining."""
from __future__ import annotations

import hashlib
import datetime
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
from micoformer.mpc_fixed_pretraining.model import MPCFixedModelConfig as MPCModelConfig
from .model import WeakUniFracModel as MPCPretrainingModel
from .teacher import ASSETS,TEACHER_T,STUDENT_T
from .contract import ARMS, BASE_INIT, EXPERIMENT, JOB_ID, weights_for


SCHEMA_VERSION = 1
TRAIN_ROWS = 2_023_580
VAL_ROWS = 10_169
CORPUS_SHAPE = (3_334_541, 8_114)
EXPECTED_CORPUS_SHA256 = "ab41012e7b3a04cbdee21b182abcee7363b8084a59f815e0efecdccc005ae6c3"
EXPECTED_PRIOR_SHA256 = "6a2c08d216efb90499720eec5aa872df5275a62d5e8052760a5105765262262c"
EXPECTED_TRAIN_SHA256 = "53e0df017a9b6644b24e7764fbd607877227c0433b7c5876be061bf7c532c05b"
EXPECTED_VAL_SHA256 = "beebe861783d1cf715265819d67a7af062c0c7593209104ccf64bb9476ed9661"


@dataclass(frozen=True)
class MPCRunConfig:
    corpus: Path
    train_rows: Path
    val_rows: Path
    prior_assets: Path
    output_root: Path
    arm: str = "A0"
    seed: int = 42
    batch_size_per_rank: int = 128
    workers_per_rank: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    gradient_clip: float = 1.0
    max_epochs: int = 15
    min_epochs: int = 15
    warmup_ratio: float = 1 / 15
    minimum_lr: float = 1e-6
    early_stopping_patience: int = 0  # Disabled: complete all 15 epochs.
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
        expected = (42, 128, 2, 3e-4, 1e-2, 1.0, 15, 15, 1 / 15, 1e-6, 0, 0.001, 12)
        if self.arm not in ARMS:
            raise ValueError("unknown ablation arm")
        if fixed != expected:
            raise ValueError("the user-approved first full-data training contract drifted")

    def normalized(self) -> "MPCRunConfig":
        return MPCRunConfig(
            corpus=self.corpus.resolve(),
            train_rows=self.train_rows.resolve(),
            val_rows=self.val_rows.resolve(),
            prior_assets=self.prior_assets.resolve(),
            output_root=self.output_root.resolve(),
            arm=self.arm,
        )

    @property
    def weights(self):
        return weights_for(self.arm)

    @property
    def identity_asset(self):
        return EXPERIMENT / f"assets/{self.arm}.npy"

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
    repo = Path(__file__).resolve().parents[2]
    exp = EXPERIMENT
    return sorted(repo.glob("micoformer/**/*.py")) + sorted((exp / "scripts").glob("*.py"))


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
        "fixed_weights": config.weights,
        "arm": {"id": config.arm, "name": ARMS[config.arm][0],
                "disabled_component": ARMS[config.arm][1]},
        "identity_asset": {"path": str(config.identity_asset), "sha256": sha256_file(config.identity_asset)},
        "identity_asset_manifest": {"path": str(EXPERIMENT / "assets/MANIFEST.json"), "sha256": sha256_file(EXPERIMENT / "assets/MANIFEST.json")},
        "model_config_note": "legacy architecture tag; actual identity is bound by arm and identity_asset",
        "teacher_assets": {str(p):sha256_file(p) for p in ASSETS.iterdir() if p.is_file()},
        "temperatures": {"teacher":TEACHER_T,"student":STUDENT_T},
        "probe_assets": {
            str(path): sha256_file(path)
            for path in sorted((config.output_root / "audit/probe").glob("*"))
        },
        "initialization_asset": {
            "path": str(BASE_INIT),
            "sha256": sha256_file(BASE_INIT),
        },
        "training_reduction": "global target-count mean per component; no duplicated or dropped rows",
        "deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
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
    if f"job_{JOB_ID}/" not in Path("/proc/self/cgroup").read_text():
        raise RuntimeError("execution must stay inside the approved current allocation")
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
        table = torch.from_numpy(np.load(config.identity_asset, allow_pickle=False)).float()
        if table.shape != (8116, 512) or not torch.isfinite(table).all():
            raise RuntimeError("invalid arm identity table")
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
    train_sampler: DistributedSampler[int] = ExactDistributedSampler(
        train_dataset,
        num_replicas=world,
        rank=rank,
        shuffle=True,
        seed=config.seed,
        drop_last=False,
    )
    val_sampler: DistributedSampler[int] = ExactDistributedSampler(
        val_dataset,
        num_replicas=world,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    if len(train_sampler) != len(range(rank, TRAIN_ROWS, world)) or len(val_sampler) != len(range(rank, VAL_ROWS, world)):
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
    path = config.output_root / "audit/loss_weights.json"
    payload = json.loads(path.read_text())
    expected = payload.pop("calibration_sha256")
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["calibration_sha256"] = expected
    if expected != actual or payload["manifest_sha256"] != sealed["manifest_sha256"]:
        raise RuntimeError("loss coefficient contract mismatch")
    if payload["lambda"] != config.weights or payload["mode"] != "MLM_NORMALIZED_SELECTED_FIXED_RECIPE":
        raise RuntimeError("fixed coefficients drifted; no recalibration permitted")
    if sha256_file(payload["epoch0"]["path"]) != payload["epoch0"]["sha256"]:
        raise RuntimeError("bound epoch0 changed")
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
    values=[]
    for n in NAMES:
        values.extend([float(getattr(output,n+'_loss').detach())*getattr(output,n+'_count'),float(getattr(output,n+'_count'))])
    values.extend([sum(float(lambdas[n])*float(getattr(output,n+'_loss').detach()) for n in NAMES),1.])
    return torch.tensor(values,device=device,dtype=torch.float64)


def _unpack_loss_statistics(values,lambdas):
    a=values.detach().cpu().numpy()
    raw={n:float(a[2*i]/a[2*i+1]) for i,n in enumerate(NAMES)}
    weighted={n:float(lambdas[n])*raw[n] for n in NAMES}
    total=sum(weighted.values())
    return dict(**raw,weighted_total=total,mean_batch_weighted_total=float(a[-2]/a[-1]),raw_components=raw,weighted_components=weighted,
                enabled_components={n:bool(lambdas[n]) for n in NAMES},
                raw_loss_note="disabled components are computed as diagnostics only, with zero optimization coefficient",
                weighted_fractions={n:weighted[n]/total if total else None for n in NAMES},
                target_counts={n:int(a[2*i+1]) for i,n in enumerate(NAMES)})


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
    statistics = torch.zeros(10, device=device, dtype=torch.float64)
    local_rows: list[np.ndarray] = []
    for batch in loader:
        local_rows.append(np.asarray(batch["row_ids"], dtype=np.int64))
        arguments = batch_to_device(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = ddp(**arguments)
        statistics += _loss_statistics_tensor(output, lambdas, device=device)
    dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
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
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)


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
    epoch_path = config.output_root / f"run/checkpoints/epoch_{epoch + 1:03d}.pt"
    if epoch_path.exists():
        raise RuntimeError("refusing to overwrite an existing completed epoch checkpoint")
    os.link(last_path, epoch_path)
    result.update(epoch_checkpoint=str(epoch_path), epoch_checkpoint_sha256=result["last_sha256"])
    _atomic_json(config.output_root / "results/checkpoint_index_latest.json", result)
    dist.barrier()
    return result


def run_ddp_smoke(config: MPCRunConfig) -> Path:
    from .support import integrated_smoke
    return integrated_smoke(config)


def _validate_smoke(config: MPCRunConfig, sealed: Mapping[str, Any], calibration: Mapping[str, Any]) -> None:
    path = config.output_root / "audit/ddp_smoke.json"
    if not path.is_file():
        raise RuntimeError("run the three-rank DDP smoke before formal training")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("batch_size_per_rank") != 128
        or payload.get("world_size") != 3
        or payload.get("strict_rng_and_next_update_restore") is not True
        or payload.get("manifest_sha256") != sealed["manifest_sha256"]
        or payload.get("calibration_sha256") != calibration["calibration_sha256"]
        or payload.get("world_size") != 3
        or payload.get("optimizer_steps") != 3
        or not payload.get("strict_rng_and_next_update_restore")
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
        if len(loaders.train_loader) != 5_270:
            raise RuntimeError("full-data updates per epoch drifted from 5,270")
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
        if resume is None:
            run_fixed_gradient_probe(config, model, observed, lambdas, epoch=0, step=0, rank=rank, world=world, device=device)
        progress_path = config.output_root / "logs/progress.jsonl"
        epoch_path = config.output_root / "results/epochs.jsonl"
        final_reason = "MAX_EPOCHS"
        for epoch in range(start_epoch, config.max_epochs):
            # Operational checkpoint boundary only; never report a partial run complete.
            remaining = (datetime.datetime.fromisoformat("2026-10-09T14:45:36+08:00") - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
            pause = torch.tensor(int(remaining < 7200) if rank == 0 else 0, device=device)
            dist.broadcast(pause, 0)
            if pause.item():
                path = config.output_root / "results/resource_wait.json"
                if rank == 0:
                    _atomic_json(path, dict(status="WAITING_RESOURCE", last_completed_epoch=epoch,
                        global_step=global_step, checkpoint=str(config.output_root / "run/checkpoints/last.pt"),
                        reason="less_than_two_hours_before_allocation_end", allocation_release=False))
                dist.barrier()
                return path
            ddp.train()
            loaders.train_sampler.set_epoch(epoch)
            loaders.train_epoch.set(epoch)
            epoch_started = perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            train_statistics = torch.zeros(10, device=device, dtype=torch.float64)
            local_samples = 0
            local_row_ids = []
            first_step_gradient_sync_max_abs = math.nan
            first_step_parameter_sync_max_abs = math.nan
            for batch_index, batch in enumerate(loaders.train_loader):
                if int(batch["corruption_epoch"]) != epoch:
                    raise RuntimeError("persistent training worker did not receive the new epoch")
                local_row_ids.append(np.asarray(batch["row_ids"], dtype=np.int64))
                arguments = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = ddp(**arguments)
                    total = distributed_training_loss(output, lambdas, world, device)
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
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
                if batch_index == 0:
                    first_step_parameter_sync_max_abs = _assert_parameter_sync(ddp, world)
                train_statistics += _loss_statistics_tensor(output, lambdas, device=device)
                batch_samples = int(arguments["genus_ids"].shape[0])
                local_samples += batch_samples
                global_step += 1
                step_statistics = _loss_statistics_tensor(output, lambdas, device=device)
                dist.all_reduce(step_statistics, op=dist.ReduceOp.SUM)
                if rank == 0:
                    record = {"epoch": epoch + 1, "batch": batch_index + 1,
                        "batches_per_epoch": len(loaders.train_loader), "global_step": global_step,
                        "loss": _unpack_loss_statistics(step_statistics, lambdas), "weights": dict(lambdas),
                        "gradient_norm_preclip": float(gradient_norm),
                        "clip_multiplier": min(1.0, config.gradient_clip / (float(gradient_norm) + 1e-6)),
                        "lr_used": lr_used, "lr_next": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_seconds": perf_counter() - epoch_started}
                    _append_jsonl(progress_path, record)
                    if batch_index == 0 or (batch_index + 1) % 100 == 0:
                        print(json.dumps(record), flush=True)
                del output, total, arguments
            sample_tensor = torch.tensor(local_samples, device=device, dtype=torch.int64)
            dist.all_reduce(sample_tensor, op=dist.ReduceOp.SUM)
            if int(sample_tensor.item()) != TRAIN_ROWS:
                raise RuntimeError("training epoch omitted, duplicated, or padded rows")
            verify_epoch_rows(local_row_ids, loaders.train_rows, rank, world)
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
            optimizer.zero_grad(set_to_none=True)
            run_fixed_gradient_probe(config, model, observed, lambdas, epoch=epoch+1, step=global_step, rank=rank, world=world, device=device)
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
            # User contract: validation is observational; complete all 15 epochs.
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

from .support import (NAMES, ExactDistributedSampler, distributed_training_loss, verify_epoch_rows, run_fixed_gradient_probe)
