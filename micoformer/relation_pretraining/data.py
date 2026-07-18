"""Strict V3 relation data and hash-bound schedule/cache consumers.

The training path deliberately avoids ``anndata.read_h5ad``.  Even backed
AnnData materializes the full ``obs`` frame, which is wasteful when three
single-GPU arms share one 50 GiB Slurm allocation.  This module reads only the
CSR row slices and the compact metadata arrays published by the strict split.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import h5py
import lightning as L
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


SCHEMA_VERSION = 1
PHYSICAL_BATCH_SIZE = 32
MAX_TOKENS = 512
PROTEIN_VALID_MASS_MIN = 0.90
TEACHER_COMPARE_TOLERANCE = "max(1e-12, 1e-10 * scale)"
TEACHER_VALIDITY_CONTRACT = {
    "protein_endpoint_valid_mass_min": 0.9,
    "protein_pair_valid": "both endpoints valid",
    "unifrac_endpoint_gate": "none",
}
EXACT_SOLVER_CONTRACT = {
    "protein": (
        "scipy.optimize.linprog(method='highs'; primal_feasibility_tolerance=1e-9; "
        "dual_feasibility_tolerance=1e-9; ipm_optimality_tolerance=1e-10); "
        "exact balanced OT; no approximation"
    ),
    "unifrac": "normalized weighted UniFrac; exact genus-to-edge incidence; no approximation",
}
REQUIRED_CACHE_ASSETS = frozenset(
    {
        "corpus",
        "pretrain_v3_all",
        "u_eval",
        "protein_cost",
        "valid_mask",
        "coverage",
        "teacher_reference",
        "tree_manifest",
        "genus_to_edge",
        "branch_lengths",
        "split_manifest",
        "split_npz",
        "schedule_manifest",
        "schedule_file",
        "runner",
    }
)
SPLIT_PREFIX = {
    "train": "relation_train",
    "val": "relation_val",
    "test": "relation_test",
}


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"required manifest is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"manifest is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"manifest root must be an object: {path}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported schema_version in {path}: {payload.get('schema_version')!r}"
        )
    return payload


def _require_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{label} must be a 64-character SHA256 string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RuntimeError(f"{label} is not hexadecimal") from exc
    return value.lower()


def _resolve_inside(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label} path must be a non-empty string")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise RuntimeError(f"{label} path must be a safe relative path: {relative!r}")
    root_resolved = root.resolve()
    direct = (root_resolved / relative_path).resolve()
    if direct.exists():
        return direct
    # The audited producer records paths relative to the MCFProjet root while
    # the consumer is handed the bundle directory.  Resolve that representation
    # without accepting absolute paths or parent traversal.
    matches = [
        (ancestor / relative_path).resolve()
        for ancestor in root_resolved.parents
        if (ancestor / relative_path).exists()
    ]
    unique = list(dict.fromkeys(matches))
    if len(unique) != 1:
        raise RuntimeError(
            f"{label} path resolves to {len(unique)} existing files: {relative!r}"
        )
    return unique[0]


def _require_int64_vector(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.ndim != 1:
        raise RuntimeError(f"{label} must be int64 [N], got {array.dtype} {array.shape}")
    return array


def _validate_unique_rows(rows: np.ndarray, label: str) -> None:
    if rows.size and np.unique(rows).size != rows.size:
        raise RuntimeError(f"{label} contains duplicate global row ids")


class RelationAnnDataDataset(Dataset[dict[str, Any]]):
    """Sparse, global-row-addressed student dataset.

    ``split_rows`` and its project/site arrays are the only metadata held in
    memory.  The HDF5 handle is opened lazily in each worker and only the
    requested CSR row is read.
    """

    def __init__(
        self,
        h5ad_path: str | os.PathLike[str],
        *,
        split_rows: np.ndarray,
        project_codes: Optional[np.ndarray] = None,
        site_codes: Optional[np.ndarray] = None,
        require_metadata: bool = True,
        max_tokens: int = MAX_TOKENS,
        expected_n_vars: Optional[int] = 8_114,
    ) -> None:
        self.h5ad_path = Path(h5ad_path).resolve()
        self.split_rows = _require_int64_vector(split_rows, "split_rows").copy()
        _validate_unique_rows(self.split_rows, "split_rows")
        if require_metadata and (project_codes is None or site_codes is None):
            raise ValueError("training requires aligned project_codes and site_codes")
        if (project_codes is None) != (site_codes is None):
            raise ValueError("project_codes and site_codes must be supplied together")
        if project_codes is None:
            self.project_codes = np.empty(0, dtype=np.int32)
            self.site_codes = np.empty(0, dtype=np.int16)
        else:
            self.project_codes = np.asarray(project_codes)
            self.site_codes = np.asarray(site_codes)
        self.require_metadata = bool(require_metadata)
        if not self.require_metadata and project_codes is not None:
            raise ValueError("metadata-free extraction may not supply selector metadata")
        if self.require_metadata and self.project_codes.shape != self.split_rows.shape:
            raise ValueError("project_codes must be aligned one-to-one with split_rows")
        if self.require_metadata and self.site_codes.shape != self.split_rows.shape:
            raise ValueError("site_codes must be aligned one-to-one with split_rows")
        if self.require_metadata and (
            self.project_codes.dtype.kind not in "iu" or self.site_codes.dtype.kind not in "iu"
        ):
            raise TypeError("project_codes and site_codes must use integer dtypes")
        if self.require_metadata and np.any(self.project_codes < 0):
            raise ValueError("training split contains a missing Project_ID code")
        if self.require_metadata and np.any(self.site_codes < 0):
            raise ValueError("relation split contains a missing RM_Sample_Site code")
        if max_tokens <= 0 or max_tokens > MAX_TOKENS:
            raise ValueError(f"max_tokens must be in [1, {MAX_TOKENS}]")
        self.max_tokens = int(max_tokens)

        with h5py.File(self.h5ad_path, "r") as handle:
            if "X" not in handle or not isinstance(handle["X"], h5py.Group):
                raise ValueError("relation corpus must store X as a CSR HDF5 group")
            x_group = handle["X"]
            if x_group.attrs.get("encoding-type") != "csr_matrix":
                raise ValueError("relation corpus X must use csr_matrix encoding")
            shape = tuple(int(x) for x in np.asarray(x_group.attrs["shape"]))
            if len(shape) != 2:
                raise ValueError("relation corpus X shape must be two-dimensional")
            self.n_obs, self.n_vars = shape
        if expected_n_vars is not None and self.n_vars != int(expected_n_vars):
            raise ValueError(
                f"corpus n_vars={self.n_vars} does not match expected_n_vars={expected_n_vars}"
            )
        if self.split_rows.size:
            if int(self.split_rows.min()) < 0 or int(self.split_rows.max()) >= self.n_obs:
                raise ValueError(f"split row ids escape [0, {self.n_obs})")
        self._position = (
            {int(row): i for i, row in enumerate(self.split_rows.tolist())}
            if self.require_metadata
            else None
        )
        self._handle: Optional[h5py.File] = None

    def __len__(self) -> int:
        return int(self.split_rows.size)

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.h5ad_path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def _metadata(self, global_row: int) -> tuple[int, int]:
        if self._position is None:
            if global_row < 0 or global_row >= self.n_obs:
                raise IndexError(f"global row {global_row} escapes [0, {self.n_obs})")
            return 0, 0
        try:
            position = self._position[int(global_row)]
        except KeyError as exc:
            raise IndexError(f"global row {global_row} is outside the strict split") from exc
        return int(self.project_codes[position]), int(self.site_codes[position])

    def full_composition(self, global_row: int) -> tuple[np.ndarray, np.ndarray]:
        """Return full raw support for audits/producers; never student-truncated."""

        self._metadata(global_row)  # strict membership check
        x_group = self._h5()["X"]
        bounds = np.asarray(x_group["indptr"][global_row : global_row + 2], dtype=np.int64)
        start, stop = int(bounds[0]), int(bounds[1])
        indices = np.asarray(x_group["indices"][start:stop], dtype=np.int64)
        abundance = np.asarray(x_group["data"][start:stop], dtype=np.float64)
        if indices.size == 0:
            raise RuntimeError(f"global row {global_row} has empty composition")
        if indices.shape != abundance.shape:
            raise RuntimeError("CSR indices/data shape mismatch")
        if np.any(indices < 0) or np.any(indices >= self.n_vars):
            raise RuntimeError(f"global row {global_row} contains an invalid var index")
        if not np.isfinite(abundance).all() or np.any(abundance <= 0):
            raise RuntimeError(f"global row {global_row} contains invalid present abundance")
        return indices, abundance

    def get_by_global_row(self, global_row: int) -> dict[str, Any]:
        project_code, site_code = self._metadata(global_row)
        var_indices, raw_abundance = self.full_composition(global_row)
        # Primary key -raw abundance, deterministic secondary key global V3 var index.
        order = np.lexsort((var_indices, -raw_abundance))
        kept = order[: self.max_tokens]
        kept_indices = var_indices[kept]
        kept_abundance = raw_abundance[kept]
        log_abundance = np.log(kept_abundance, dtype=np.float64)
        rclr = (log_abundance - log_abundance.mean(dtype=np.float64)).astype(
            np.float32,
            copy=False,
        )
        return {
            "global_row_id": np.int64(global_row),
            "project_id": np.int64(project_code),
            "site_id": np.int64(site_code),
            "var_indices": kept_indices.astype(np.int64, copy=False),
            "genus_ids": (kept_indices + 2).astype(np.int64, copy=False),
            "rclr": rclr,
            "raw_richness": np.int64(var_indices.size),
            "student_richness": np.int64(kept_indices.size),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_by_global_row(int(self.split_rows[index]))


@dataclass(frozen=True)
class _ScheduleEntry:
    kind: str
    index: int
    path: Path
    sha256: str
    scheduled_rows_sha256: str
    batch_offsets_sha256: str
    omitted_rows_sha256: str
    n_batches: int
    n_scheduled: int
    n_omitted: int
    last_batch_size: int


def _collate_relation_samples(samples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not samples or len(samples) > PHYSICAL_BATCH_SIZE:
        raise RuntimeError(
            f"relation batch size must be in [1, {PHYSICAL_BATCH_SIZE}]"
        )
    max_length = max(int(sample["genus_ids"].size) for sample in samples)
    genus_ids = np.zeros((len(samples), max_length), dtype=np.int64)
    rclr = np.zeros((len(samples), max_length), dtype=np.float32)
    var_indices = np.full((len(samples), max_length), -1, dtype=np.int64)
    padding_mask = np.ones((len(samples), max_length), dtype=bool)
    for i, sample in enumerate(samples):
        length = int(sample["genus_ids"].size)
        genus_ids[i, :length] = sample["genus_ids"]
        rclr[i, :length] = sample["rclr"]
        var_indices[i, :length] = sample["var_indices"]
        padding_mask[i, :length] = False
    return {
        "genus_ids": genus_ids,
        "rclr": rclr,
        "var_indices": var_indices,
        "padding_mask": padding_mask,
        "row_ids": np.asarray([sample["global_row_id"] for sample in samples], dtype=np.int64),
        "project_ids": np.asarray([sample["project_id"] for sample in samples], dtype=np.int64),
        "site_ids": np.asarray([sample["site_id"] for sample in samples], dtype=np.int64),
        "raw_richness": np.asarray([sample["raw_richness"] for sample in samples], dtype=np.int64),
        "student_richness": np.asarray(
            [sample["student_richness"] for sample in samples], dtype=np.int64
        ),
    }


def _upper_to_symmetric(upper: np.ndarray, batch_size: int) -> np.ndarray:
    if upper.dtype != np.float64:
        raise RuntimeError(
            f"upper-triangle teacher vector must remain float64, got {upper.dtype}"
        )
    if upper.shape != (batch_size * (batch_size - 1) // 2,):
        raise RuntimeError(f"upper-triangle teacher vector has wrong shape: {upper.shape}")
    matrix = np.zeros((batch_size, batch_size), dtype=np.float64)
    i, j = np.triu_indices(batch_size, k=1)
    matrix[i, j] = upper
    matrix[j, i] = matrix[i, j]
    return matrix


def _upper_bool_to_symmetric(upper: np.ndarray, batch_size: int) -> np.ndarray:
    if upper.dtype != np.bool_:
        raise RuntimeError(f"upper-triangle validity vector must be bool, got {upper.dtype}")
    if upper.shape != (batch_size * (batch_size - 1) // 2,):
        raise RuntimeError(f"upper-triangle validity vector has wrong shape: {upper.shape}")
    matrix = np.zeros((batch_size, batch_size), dtype=bool)
    i, j = np.triu_indices(batch_size, k=1)
    matrix[i, j] = upper
    matrix[j, i] = matrix[i, j]
    return matrix


class RelationEpochDataset(Dataset[dict[str, Any]]):
    """One deterministic schedule and its completed exact teacher cache."""

    def __init__(
        self,
        samples: RelationAnnDataDataset,
        *,
        entry: _ScheduleEntry,
        scheduled_rows: np.ndarray,
        batch_offsets: np.ndarray,
        pair_offsets: np.ndarray,
        cache_arrays: Mapping[str, np.ndarray],
        cache_manifest_sha256: str,
        cache_sha256: str,
        protein_valid_mass_min: float,
    ) -> None:
        self.samples = samples
        self.entry = entry
        self.scheduled_rows = scheduled_rows
        self.batch_offsets = batch_offsets
        self.pair_offsets = pair_offsets
        self.protein_distance_upper = cache_arrays["protein_distance_upper"]
        self.unifrac_distance_upper = cache_arrays["unifrac_distance_upper"]
        self.protein_valid_mass = cache_arrays["protein_valid_mass"]
        self.protein_borrowed_mass = cache_arrays["protein_borrowed_mass"]
        self.project_codes = cache_arrays["project_codes"]
        self.site_codes = cache_arrays["site_codes"]
        self.protein_endpoint_valid = cache_arrays["protein_endpoint_valid"]
        self.protein_pair_valid_upper = cache_arrays["protein_pair_valid_upper"]
        self.cache_manifest_sha256 = cache_manifest_sha256
        self.cache_sha256 = cache_sha256
        self.protein_valid_mass_min = float(protein_valid_mass_min)

    def __len__(self) -> int:
        return int(self.batch_offsets.size - 1)

    def __getitem__(self, batch_index: int) -> dict[str, Any]:
        row_start, row_stop = (int(x) for x in self.batch_offsets[batch_index : batch_index + 2])
        pair_start, pair_stop = (int(x) for x in self.pair_offsets[batch_index : batch_index + 2])
        rows = self.scheduled_rows[row_start:row_stop]
        batch = _collate_relation_samples(
            [self.samples.get_by_global_row(int(row)) for row in rows]
        )
        batch_size = int(rows.size)
        protein = _upper_to_symmetric(
            self.protein_distance_upper[pair_start:pair_stop], batch_size
        )
        unifrac = _upper_to_symmetric(
            self.unifrac_distance_upper[pair_start:pair_stop], batch_size
        )
        protein_validity = _upper_bool_to_symmetric(
            self.protein_pair_valid_upper[pair_start:pair_stop], batch_size
        )
        unifrac_validity = np.isfinite(unifrac)
        np.fill_diagonal(unifrac_validity, False)
        expected_projects = self.project_codes[row_start:row_stop].astype(np.int64)
        expected_sites = self.site_codes[row_start:row_stop].astype(np.int64)
        if not np.array_equal(batch["project_ids"], expected_projects):
            raise RuntimeError("cache project_codes are misaligned with scheduled corpus rows")
        if not np.array_equal(batch["site_ids"], expected_sites):
            raise RuntimeError("cache site_codes are misaligned with scheduled corpus rows")
        batch.update(
            {
                "teacher_distances": {
                    "protein": protein,
                    "unifrac": unifrac,
                },
                "teacher_validity": {
                    "protein": protein_validity,
                    "unifrac": unifrac_validity,
                },
                "protein_valid_mass": self.protein_valid_mass[row_start:row_stop].astype(
                    np.float64, copy=False
                ),
                "protein_borrowed_mass": self.protein_borrowed_mass[row_start:row_stop].astype(
                    np.float64, copy=False
                ),
                "protein_endpoint_valid": self.protein_endpoint_valid[row_start:row_stop],
                "schedule_kind": self.entry.kind,
                "schedule_index": np.int64(self.entry.index),
                "schedule_batch_index": np.int64(batch_index),
                "schedule_file_sha256": self.entry.sha256,
                "cache_manifest_sha256": self.cache_manifest_sha256,
                "cache_sha256": self.cache_sha256,
            }
        )
        return batch


class RelationScheduleStore:
    """Validate strict split, schedules and exact caches before consumption."""

    def __init__(
        self,
        schedule_root: str | os.PathLike[str],
        cache_root: str | os.PathLike[str],
        corpus_path: str | os.PathLike[str],
    ) -> None:
        self.schedule_root = Path(schedule_root).resolve()
        self.cache_root = Path(cache_root).resolve()
        self.corpus_path = Path(corpus_path).resolve()
        self.schedule_manifest_path = self.schedule_root / "schedule_manifest.json"
        self.split_manifest_path = self.schedule_root / "split_manifest.json"
        self.split_path = self.schedule_root / "split.npz"
        self.schedule_manifest = _load_json(self.schedule_manifest_path)
        self.split_manifest = _load_json(self.split_manifest_path)
        self.schedule_manifest_sha256 = sha256_file(self.schedule_manifest_path)
        self.split_manifest_sha256 = sha256_file(self.split_manifest_path)
        self.split_sha256 = sha256_file(self.split_path)
        self.corpus_sha256 = sha256_file(self.corpus_path)
        self._validate_root_assets()
        self._split_arrays = self._load_split_arrays()
        self._entries = self._load_schedule_entries()
        self._invariant_cache_contract: Optional[dict[str, Any]] = None

    def _validate_root_assets(self) -> None:
        assets = self.schedule_manifest.get("assets")
        required_assets = {
            "corpus",
            "pretrain_v3_all",
            "u_eval",
            "split_manifest",
            "split_npz",
            "producer",
        }
        if not isinstance(assets, dict) or set(assets) != required_assets:
            raise RuntimeError("schedule manifest assets must match the exact schema")
        for name in required_assets:
            _require_hash(assets.get(name), f"schedule assets.{name}")
        if assets["corpus"] != self.corpus_sha256:
            raise RuntimeError("schedule corpus sha256 does not match the supplied h5ad")
        if assets["split_manifest"] != self.split_manifest_sha256:
            raise RuntimeError("schedule split_manifest sha256 mismatch")
        if assets["split_npz"] != self.split_sha256:
            raise RuntimeError("schedule split_npz sha256 mismatch")
        split_output = self.split_manifest.get("split_npz")
        if not isinstance(split_output, dict) or split_output.get("sha256") != self.split_sha256:
            raise RuntimeError("split manifest does not bind split.npz")
        if int(split_output.get("bytes", -1)) != self.split_path.stat().st_size:
            raise RuntimeError("split.npz byte size mismatch")

    def _load_split_arrays(self) -> dict[str, np.ndarray]:
        expected_hashes = self.split_manifest.get("array_sha256")
        if not isinstance(expected_hashes, dict):
            raise RuntimeError("split manifest array_sha256 is missing")
        arrays: dict[str, np.ndarray] = {}
        with np.load(self.split_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected_hashes):
                raise RuntimeError("split.npz keys do not exactly match split manifest")
            for name in archive.files:
                value = np.asarray(archive[name])
                if sha256_array(value) != expected_hashes[name]:
                    raise RuntimeError(f"split array sha256 mismatch: {name}")
                arrays[name] = value
        for kind, prefix in SPLIT_PREFIX.items():
            rows = _require_int64_vector(arrays.get(f"{prefix}_rows"), f"{kind} rows")
            _validate_unique_rows(rows, f"{kind} rows")
            for suffix in ("project_codes", "site_codes"):
                metadata = np.asarray(arrays.get(f"{prefix}_{suffix}"))
                if metadata.shape != rows.shape or metadata.dtype.kind not in "iu":
                    raise RuntimeError(f"{kind} {suffix} must be integer [N] aligned to rows")
                if np.any(metadata < 0):
                    raise RuntimeError(f"{kind} {suffix} contains missing codes")
        return arrays

    def split(self, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            prefix = SPLIT_PREFIX[kind]
        except KeyError as exc:
            raise ValueError(f"unknown split kind: {kind!r}") from exc
        return (
            np.asarray(self._split_arrays[f"{prefix}_rows"], dtype=np.int64),
            np.asarray(self._split_arrays[f"{prefix}_project_codes"]),
            np.asarray(self._split_arrays[f"{prefix}_site_codes"]),
        )

    def validate_disease_rows(
        self,
        path: str | os.PathLike[str],
        *,
        expected_count: int = 12_102,
    ) -> np.ndarray:
        """Bind the canonical disease rows to file, split sidecar and logical hash."""

        source = Path(path).resolve()
        file_hash = sha256_file(source)
        schedule_u_eval = self.schedule_manifest.get("assets", {}).get("u_eval")
        split_u_eval = self.split_manifest.get("assets", {}).get("u_eval")
        if file_hash != schedule_u_eval or file_hash != split_u_eval:
            raise RuntimeError("canonical disease row file hash differs from strict split assets")
        rows = np.asarray(np.load(source, allow_pickle=False))
        if rows.dtype != np.int64 or rows.shape != (expected_count,):
            raise RuntimeError(
                f"canonical disease rows must be int64 [{expected_count}], got {rows.dtype} {rows.shape}"
            )
        if np.unique(rows).size != rows.size:
            raise RuntimeError("canonical disease rows contain duplicates")
        sidecar_rows = self._split_arrays.get("u_eval_rows")
        if sidecar_rows is None or not np.array_equal(rows, sidecar_rows):
            raise RuntimeError("canonical disease rows differ from split.npz u_eval_rows")
        expected_logical = self.split_manifest.get("array_sha256", {}).get("u_eval_rows")
        if sha256_array(rows) != expected_logical:
            raise RuntimeError("canonical disease row-list logical hash mismatch")
        return rows

    def _load_schedule_entries(self) -> dict[tuple[str, int], _ScheduleEntry]:
        schedules = self.schedule_manifest.get("schedules")
        if not isinstance(schedules, dict):
            raise RuntimeError("schedule manifest schedules must be an object")
        entries: dict[tuple[str, int], _ScheduleEntry] = {}
        for kind in SPLIT_PREFIX:
            raw_entries = schedules.get(kind)
            if not isinstance(raw_entries, list):
                raise RuntimeError(f"schedule manifest schedules.{kind} must be a list")
            for raw in raw_entries:
                if not isinstance(raw, dict):
                    raise RuntimeError(f"schedule {kind} entry must be an object")
                required_entry_keys = {
                    "index",
                    "path",
                    "sha256",
                    "scheduled_rows_sha256",
                    "batch_offsets_sha256",
                    "omitted_rows_sha256",
                    "n_batches",
                    "n_scheduled",
                    "n_omitted",
                    "last_batch_size",
                }
                if set(raw) != required_entry_keys:
                    raise RuntimeError(f"schedule {kind} entry schema mismatch")
                index = int(raw["index"])
                entry = _ScheduleEntry(
                    kind=kind,
                    index=index,
                    path=_resolve_inside(self.schedule_root, raw["path"], f"schedule {kind}/{index}"),
                    sha256=_require_hash(raw["sha256"], f"schedule {kind}/{index}.sha256"),
                    scheduled_rows_sha256=_require_hash(
                        raw["scheduled_rows_sha256"],
                        f"schedule {kind}/{index}.scheduled_rows_sha256",
                    ),
                    batch_offsets_sha256=_require_hash(
                        raw["batch_offsets_sha256"],
                        f"schedule {kind}/{index}.batch_offsets_sha256",
                    ),
                    omitted_rows_sha256=_require_hash(
                        raw["omitted_rows_sha256"], f"schedule {kind}/{index}.omitted_rows_sha256"
                    ),
                    n_batches=int(raw["n_batches"]),
                    n_scheduled=int(raw["n_scheduled"]),
                    n_omitted=int(raw["n_omitted"]),
                    last_batch_size=int(raw["last_batch_size"]),
                )
                if (kind, index) in entries:
                    raise RuntimeError(f"duplicate schedule entry {kind}/{index}")
                if sha256_file(entry.path) != entry.sha256:
                    raise RuntimeError(f"schedule {kind}/{index} sha256 mismatch")
                scheduled_rows, batch_offsets, omitted_rows = self._load_schedule_arrays(entry)
                self._validate_schedule_coverage(
                    kind, entry, scheduled_rows, batch_offsets, omitted_rows
                )
                entries[(kind, index)] = entry
        expected_indices = {
            "train": set(range(10)),
            "val": set(range(4)),
            "test": set(range(4)),
        }
        for kind, expected in expected_indices.items():
            actual = {index for entry_kind, index in entries if entry_kind == kind}
            if actual != expected:
                raise RuntimeError(
                    f"schedule {kind} indices must be exactly {sorted(expected)}, got {sorted(actual)}"
                )
        return entries

    def _load_schedule_arrays(
        self, entry: _ScheduleEntry
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(entry.path, allow_pickle=False) as archive:
            if set(archive.files) != {"scheduled_rows", "batch_offsets", "omitted_rows"}:
                raise RuntimeError(f"schedule {entry.kind}/{entry.index} has unexpected arrays")
            scheduled_rows = np.asarray(archive["scheduled_rows"])
            batch_offsets = np.asarray(archive["batch_offsets"])
            omitted_rows = np.asarray(archive["omitted_rows"])
        scheduled_rows = _require_int64_vector(scheduled_rows, "schedule scheduled_rows")
        batch_offsets = _require_int64_vector(batch_offsets, "schedule batch_offsets")
        omitted_rows = _require_int64_vector(omitted_rows, "schedule omitted_rows")
        if scheduled_rows.size != entry.n_scheduled:
            raise RuntimeError("schedule scheduled_rows size mismatch")
        if batch_offsets.shape != (entry.n_batches + 1,):
            raise RuntimeError("schedule batch_offsets size mismatch")
        if batch_offsets[0] != 0 or batch_offsets[-1] != scheduled_rows.size:
            raise RuntimeError("schedule batch_offsets do not span scheduled_rows")
        increments = np.diff(batch_offsets)
        if np.any(increments <= 0) or np.any(increments > PHYSICAL_BATCH_SIZE):
            raise RuntimeError("schedule batches must contain 1..32 samples")
        if entry.kind == "train" and np.any(increments != PHYSICAL_BATCH_SIZE):
            raise RuntimeError("every scheduled training batch must have physical B=32")
        if entry.last_batch_size != int(increments[-1]):
            raise RuntimeError("schedule last_batch_size disagrees with batch_offsets")
        if omitted_rows.size != entry.n_omitted:
            raise RuntimeError("schedule omitted_rows size mismatch")
        if entry.kind in {"val", "test"} and omitted_rows.size:
            raise RuntimeError(f"{entry.kind} schedule may not omit tail rows")
        if sha256_array(scheduled_rows) != entry.scheduled_rows_sha256:
            raise RuntimeError("schedule scheduled_rows sha256 mismatch")
        if sha256_array(batch_offsets) != entry.batch_offsets_sha256:
            raise RuntimeError("schedule batch_offsets sha256 mismatch")
        if sha256_array(omitted_rows) != entry.omitted_rows_sha256:
            raise RuntimeError("schedule omitted_rows sha256 mismatch")
        return scheduled_rows, batch_offsets, omitted_rows

    def _validate_schedule_coverage(
        self,
        kind: str,
        entry: _ScheduleEntry,
        scheduled_rows: np.ndarray,
        batch_offsets: np.ndarray,
        omitted_rows: np.ndarray,
    ) -> None:
        coverage = np.concatenate([scheduled_rows, omitted_rows])
        _validate_unique_rows(coverage, f"schedule {kind}/{entry.index}")
        split_rows = self.split(kind)[0]
        if not np.array_equal(np.sort(coverage), np.sort(split_rows)):
            raise RuntimeError(f"schedule {kind}/{entry.index} does not exactly cover strict split")

    def available_indices(self, kind: str) -> tuple[int, ...]:
        return tuple(sorted(index for entry_kind, index in self._entries if entry_kind == kind))

    def _cache_directory(self, entry: _ScheduleEntry) -> Path:
        return self.cache_root / entry.path.stem

    def _validate_cache(
        self,
        entry: _ScheduleEntry,
        scheduled_rows: np.ndarray,
        batch_offsets: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], str, str, float]:
        directory = self._cache_directory(entry)
        complete_path = directory / ".complete"
        manifest_path = directory / "teacher_cache_manifest.json"
        cache_path = directory / "teacher_cache.npz"
        complete = _load_json(complete_path)
        expected_complete_keys = {
            "schema_version",
            "manifest_sha256",
            "cache_sha256",
            "schedule_file_sha256",
            "schedule_scheduled_rows_sha256",
        }
        if set(complete) != expected_complete_keys:
            raise RuntimeError(".complete sentinel schema mismatch")
        manifest_sha = sha256_file(manifest_path)
        cache_sha = sha256_file(cache_path)
        checks = {
            "manifest_sha256": manifest_sha,
            "cache_sha256": cache_sha,
            "schedule_file_sha256": entry.sha256,
            "schedule_scheduled_rows_sha256": entry.scheduled_rows_sha256,
        }
        for key, actual in checks.items():
            if complete.get(key) != actual:
                raise RuntimeError(f"completed cache {entry.kind}/{entry.index} {key} mismatch")

        manifest = _load_json(manifest_path)
        contract = manifest.get("contract")
        if not isinstance(contract, dict):
            raise RuntimeError("cache manifest contract is missing")
        assets = contract.get("assets")
        if not isinstance(assets, dict) or set(assets) != REQUIRED_CACHE_ASSETS:
            raise RuntimeError("cache contract assets do not match required exact schema")
        for name, value in assets.items():
            _require_hash(value, f"cache assets.{name}")
        expected_asset_values = {
            "corpus": self.corpus_sha256,
            "pretrain_v3_all": self.schedule_manifest["assets"]["pretrain_v3_all"],
            "u_eval": self.schedule_manifest["assets"]["u_eval"],
            "split_manifest": self.split_manifest_sha256,
            "split_npz": self.split_sha256,
            "schedule_manifest": self.schedule_manifest_sha256,
            "schedule_file": entry.sha256,
        }
        for name, value in expected_asset_values.items():
            if assets[name] != value:
                raise RuntimeError(f"cache contract asset {name} sha256 mismatch")
        if contract.get("solver") != EXACT_SOLVER_CONTRACT:
            raise RuntimeError("cache solver contract is not the locked exact implementation")
        teacher_validity_contract = contract.get("teacher_validity")
        if teacher_validity_contract != TEACHER_VALIDITY_CONTRACT:
            raise RuntimeError("cache teacher-validity contract drifted")
        valid_mass_min = float(
            teacher_validity_contract["protein_endpoint_valid_mass_min"]
        )
        if contract.get("teacher_compare_tolerance") != TEACHER_COMPARE_TOLERANCE:
            raise RuntimeError("cache teacher comparison tolerance drifted")
        if contract.get("pair_order") != "row-major np.triu_indices(actual_batch_size, k=1)":
            raise RuntimeError("cache upper-triangle pair order mismatch")
        schedule_contract = contract.get("schedule")
        if not isinstance(schedule_contract, dict):
            raise RuntimeError("cache schedule contract is missing")
        expected_schedule = {
            "kind": entry.kind,
            "index": entry.index,
            "nominal_batch_size": PHYSICAL_BATCH_SIZE,
            "n_batches": entry.n_batches,
            "n_scheduled": entry.n_scheduled,
            "n_omitted": entry.n_omitted,
            "last_batch_size": entry.last_batch_size,
            "schedule_file_sha256": entry.sha256,
            "scheduled_rows_sha256": entry.scheduled_rows_sha256,
            "batch_offsets_sha256": entry.batch_offsets_sha256,
            "omitted_rows_sha256": entry.omitted_rows_sha256,
        }
        if schedule_contract != expected_schedule:
            raise RuntimeError("cache schedule contract mismatch")

        invariant = {
            "assets": {k: v for k, v in assets.items() if k != "schedule_file"},
            "solver": contract["solver"],
            "teacher_validity": teacher_validity_contract,
            "teacher_compare_tolerance": contract["teacher_compare_tolerance"],
        }
        if self._invariant_cache_contract is None:
            self._invariant_cache_contract = invariant
        elif invariant != self._invariant_cache_contract:
            raise RuntimeError("teacher cache contract drifted across schedules")

        output = manifest.get("output")
        if not isinstance(output, dict) or output.get("sha256") != cache_sha:
            raise RuntimeError("cache manifest output sha256 mismatch")
        expected_array_hashes = output.get("array_sha256")
        required_arrays = {
            "scheduled_rows",
            "batch_offsets",
            "pair_offsets",
            "project_codes",
            "site_codes",
            "protein_distance_upper",
            "unifrac_distance_upper",
            "protein_valid_mass",
            "protein_borrowed_mass",
            "protein_endpoint_valid",
            "protein_pair_valid_upper",
        }
        if not isinstance(expected_array_hashes, dict) or set(expected_array_hashes) != required_arrays:
            raise RuntimeError("cache output array_sha256 schema mismatch")
        arrays: dict[str, np.ndarray] = {}
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != required_arrays:
                raise RuntimeError("teacher_cache.npz array schema mismatch")
            for name in archive.files:
                value = np.asarray(archive[name])
                if sha256_array(value) != expected_array_hashes[name]:
                    raise RuntimeError(f"cache array sha256 mismatch: {name}")
                arrays[name] = value
        if arrays["scheduled_rows"].dtype != np.int64 or not np.array_equal(
            arrays["scheduled_rows"], scheduled_rows
        ):
            raise RuntimeError("cache scheduled_rows does not exactly equal schedule scheduled_rows")
        if arrays["batch_offsets"].dtype != np.int64 or not np.array_equal(
            arrays["batch_offsets"], batch_offsets
        ):
            raise RuntimeError("cache batch_offsets does not exactly equal schedule batch_offsets")
        pair_offsets = arrays["pair_offsets"]
        if pair_offsets.dtype != np.int64 or pair_offsets.shape != (entry.n_batches + 1,):
            raise RuntimeError("cache pair_offsets must be int64 [n_batches+1]")
        expected_pair_counts = np.asarray(
            [size * (size - 1) // 2 for size in np.diff(batch_offsets)], dtype=np.int64
        )
        if pair_offsets[0] != 0 or not np.array_equal(np.diff(pair_offsets), expected_pair_counts):
            raise RuntimeError("cache pair_offsets do not match schedule batch sizes")
        expected_upper_shape = (int(pair_offsets[-1]),)
        for name in ("protein_distance_upper", "unifrac_distance_upper"):
            value = arrays[name]
            if value.dtype != np.float64 or value.shape != expected_upper_shape:
                raise RuntimeError(f"cache {name} must be float64 {expected_upper_shape}")
            if not np.isfinite(value).all() or np.any(value < 0):
                raise RuntimeError(f"cache {name} contains invalid distances")
        expected_endpoint_shape = (entry.n_scheduled,)
        if arrays["project_codes"].dtype != np.int32 or arrays["project_codes"].shape != expected_endpoint_shape:
            raise RuntimeError("cache project_codes must be int32 [N_scheduled]")
        if arrays["site_codes"].dtype != np.int16 or arrays["site_codes"].shape != expected_endpoint_shape:
            raise RuntimeError("cache site_codes must be int16 [N_scheduled]")
        split_rows, split_projects, split_sites = self.split(entry.kind)
        sort_order = np.argsort(split_rows)
        sorted_rows = split_rows[sort_order]
        positions = np.searchsorted(sorted_rows, scheduled_rows)
        if np.any(positions >= sorted_rows.size) or not np.array_equal(
            sorted_rows[positions], scheduled_rows
        ):
            raise RuntimeError("scheduled rows cannot be mapped back to strict split metadata")
        if not np.array_equal(
            arrays["project_codes"], split_projects[sort_order[positions]].astype(np.int32)
        ):
            raise RuntimeError("cache project_codes do not match strict split metadata")
        if not np.array_equal(
            arrays["site_codes"], split_sites[sort_order[positions]].astype(np.int16)
        ):
            raise RuntimeError("cache site_codes do not match strict split metadata")
        for name in ("protein_valid_mass", "protein_borrowed_mass"):
            value = arrays[name]
            if value.dtype != np.float64 or value.shape != expected_endpoint_shape:
                raise RuntimeError(f"cache {name} must be float64 {expected_endpoint_shape}")
            if not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1 + 1e-9):
                raise RuntimeError(f"cache {name} lies outside [0, 1]")
        if np.any(arrays["protein_borrowed_mass"] > arrays["protein_valid_mass"] + 1e-12):
            raise RuntimeError("cache protein_borrowed_mass exceeds protein_valid_mass")
        endpoint_valid = arrays["protein_endpoint_valid"]
        if endpoint_valid.dtype != np.bool_ or endpoint_valid.shape != expected_endpoint_shape:
            raise RuntimeError("cache protein_endpoint_valid must be bool [N_scheduled]")
        if not np.array_equal(
            endpoint_valid,
            arrays["protein_valid_mass"] >= PROTEIN_VALID_MASS_MIN,
        ):
            raise RuntimeError("cache protein_endpoint_valid disagrees with the 0.90 mass gate")
        pair_valid = arrays["protein_pair_valid_upper"]
        if pair_valid.dtype != np.bool_ or pair_valid.shape != expected_upper_shape:
            raise RuntimeError("cache protein_pair_valid_upper must be bool [P]")
        expected_pair_valid_parts: list[np.ndarray] = []
        for batch_index in range(entry.n_batches):
            start, stop = (int(x) for x in batch_offsets[batch_index : batch_index + 2])
            local = endpoint_valid[start:stop]
            i, j = np.triu_indices(local.size, k=1)
            expected_pair_valid_parts.append(local[i] & local[j])
        expected_pair_valid = np.concatenate(expected_pair_valid_parts)
        if not np.array_equal(pair_valid, expected_pair_valid):
            raise RuntimeError("cache protein pair validity is not both-endpoints-valid")
        return arrays, manifest_sha, cache_sha, valid_mass_min

    def epoch_dataset(
        self,
        kind: str,
        index: int,
        samples: RelationAnnDataDataset,
    ) -> RelationEpochDataset:
        try:
            entry = self._entries[(kind, int(index))]
        except KeyError as exc:
            raise RuntimeError(f"no published schedule for {kind}/{index}") from exc
        expected_split = self.split(kind)[0]
        if not np.array_equal(samples.split_rows, expected_split):
            raise RuntimeError("sample dataset split/order differs from strict split sidecar")
        scheduled_rows, batch_offsets, _ = self._load_schedule_arrays(entry)
        arrays, manifest_sha, cache_sha, valid_mass_min = self._validate_cache(
            entry, scheduled_rows, batch_offsets
        )
        return RelationEpochDataset(
            samples,
            entry=entry,
            scheduled_rows=scheduled_rows,
            batch_offsets=batch_offsets,
            pair_offsets=arrays["pair_offsets"],
            cache_arrays=arrays,
            cache_manifest_sha256=manifest_sha,
            cache_sha256=cache_sha,
            protein_valid_mass_min=valid_mass_min,
        )

    def checkpoint_binding(self) -> dict[str, Any]:
        binding = {
            "schema_version": SCHEMA_VERSION,
            "corpus_sha256": self.corpus_sha256,
            "schedule_manifest_sha256": self.schedule_manifest_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
            "split_npz_sha256": self.split_sha256,
            "physical_batch_size": PHYSICAL_BATCH_SIZE,
            "schedule_parameters": self.schedule_manifest.get("parameters"),
            "schedule_assets": self.schedule_manifest.get("assets"),
        }
        binding["binding_sha256"] = _canonical_json_hash(binding)
        return binding

    def validate_all_caches(self) -> list[dict[str, str]]:
        """Validate every published cache and return its checkpoint-safe hashes."""

        records: list[dict[str, str]] = []
        for key in sorted(self._entries):
            entry = self._entries[key]
            scheduled_rows, batch_offsets, _ = self._load_schedule_arrays(entry)
            _, manifest_sha, cache_sha, _ = self._validate_cache(
                entry, scheduled_rows, batch_offsets
            )
            records.append(
                {
                    "cache_manifest_sha256": manifest_sha,
                    "cache_sha256": cache_sha,
                    "schedule_file_sha256": entry.sha256,
                }
            )
        return records

    def verify_cache_provenance(self, records: Sequence[Mapping[str, str]]) -> None:
        available = {
            (
                record["cache_manifest_sha256"],
                record["cache_sha256"],
                record["schedule_file_sha256"],
            )
            for record in self.validate_all_caches()
        }
        requested = {
            (
                record.get("cache_manifest_sha256"),
                record.get("cache_sha256"),
                record.get("schedule_file_sha256"),
            )
            for record in records
        }
        if not requested.issubset(available):
            raise RuntimeError("checkpoint references a teacher cache absent from this exact bundle")


class RelationDataModule(L.LightningDataModule):
    """Reload an exact, pre-batched schedule at each Lightning epoch."""

    def __init__(
        self,
        *,
        h5ad_path: str | os.PathLike[str],
        schedule_root: str | os.PathLike[str],
        cache_root: str | os.PathLike[str],
        num_workers: int = 0,
        pin_memory: bool = True,
        expected_n_vars: int = 8_114,
    ) -> None:
        super().__init__()
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        self.h5ad_path = Path(h5ad_path)
        self.store = RelationScheduleStore(schedule_root, cache_root, h5ad_path)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.expected_n_vars = int(expected_n_vars)
        self.datasets: dict[str, RelationAnnDataDataset] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        for kind in SPLIT_PREFIX:
            if kind not in self.datasets:
                rows, projects, sites = self.store.split(kind)
                self.datasets[kind] = RelationAnnDataDataset(
                    self.h5ad_path,
                    split_rows=rows,
                    project_codes=projects,
                    site_codes=sites,
                    max_tokens=MAX_TOKENS,
                    expected_n_vars=self.expected_n_vars,
                )

    def _loader(self, dataset: Dataset[Any]) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        self.setup("fit")
        epoch = int(getattr(getattr(self, "trainer", None), "current_epoch", 0))
        dataset = self.store.epoch_dataset("train", epoch, self.datasets["train"])
        return self._loader(dataset)

    def _evaluation_loader(self, kind: str) -> DataLoader[Any]:
        self.setup("validate" if kind == "val" else "test")
        indices = self.store.available_indices(kind)
        if not indices:
            raise RuntimeError(f"no {kind} schedules are published")
        datasets = [
            self.store.epoch_dataset(kind, index, self.datasets[kind]) for index in indices
        ]
        return self._loader(ConcatDataset(datasets))

    def val_dataloader(self) -> DataLoader[Any]:
        return self._evaluation_loader("val")

    def test_dataloader(self) -> DataLoader[Any]:
        return self._evaluation_loader("test")

    @property
    def checkpoint_binding(self) -> dict[str, Any]:
        return self.store.checkpoint_binding()
