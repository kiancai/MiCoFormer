"""Full-V3 sparse rows and deterministic F1/F2 gate batches."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
from torch.utils.data import Dataset


MAX_TOKENS = 512
MAX_PHYSICAL_BATCH_SIZE = 64


def _int64_vector(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.int64 or array.ndim != 1:
        raise TypeError(f"{label} must be int64 [N]")
    return array


class FullscaleAnnDataDataset(Dataset[dict[str, Any]]):
    """Read only requested CSR rows with aligned composite-study metadata."""

    def __init__(
        self,
        h5ad_path: str | os.PathLike[str],
        *,
        split_rows: np.ndarray,
        study_codes: np.ndarray,
        database_codes: np.ndarray,
        max_tokens: int = MAX_TOKENS,
        expected_n_vars: int = 8_114,
    ) -> None:
        self.h5ad_path = Path(h5ad_path).resolve()
        self.split_rows = _int64_vector(split_rows, "split_rows").copy()
        self.study_codes = _int64_vector(study_codes, "study_codes").copy()
        database = np.asarray(database_codes)
        if database.ndim != 1 or database.dtype.kind not in "iu":
            raise TypeError("database_codes must be an integer [N] vector")
        self.database_codes = database.astype(np.int16, copy=True)
        if not (
            self.study_codes.shape == self.database_codes.shape == self.split_rows.shape
        ):
            raise ValueError("split_rows/study_codes/database_codes must align one-to-one")
        if self.split_rows.size and np.any(self.split_rows[1:] <= self.split_rows[:-1]):
            raise ValueError("split_rows must be strictly increasing for fail-closed lookup")
        if np.any(self.study_codes < 0) or np.any(self.database_codes < 0):
            raise ValueError("training metadata may not contain missing codes")
        if max_tokens <= 0 or max_tokens > MAX_TOKENS:
            raise ValueError(f"max_tokens must be in [1,{MAX_TOKENS}]")
        self.max_tokens = int(max_tokens)
        with h5py.File(self.h5ad_path, "r") as handle:
            x = handle.get("X")
            if not isinstance(x, h5py.Group) or x.attrs.get("encoding-type") != "csr_matrix":
                raise ValueError("fullscale relation corpus X must be an HDF5 CSR group")
            shape = tuple(int(item) for item in np.asarray(x.attrs["shape"]))
        if len(shape) != 2 or shape[1] != expected_n_vars:
            raise ValueError("corpus shape/genus universe differs from the frozen contract")
        self.n_obs, self.n_vars = shape
        if self.split_rows.size and (
            int(self.split_rows.min()) < 0 or int(self.split_rows.max()) >= self.n_obs
        ):
            raise ValueError("split rows escape the corpus")
        self._handle: h5py.File | None = None

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

    def _position(self, global_row: int) -> int:
        position = int(np.searchsorted(self.split_rows, global_row))
        if position >= self.split_rows.size or int(self.split_rows[position]) != global_row:
            raise IndexError(f"global row {global_row} is outside the bound split")
        return position

    def get_by_global_row(self, global_row: int) -> dict[str, Any]:
        position = self._position(int(global_row))
        x = self._h5()["X"]
        bounds = np.asarray(x["indptr"][global_row : global_row + 2], dtype=np.int64)
        begin, end = int(bounds[0]), int(bounds[1])
        var_indices = np.asarray(x["indices"][begin:end], dtype=np.int64)
        abundance = np.asarray(x["data"][begin:end], dtype=np.float64)
        if not var_indices.size or var_indices.shape != abundance.shape:
            raise RuntimeError(f"global row {global_row} has an invalid sparse composition")
        if np.any(var_indices < 0) or np.any(var_indices >= self.n_vars):
            raise RuntimeError(f"global row {global_row} contains an invalid genus index")
        if not np.isfinite(abundance).all() or np.any(abundance <= 0):
            raise RuntimeError(f"global row {global_row} contains invalid present abundance")
        order = np.lexsort((var_indices, -abundance))[: self.max_tokens]
        kept_indices = var_indices[order]
        kept_abundance = abundance[order]
        log_abundance = np.log(kept_abundance, dtype=np.float64)
        rclr = (log_abundance - log_abundance.mean(dtype=np.float64)).astype(np.float32)
        return {
            "global_row_id": np.int64(global_row),
            "study_id": np.int64(self.study_codes[position]),
            "database_id": np.int16(self.database_codes[position]),
            "var_indices": kept_indices.astype(np.int64, copy=False),
            "genus_ids": (kept_indices + 2).astype(np.int64, copy=False),
            "rclr": rclr,
            "raw_richness": np.int64(var_indices.size),
            "student_richness": np.int64(kept_indices.size),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.get_by_global_row(int(self.split_rows[index]))


def collate_fullscale_samples(samples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    if not samples or len(samples) > MAX_PHYSICAL_BATCH_SIZE:
        raise ValueError(f"physical batch must contain 1..{MAX_PHYSICAL_BATCH_SIZE} samples")
    max_length = max(int(sample["genus_ids"].size) for sample in samples)
    genus_ids = np.zeros((len(samples), max_length), dtype=np.int64)
    rclr = np.zeros((len(samples), max_length), dtype=np.float32)
    var_indices = np.full((len(samples), max_length), -1, dtype=np.int64)
    padding_mask = np.ones((len(samples), max_length), dtype=bool)
    for index, sample in enumerate(samples):
        length = int(sample["genus_ids"].size)
        genus_ids[index, :length] = sample["genus_ids"]
        rclr[index, :length] = sample["rclr"]
        var_indices[index, :length] = sample["var_indices"]
        padding_mask[index, :length] = False
    return {
        "genus_ids": genus_ids,
        "rclr": rclr,
        "var_indices": var_indices,
        "padding_mask": padding_mask,
        "row_ids": np.asarray([item["global_row_id"] for item in samples], dtype=np.int64),
        "project_ids": np.asarray([item["study_id"] for item in samples], dtype=np.int64),
        "database_ids": np.asarray([item["database_id"] for item in samples], dtype=np.int16),
        "raw_richness": np.asarray([item["raw_richness"] for item in samples], dtype=np.int64),
        "student_richness": np.asarray(
            [item["student_richness"] for item in samples], dtype=np.int64
        ),
    }


def deterministic_abundance_mask(
    padding_mask: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    schedule_index: int,
    batch_index: int,
    probability: float = 0.15,
) -> np.ndarray:
    padding = np.asarray(padding_mask)
    rows = _int64_vector(np.asarray(row_ids), "row_ids")
    if padding.dtype != np.bool_ or padding.ndim != 2 or padding.shape[0] != rows.size:
        raise TypeError("padding_mask must be bool [B,L] aligned with row_ids")
    if probability != 0.15:
        raise ValueError("the frozen mask probability is exactly 0.15")
    output = np.zeros_like(padding)
    for position, row in enumerate(rows.tolist()):
        valid = np.flatnonzero(~padding[position])
        if not valid.size:
            raise RuntimeError("MLM sample contains no valid token")
        row_low = int(row) & 0xFFFFFFFF
        row_high = int(row) >> 32
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [int(seed), int(schedule_index), int(batch_index), row_low, row_high]
            )
        )
        selected = valid[rng.random(valid.size) < probability]
        if not selected.size:
            selected = valid[[int(rng.integers(0, valid.size))]]
        output[position, selected] = True
    if np.any(output & padding):
        raise RuntimeError("deterministic MLM mask selected padding")
    return output


class ExactTeacherBatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: FullscaleAnnDataDataset,
        *,
        batch_rows: np.ndarray,
        teacher_distances: Mapping[str, np.ndarray],
        teacher_validity: Mapping[str, np.ndarray],
        physical_batch_size: int,
    ) -> None:
        rows = np.asarray(batch_rows)
        if rows.dtype != np.int64 or rows.ndim != 2:
            raise TypeError("batch_rows must be int64 [N,Bmax]")
        if physical_batch_size not in {32, 64} or rows.shape[1] < physical_batch_size:
            raise ValueError("physical_batch_size must be B32/B64 covered by batch_rows")
        self.samples = samples
        self.batch_rows = rows[:, :physical_batch_size].copy()
        expected_shape = (rows.shape[0], physical_batch_size, physical_batch_size)
        self.teacher_distances: dict[str, np.ndarray] = {}
        self.teacher_validity: dict[str, np.ndarray] = {}
        if set(teacher_distances) != {"protein", "unifrac"} or set(teacher_validity) != set(teacher_distances):
            raise ValueError("teacher mappings must contain exactly protein and unifrac")
        for name in teacher_distances:
            distance = np.asarray(teacher_distances[name])[:, :physical_batch_size, :physical_batch_size]
            validity = np.asarray(teacher_validity[name])[:, :physical_batch_size, :physical_batch_size]
            if distance.shape != expected_shape or distance.dtype != np.float64:
                raise TypeError(f"teacher {name} distance must be float64 {expected_shape}")
            if validity.shape != expected_shape or validity.dtype != np.bool_:
                raise TypeError(f"teacher {name} validity must be bool {expected_shape}")
            self.teacher_distances[name] = distance.copy()
            self.teacher_validity[name] = validity.copy()

    def __len__(self) -> int:
        return int(self.batch_rows.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = self.batch_rows[index]
        batch = collate_fullscale_samples(
            [self.samples.get_by_global_row(int(row)) for row in rows]
        )
        if not np.array_equal(batch["row_ids"], rows):
            raise RuntimeError("student rows drifted from exact teacher batch order")
        batch["teacher_distances"] = {
            name: values[index] for name, values in self.teacher_distances.items()
        }
        batch["teacher_validity"] = {
            name: values[index] for name, values in self.teacher_validity.items()
        }
        batch["schedule_batch_index"] = np.int64(index)
        return batch


class DeterministicMLMBatchDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: FullscaleAnnDataDataset,
        *,
        batch_rows: np.ndarray,
        physical_batch_size: int,
        seed: int,
        schedule_index: int,
    ) -> None:
        rows = np.asarray(batch_rows)
        if rows.dtype != np.int64 or rows.ndim != 2 or rows.shape[1] < physical_batch_size:
            raise TypeError("MLM batch_rows must be int64 [N,Bmax]")
        if physical_batch_size not in {32, 64}:
            raise ValueError("MLM physical batch must be B32 or B64")
        self.samples = samples
        self.batch_rows = rows[:, :physical_batch_size].copy()
        self.seed = int(seed)
        self.schedule_index = int(schedule_index)

    def __len__(self) -> int:
        return int(self.batch_rows.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = self.batch_rows[index]
        batch = collate_fullscale_samples(
            [self.samples.get_by_global_row(int(row)) for row in rows]
        )
        batch["abundance_mask"] = deterministic_abundance_mask(
            batch["padding_mask"],
            batch["row_ids"],
            seed=self.seed,
            schedule_index=self.schedule_index,
            batch_index=index,
        )
        return batch


class CadencedRelationMLMDataset(Dataset[dict[str, Any]]):
    """Load an independent MLM batch only at each frozen fourth relation batch."""

    def __init__(
        self,
        relation: ExactTeacherBatchDataset,
        mlm: DeterministicMLMBatchDataset,
        *,
        cadence: int = 4,
    ) -> None:
        if cadence != 4:
            raise ValueError("the frozen relation:MLM cadence is exactly 4:1")
        required = len(relation) // cadence
        if len(mlm) < required:
            raise ValueError("MLM schedule does not cover every fourth relation batch")
        self.relation = relation
        self.mlm = mlm
        self.cadence = cadence

    def __len__(self) -> int:
        return len(self.relation)

    def __getitem__(self, index: int) -> dict[str, Any]:
        mlm_batch = self.mlm[index // self.cadence] if (index + 1) % self.cadence == 0 else None
        return {"relation": self.relation[index], "mlm": mlm_batch}
