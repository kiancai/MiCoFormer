"""Full-support HDF5 input and replayable native MPC corruptions."""
from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


MAX_TOKENS = 512


class SharedEpoch:
    """Epoch counter visible to persistent DataLoader worker processes."""

    def __init__(self, initial: int = 0) -> None:
        self._value = mp.Value("q", int(initial))

    @property
    def value(self) -> int:
        with self._value.get_lock():
            return int(self._value.value)

    def set(self, epoch: int) -> None:
        with self._value.get_lock():
            self._value.value = int(epoch)


class FullSupportRowDataset(Dataset[dict[str, Any]]):
    """Read a CSR row once and retain top-512 content plus pre-truncation support."""

    def __init__(self, corpus: str | Path, rows: np.ndarray) -> None:
        self.corpus = Path(corpus).resolve()
        self.rows = np.asarray(rows, dtype=np.int64)
        if self.rows.ndim != 1 or self.rows.size == 0:
            raise ValueError("rows must be a non-empty int64 vector")
        with h5py.File(self.corpus, "r") as handle:
            self.shape = tuple(int(x) for x in np.asarray(handle["X"].attrs["shape"]))
        if np.any(self.rows < 0) or np.any(self.rows >= self.shape[0]):
            raise ValueError("row index lies outside the corpus")
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return int(self.rows.size)

    def _h5(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.corpus, "r")
        return self._handle

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handle"] = None
        return state

    def __getitem__(self, position: int) -> dict[str, Any]:
        row_id = int(self.rows[position])
        x = self._h5()["X"]
        bounds = np.asarray(x["indptr"][row_id : row_id + 2], dtype=np.int64)
        begin, end = int(bounds[0]), int(bounds[1])
        full_indices = np.asarray(x["indices"][begin:end], dtype=np.int64)
        full_abundance = np.asarray(x["data"][begin:end], dtype=np.float64)
        if full_indices.size < 5 or full_indices.shape != full_abundance.shape:
            raise RuntimeError(f"row {row_id} violates the >=5-present-genus contract")
        order = np.lexsort((full_indices, -full_abundance))[:MAX_TOKENS]
        kept_indices = full_indices[order]
        log_abundance = np.log(full_abundance[order], dtype=np.float64)
        rclr = (log_abundance - log_abundance.mean(dtype=np.float64)).astype(np.float32)
        return {
            "row_id": np.int64(row_id),
            "genus_ids": (kept_indices + 2).astype(np.int64, copy=False),
            "rclr": rclr,
            "full_indices": full_indices.astype(np.int64, copy=False),
            "richness": np.int64(full_indices.size),
        }


def _row_rng(seed: int, stream: int, epoch: int, row_id: int, magic: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([
        int(seed), int(stream), int(epoch), int(row_id) & 0xFFFFFFFF,
        int(row_id) >> 32, int(magic),
    ]))


def deterministic_native_mlm_mask(
    padding_mask: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    stream: int,
    epoch: int,
    probability: float = 0.15,
) -> np.ndarray:
    if probability != 0.15:
        raise ValueError("the frozen native MLM probability is .15")
    padding = np.asarray(padding_mask)
    rows = np.asarray(row_ids)
    if padding.dtype != np.bool_ or padding.ndim != 2:
        raise TypeError("padding_mask must be bool [B,L]")
    if rows.dtype != np.int64 or rows.shape != (padding.shape[0],):
        raise TypeError("row_ids must be aligned int64 [B]")
    mask = np.zeros_like(padding)
    for sample_index, row_id in enumerate(rows.tolist()):
        valid = np.flatnonzero(~padding[sample_index])
        rng = _row_rng(seed, stream, epoch, int(row_id), 0x4D4C4D)
        selected = valid[rng.random(valid.size) < probability]
        if not selected.size:
            selected = valid[[int(rng.integers(0, valid.size))]]
        if selected.size >= valid.size:
            selected = np.sort(selected)[: valid.size - 1]
        mask[sample_index, selected] = True
    if np.any(mask & padding) or np.any((~(padding | mask)).sum(1) < 1):
        raise RuntimeError("invalid native MLM corruption")
    return mask


def deterministic_query_mask(
    padding_mask: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    stream: int,
    epoch: int,
    probability: float = 0.10,
) -> np.ndarray:
    if probability != 0.10:
        raise ValueError("the frozen Query probability is .10")
    padding = np.asarray(padding_mask)
    rows = np.asarray(row_ids)
    if padding.dtype != np.bool_ or padding.ndim != 2:
        raise TypeError("padding_mask must be bool [B,L]")
    if rows.dtype != np.int64 or rows.shape != (padding.shape[0],):
        raise TypeError("row_ids must be aligned int64 [B]")
    mask = np.zeros_like(padding)
    for sample_index, row_id in enumerate(rows.tolist()):
        valid = np.flatnonzero(~padding[sample_index])
        rng = _row_rng(seed, stream, epoch, int(row_id), 0x5155)
        selected = valid[rng.random(valid.size) < probability]
        if not selected.size:
            selected = valid[[int(rng.integers(0, valid.size))]]
        if selected.size > valid.size - 2:
            selected = np.sort(selected)[: valid.size - 2]
        mask[sample_index, selected] = True
    if np.any(mask & padding) or np.any((~(padding | mask)).sum(1) < 2):
        raise RuntimeError("invalid native Query corruption")
    return mask


def deterministic_negative_ids(
    genus_ids: np.ndarray,
    query_mask: np.ndarray,
    full_supports: Sequence[np.ndarray],
    observed_genus_ids: np.ndarray,
    row_ids: np.ndarray,
    *,
    seed: int,
    stream: int,
    epoch: int,
) -> np.ndarray:
    genus = np.asarray(genus_ids)
    query = np.asarray(query_mask)
    rows = np.asarray(row_ids)
    observed = np.asarray(observed_genus_ids)
    if genus.dtype != np.int64 or query.dtype != np.bool_ or genus.shape != query.shape:
        raise TypeError("genus_ids/query_mask must be aligned int64/bool [B,L]")
    if observed.dtype != np.int64 or observed.ndim != 1 or np.any(observed < 2):
        raise TypeError("observed_genus_ids must contain real vocabulary IDs")
    if len(full_supports) != genus.shape[0] or rows.shape != (genus.shape[0],):
        raise ValueError("full supports and row IDs must align with batch")
    result = np.zeros_like(genus)
    batch_pools = [np.asarray(support, dtype=np.int64) + 2 for support in full_supports]
    for sample_index, (support, row_id) in enumerate(zip(full_supports, rows.tolist(), strict=True)):
        support_array = np.asarray(support, dtype=np.int64) + 2
        other = [pool for index, pool in enumerate(batch_pools) if index != sample_index]
        primary = np.unique(np.concatenate(other)) if other else np.empty(0, dtype=np.int64)
        primary = primary[~np.isin(primary, support_array)]
        fallback = observed[~np.isin(observed, support_array)]
        positions = np.flatnonzero(query[sample_index])
        rng = _row_rng(seed, stream, epoch, int(row_id), 0x4E4547)
        pool = primary if primary.size >= positions.size else fallback
        if pool.size < positions.size:
            raise RuntimeError("insufficient confirmed-absent negative genera")
        chosen = rng.choice(pool, size=positions.size, replace=False)
        if np.any(np.isin(chosen, support_array)):
            raise RuntimeError("negative candidate is present in full support")
        result[sample_index, positions] = chosen
    if np.any(result[query] < 2) or np.any(result[~query] != 0):
        raise RuntimeError("negative ID layout drifted")
    return result


class DualCorruptionCollator:
    """Build independent native MLM and Query branches for a shared update."""

    def __init__(
        self,
        *,
        observed_genus_ids: np.ndarray,
        seed: int,
        stream: int,
        shared_epoch: SharedEpoch,
    ) -> None:
        self.observed = np.asarray(observed_genus_ids, dtype=np.int64)
        self.seed = int(seed)
        self.stream = int(stream)
        self.shared_epoch = shared_epoch

    def __call__(self, samples: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not samples:
            raise ValueError("empty batch")
        batch_size = len(samples)
        max_length = max(int(item["genus_ids"].size) for item in samples)
        genus = np.zeros((batch_size, max_length), dtype=np.int64)
        rclr = np.zeros((batch_size, max_length), dtype=np.float32)
        padding = np.ones((batch_size, max_length), dtype=np.bool_)
        rows = np.asarray([item["row_id"] for item in samples], dtype=np.int64)
        richness = np.asarray([item["richness"] for item in samples], dtype=np.int64)
        supports: list[np.ndarray] = []
        for index, sample in enumerate(samples):
            length = int(sample["genus_ids"].size)
            genus[index, :length] = sample["genus_ids"]
            rclr[index, :length] = sample["rclr"]
            padding[index, :length] = False
            supports.append(np.asarray(sample["full_indices"], dtype=np.int64))
        epoch = self.shared_epoch.value
        query = deterministic_query_mask(
            padding, rows, seed=self.seed, stream=self.stream, epoch=epoch
        )
        mlm = deterministic_native_mlm_mask(
            padding, rows, seed=self.seed, stream=self.stream, epoch=epoch
        )
        negatives = deterministic_negative_ids(
            genus,
            query,
            supports,
            self.observed,
            rows,
            seed=self.seed,
            stream=self.stream,
            epoch=epoch,
        )
        return {
            "genus_ids": genus,
            "rclr": rclr,
            "padding_mask": padding,
            "query_mask": query,
            "mlm_mask": mlm,
            "negative_genus_ids": negatives,
            "row_ids": rows,
            "richness": richness,
            "corruption_epoch": np.int64(epoch),
        }


def batch_to_device(batch: dict[str, np.ndarray], device: torch.device) -> dict[str, Tensor]:
    return {
        "genus_ids": torch.as_tensor(batch["genus_ids"], device=device, dtype=torch.long),
        "rclr": torch.as_tensor(batch["rclr"], device=device, dtype=torch.float32),
        "padding_mask": torch.as_tensor(batch["padding_mask"], device=device, dtype=torch.bool),
        "query_mask": torch.as_tensor(batch["query_mask"], device=device, dtype=torch.bool),
        "mlm_mask": torch.as_tensor(batch["mlm_mask"], device=device, dtype=torch.bool),
        "negative_genus_ids": torch.as_tensor(
            batch["negative_genus_ids"], device=device, dtype=torch.long
        ),
    }
