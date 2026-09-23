"""Pilot data reader and collator for the fixed-PP continuation."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from micoformer.mpc_pretraining.data import (
    deterministic_native_mlm_mask,
    deterministic_negative_ids,
    deterministic_query_mask,
)


MAX_TOKENS = 512


class PilotRowDataset(Dataset[dict[str, Any]]):
    """Read one full-support H5AD row and expose both student and teacher views."""

    def __init__(self, corpus: str | Path, rows: np.ndarray) -> None:
        self.corpus = Path(corpus).resolve()
        self.rows = np.asarray(rows, dtype=np.int64)
        if self.rows.ndim != 1 or self.rows.size == 0:
            raise ValueError("rows must be a non-empty int64 vector")
        with h5py.File(self.corpus, "r") as handle:
            self.shape = tuple(int(value) for value in np.asarray(handle["X"].attrs["shape"]))
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
        row = int(self.rows[position])
        x = self._h5()["X"]
        begin, end = int(x["indptr"][row]), int(x["indptr"][row + 1])
        indices = np.asarray(x["indices"][begin:end], dtype=np.int64)
        abundance = np.asarray(x["data"][begin:end], dtype=np.float32)
        if indices.size < 5 or indices.shape != abundance.shape:
            raise RuntimeError(f"row {row} violates the five-present-genus contract")
        order = np.lexsort((indices, -abundance))[:MAX_TOKENS]
        kept_ids = indices[order]
        kept_abundance = abundance[order]
        log_abundance = np.log(kept_abundance.astype(np.float64))
        rclr = (log_abundance - log_abundance.mean(dtype=np.float64)).astype(np.float32)
        return {
            "row_id": np.int64(row),
            "full_indices": indices,
            "full_abundance": abundance,
            "genus_ids": (kept_ids + 2).astype(np.int64),
            "rclr": rclr,
        }


class PilotCollator:
    """Build MPC native corruption plus full composition for the teacher."""

    def __init__(
        self,
        *,
        observed_genus_ids: np.ndarray,
        seed: int,
        epoch: int = 0,
        shared_epoch: Any | None = None,
    ) -> None:
        self.observed = np.asarray(observed_genus_ids, dtype=np.int64)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shared_epoch = shared_epoch

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        if not samples:
            raise ValueError("empty batch")
        batch_size = len(samples)
        max_length = max(int(item["genus_ids"].size) for item in samples)
        genus = np.zeros((batch_size, max_length), dtype=np.int64)
        rclr = np.zeros((batch_size, max_length), dtype=np.float32)
        padding = np.ones((batch_size, max_length), dtype=np.bool_)
        full_composition = np.zeros((batch_size, 8_114), dtype=np.float32)
        supports: list[np.ndarray] = []
        rows = np.asarray([int(item["row_id"]) for item in samples], dtype=np.int64)
        for index, sample in enumerate(samples):
            length = int(sample["genus_ids"].size)
            genus[index, :length] = sample["genus_ids"]
            rclr[index, :length] = sample["rclr"]
            padding[index, :length] = False
            full_indices = np.asarray(sample["full_indices"], dtype=np.int64)
            full_composition[index, full_indices] = np.asarray(
                sample["full_abundance"], dtype=np.float32
            )
            supports.append(full_indices)
        epoch = int(self.shared_epoch.value) if self.shared_epoch is not None else self.epoch
        query = deterministic_query_mask(
            padding, rows, seed=self.seed, stream=1, epoch=epoch
        )
        mlm = deterministic_native_mlm_mask(
            padding, rows, seed=self.seed, stream=1, epoch=epoch
        )
        negatives = deterministic_negative_ids(
            genus,
            query,
            supports,
            self.observed,
            rows,
            seed=self.seed,
            stream=1,
            epoch=epoch,
        )
        return {
            "row_ids": torch.from_numpy(rows),
            "genus_ids": torch.from_numpy(genus),
            "rclr": torch.from_numpy(rclr),
            "padding_mask": torch.from_numpy(padding),
            "query_mask": torch.from_numpy(query),
            "mlm_mask": torch.from_numpy(mlm),
            "negative_genus_ids": torch.from_numpy(negatives),
            "full_composition": torch.from_numpy(full_composition),
        }
