"""Deterministic full-corpus schedules for the bounded fast run."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
from torch.utils.data import DataLoader, Dataset

from .data import (
    FullscaleAnnDataDataset,
    collate_fullscale_samples,
    deterministic_abundance_mask,
)


class FastScheduledRelationMLMDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        samples: FullscaleAnnDataDataset,
        *,
        relation_rows: np.ndarray,
        mlm_rows: np.ndarray,
        seed: int = 42,
    ) -> None:
        relation = np.asarray(relation_rows)
        mlm = np.asarray(mlm_rows)
        if relation.dtype != np.int64 or relation.ndim != 2 or relation.shape[1] != 32:
            raise TypeError("rank-local relation_rows must be int64 [N,32]")
        if mlm.dtype != np.int64 or mlm.ndim != 2 or mlm.shape[1] != 32:
            raise TypeError("rank-local mlm_rows must be int64 [M,32]")
        expected_mlm = relation.shape[0] // 4
        if mlm.shape[0] < expected_mlm:
            raise ValueError("MLM schedule does not cover the 4:1 cadence")
        if np.any(relation < -1) or np.any(mlm < 0):
            raise ValueError("only relation tail padding may use -1")
        valid_per_batch = (relation >= 0).sum(axis=1)
        if np.any(valid_per_batch < 2) or np.any(valid_per_batch > 32):
            raise ValueError("each rank-local relation batch must contain 2..32 rows")
        self.samples = samples
        self.relation_rows = relation.copy()
        self.mlm_rows = mlm[:expected_mlm].copy()
        self.seed = int(seed)

    def __len__(self) -> int:
        return int(self.relation_rows.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        rows = self.relation_rows[index]
        rows = rows[rows >= 0]
        relation = collate_fullscale_samples(
            [self.samples.get_by_global_row(int(row)) for row in rows]
        )
        relation["schedule_batch_index"] = np.int64(index)
        mlm_batch: dict[str, Any] | None = None
        if (index + 1) % 4 == 0:
            mlm_index = (index + 1) // 4 - 1
            mlm_rows = self.mlm_rows[mlm_index]
            mlm_batch = collate_fullscale_samples(
                [self.samples.get_by_global_row(int(row)) for row in mlm_rows]
            )
            mlm_batch["abundance_mask"] = deterministic_abundance_mask(
                mlm_batch["padding_mask"],
                mlm_batch["row_ids"],
                seed=self.seed,
                schedule_index=0x46415354,
                batch_index=mlm_index,
            )
        return {"relation": relation, "mlm": mlm_batch}


class FastFullCorpusDataModule(L.LightningDataModule):
    def __init__(
        self,
        *,
        corpus: str | Path,
        metadata: dict[str, np.ndarray],
        schedule: dict[str, np.ndarray],
        epochs: int,
        loader_workers: int,
        max_relation_steps: int | None = None,
    ) -> None:
        super().__init__()
        self.corpus = Path(corpus)
        self.metadata = metadata
        self.schedule = schedule
        self.epochs = int(epochs)
        self.loader_workers = int(loader_workers)
        self.max_relation_steps = max_relation_steps
        self.samples: FullscaleAnnDataDataset | None = None
        self.dataset: FastScheduledRelationMLMDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.trainer.world_size != 2 or self.trainer.global_rank not in {0, 1}:
            raise RuntimeError("fast full-corpus run requires exactly two DDP ranks")
        relation_all = self.schedule["relation_rows"][: self.epochs]
        relation = relation_all[:, :, self.trainer.global_rank, :].reshape(-1, 32)
        if self.max_relation_steps is not None:
            relation = relation[: self.max_relation_steps]
        mlm_count = relation.shape[0] // 4
        mlm = self.schedule["mlm_rows"][:mlm_count, self.trainer.global_rank, :]
        self.samples = FullscaleAnnDataDataset(
            self.corpus,
            split_rows=self.metadata["train_rows"],
            study_codes=self.metadata["study_codes"],
            database_codes=self.metadata["database_codes"],
        )
        self.dataset = FastScheduledRelationMLMDataset(
            self.samples,
            relation_rows=relation,
            mlm_rows=mlm,
            seed=42,
        )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            raise RuntimeError("fast data module was not set up")
        return DataLoader(
            self.dataset,
            batch_size=None,
            shuffle=False,
            num_workers=self.loader_workers,
            persistent_workers=self.loader_workers > 0,
            prefetch_factor=2 if self.loader_workers > 0 else None,
        )

    def teardown(self, stage: str | None = None) -> None:
        del stage
        if self.samples is not None:
            self.samples.close()
