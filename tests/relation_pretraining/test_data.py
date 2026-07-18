from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from micoformer.relation_pretraining.data import (
    RelationAnnDataDataset,
    RelationScheduleStore,
    sha256_array,
    sha256_file,
)


def _write_toy_h5ad(path: Path, n_rows: int = 33) -> None:
    base = np.asarray([0.10, 0.10, 0.40, 0.0, 0.20, 0.10], dtype=np.float32)
    matrix = np.stack([np.roll(base, i % base.size) for i in range(n_rows)])
    obs = pd.DataFrame(
        {
            "Project_ID": pd.Categorical([f"P{i % 5}" for i in range(n_rows)]),
            "RM_Sample_Site": pd.Categorical([f"S{i % 3}" for i in range(n_rows)]),
        },
        index=[f"row-{i}" for i in range(n_rows)],
    )
    var = pd.DataFrame(index=[f"g-{i}" for i in range(base.size)])
    ad.AnnData(X=sparse.csr_matrix(matrix), obs=obs, var=var).write_h5ad(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)


def _write_schedule_and_cache(root: Path, cache_root: Path, corpus: Path) -> None:
    split_rows = np.arange(33, dtype=np.int64)
    train_projects = (split_rows % 5).astype(np.int32)
    train_sites = (split_rows % 3).astype(np.int32)
    split_arrays = {
        "u_eval_rows": np.asarray([0, 2, 4], dtype=np.int64),
        "relation_train_rows": split_rows,
        "relation_train_project_codes": train_projects,
        "relation_train_site_codes": train_sites,
        "relation_val_rows": np.arange(32, dtype=np.int64),
        "relation_val_project_codes": train_projects[:32],
        "relation_val_site_codes": train_sites[:32],
        "relation_test_rows": np.arange(32, dtype=np.int64),
        "relation_test_project_codes": train_projects[:32],
        "relation_test_site_codes": train_sites[:32],
    }
    split_path = root / "split.npz"
    _atomic_npz(split_path, **split_arrays)
    split_manifest = {
        "schema_version": 1,
        "array_sha256": {name: sha256_array(value) for name, value in split_arrays.items()},
        "split_npz": {
            "path": "split.npz",
            "bytes": split_path.stat().st_size,
            "sha256": sha256_file(split_path),
        },
    }
    split_manifest_path = root / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, sort_keys=True), encoding="utf-8")

    schedule_entries: dict[str, list[dict[str, object]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    train_zero: tuple[np.ndarray, np.ndarray, np.ndarray, Path] | None = None
    for kind, count in (("train", 10), ("val", 4), ("test", 4)):
        source_rows = split_arrays[f"relation_{kind}_rows"]
        for index in range(count):
            permutation = np.roll(source_rows, -index).astype(np.int64, copy=False)
            if kind == "train":
                local_scheduled = permutation[:32]
                local_omitted = permutation[32:]
                filename = f"train_epoch_{index:03d}.npz"
            else:
                local_scheduled = permutation
                local_omitted = np.empty(0, dtype=np.int64)
                filename = f"{kind}_perm_{index:03d}.npz"
            local_offsets = np.asarray([0, local_scheduled.size], dtype=np.int64)
            local_path = root / "schedules" / filename
            _atomic_npz(
                local_path,
                scheduled_rows=local_scheduled,
                batch_offsets=local_offsets,
                omitted_rows=local_omitted,
            )
            schedule_entries[kind].append(
                {
                    "index": index,
                    "path": f"schedules/{filename}",
                    "sha256": sha256_file(local_path),
                    "scheduled_rows_sha256": sha256_array(local_scheduled),
                    "batch_offsets_sha256": sha256_array(local_offsets),
                    "omitted_rows_sha256": sha256_array(local_omitted),
                    "n_batches": 1,
                    "n_scheduled": int(local_scheduled.size),
                    "n_omitted": int(local_omitted.size),
                    "last_batch_size": int(local_offsets[-1] - local_offsets[-2]),
                }
            )
            if kind == "train" and index == 0:
                train_zero = (local_scheduled, local_offsets, local_omitted, local_path)
    assert train_zero is not None
    scheduled_rows, batch_offsets, omitted_rows, schedule_path = train_zero
    fake = "a" * 64
    schedule_manifest = {
        "schema_version": 1,
        "assets": {
            "corpus": sha256_file(corpus),
            "pretrain_v3_all": fake,
            "u_eval": fake,
            "split_manifest": sha256_file(split_manifest_path),
            "split_npz": sha256_file(split_path),
            "producer": fake,
        },
        "parameters": {"batch_size": 32, "epochs": 10},
        "schedules": schedule_entries,
        "split": {
            "manifest_sha256": sha256_file(split_manifest_path),
            "npz_sha256": sha256_file(split_path),
        },
    }
    schedule_manifest_path = root / "schedule_manifest.json"
    schedule_manifest_path.write_text(json.dumps(schedule_manifest, sort_keys=True), encoding="utf-8")

    upper_i, upper_j = np.triu_indices(32, k=1)
    protein = (upper_j - upper_i).astype(np.float64) / 32.0
    unifrac = (upper_i + upper_j + 1).astype(np.float64) / 64.0
    cache_arrays = {
        "scheduled_rows": scheduled_rows,
        "batch_offsets": batch_offsets,
        "pair_offsets": np.asarray([0, 496], dtype=np.int64),
        "project_codes": train_projects[:32].astype(np.int32),
        "site_codes": train_sites[:32].astype(np.int16),
        "protein_distance_upper": protein,
        "unifrac_distance_upper": unifrac,
        "protein_valid_mass": np.ones(32, dtype=np.float64),
        "protein_borrowed_mass": np.zeros(32, dtype=np.float64),
        "protein_endpoint_valid": np.ones(32, dtype=bool),
        "protein_pair_valid_upper": np.ones(496, dtype=bool),
    }
    epoch_cache = cache_root / "train_epoch_000"
    cache_path = epoch_cache / "teacher_cache.npz"
    _atomic_npz(cache_path, **cache_arrays)
    assets = {
        "corpus": sha256_file(corpus),
        "pretrain_v3_all": fake,
        "u_eval": fake,
        "protein_cost": fake,
        "valid_mask": fake,
        "coverage": fake,
        "teacher_reference": fake,
        "tree_manifest": fake,
        "genus_to_edge": fake,
        "branch_lengths": fake,
        "split_manifest": sha256_file(split_manifest_path),
        "split_npz": sha256_file(split_path),
        "schedule_manifest": sha256_file(schedule_manifest_path),
        "schedule_file": sha256_file(schedule_path),
        "runner": fake,
    }
    cache_manifest = {
        "schema_version": 1,
        "contract": {
            "assets": assets,
            "solver": {
                "protein": (
                    "scipy.optimize.linprog(method='highs'; primal_feasibility_tolerance=1e-9; "
                    "dual_feasibility_tolerance=1e-9; ipm_optimality_tolerance=1e-10); "
                    "exact balanced OT; no approximation"
                ),
                "unifrac": "normalized weighted UniFrac; exact genus-to-edge incidence; no approximation",
            },
            "teacher_validity": {
                "protein_endpoint_valid_mass_min": 0.9,
                "protein_pair_valid": "both endpoints valid",
                "unifrac_endpoint_gate": "none",
            },
            "teacher_compare_tolerance": "max(1e-12, 1e-10 * scale)",
            "pair_order": "row-major np.triu_indices(actual_batch_size, k=1)",
            "schedule": {
                "kind": "train",
                "index": 0,
                "nominal_batch_size": 32,
                "n_batches": 1,
                "n_scheduled": 32,
                "n_omitted": 1,
                "last_batch_size": 32,
                "scheduled_rows_sha256": sha256_array(scheduled_rows),
                "batch_offsets_sha256": sha256_array(batch_offsets),
                "omitted_rows_sha256": sha256_array(omitted_rows),
                "schedule_file_sha256": sha256_file(schedule_path),
            },
        },
        "output": {
            "path": "teacher_cache.npz",
            "sha256": sha256_file(cache_path),
            "array_sha256": {name: sha256_array(value) for name, value in cache_arrays.items()},
        },
    }
    cache_manifest_path = epoch_cache / "teacher_cache_manifest.json"
    cache_manifest_path.write_text(json.dumps(cache_manifest, sort_keys=True), encoding="utf-8")
    complete = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(cache_manifest_path),
        "cache_sha256": sha256_file(cache_path),
        "schedule_file_sha256": sha256_file(schedule_path),
        "schedule_scheduled_rows_sha256": sha256_array(scheduled_rows),
    }
    (epoch_cache / ".complete").write_text(json.dumps(complete, sort_keys=True), encoding="utf-8")


class RelationDataTests(unittest.TestCase):
    def test_topk_tie_break_and_retained_support_rclr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            corpus = tmp_path / "toy.h5ad"
            _write_toy_h5ad(corpus, n_rows=3)
            dataset = RelationAnnDataDataset(
                corpus,
                split_rows=np.asarray([0, 1, 2], dtype=np.int64),
                project_codes=np.asarray([0, 1, 2], dtype=np.int32),
                site_codes=np.asarray([0, 0, 1], dtype=np.int32),
                max_tokens=3,
                expected_n_vars=6,
            )

            sample = dataset.get_by_global_row(0)
            # 0.4(var2), 0.2(var4), then the 0.1 tie is won by var0.
            self.assertEqual(sample["var_indices"].tolist(), [2, 4, 0])
            self.assertEqual(sample["genus_ids"].tolist(), [4, 6, 2])
            self.assertAlmostEqual(float(sample["rclr"].mean()), 0.0, places=6)
            self.assertEqual(sample["rclr"].dtype, np.float32)

            full_indices, full_abundance = dataset.full_composition(0)
            self.assertEqual(full_indices.tolist(), [0, 1, 2, 4, 5])
            self.assertEqual(full_abundance.size, 5)
            dataset.close()

    def test_epoch_cache_is_hash_bound_and_batch_aligned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            corpus = tmp_path / "toy.h5ad"
            schedule_root = tmp_path / "schedule"
            cache_root = tmp_path / "cache"
            _write_toy_h5ad(corpus)
            _write_schedule_and_cache(schedule_root, cache_root, corpus)

            store = RelationScheduleStore(schedule_root, cache_root, corpus)
            rows, projects, sites = store.split("train")
            samples = RelationAnnDataDataset(
                corpus,
                split_rows=rows,
                project_codes=projects,
                site_codes=sites,
                max_tokens=3,
                expected_n_vars=6,
            )
            epoch = store.epoch_dataset("train", 0, samples)
            batch = epoch[0]
            self.assertEqual(batch["genus_ids"].shape, (32, 3))
            self.assertEqual(batch["padding_mask"].dtype.name, "bool")
            self.assertEqual(batch["teacher_distances"]["protein"].shape, (32, 32))
            self.assertEqual(
                batch["teacher_distances"]["unifrac"].shape, (32, 32)
            )
            self.assertTrue(
                np.allclose(np.diag(batch["teacher_distances"]["protein"]), 0.0)
            )
            self.assertEqual(batch["teacher_distances"]["protein"].dtype, np.float64)
            self.assertEqual(
                batch["teacher_distances"]["unifrac"].dtype, np.float64
            )
            self.assertEqual(batch["protein_valid_mass"].dtype, np.float64)
            self.assertEqual(batch["protein_borrowed_mass"].dtype, np.float64)
            self.assertEqual(batch["protein_endpoint_valid"].dtype, np.bool_)
            off_diagonal = ~np.eye(32, dtype=bool)
            self.assertTrue(batch["teacher_validity"]["protein"][off_diagonal].all())
            self.assertFalse(batch["teacher_validity"]["protein"].diagonal().any())
            self.assertEqual(batch["row_ids"].tolist(), list(range(32)))

            cache_path = cache_root / "train_epoch_000/teacher_cache.npz"
            with cache_path.open("ab") as handle:
                handle.write(b"drift")
            with self.assertRaisesRegex(RuntimeError, "cache_sha256"):
                store.epoch_dataset("train", 0, samples)
            samples.close()

    def test_schedule_cannot_escape_strict_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            corpus = tmp_path / "toy.h5ad"
            schedule_root = tmp_path / "schedule"
            cache_root = tmp_path / "cache"
            _write_toy_h5ad(corpus)
            _write_schedule_and_cache(schedule_root, cache_root, corpus)

            schedule_path = schedule_root / "schedules/train_epoch_000.npz"
            with np.load(schedule_path) as archive:
                rows = np.asarray(archive["scheduled_rows"])
            rows = rows.copy()
            rows[0] = 10_000
            _atomic_npz(
                schedule_path,
                scheduled_rows=rows,
                batch_offsets=np.asarray([0, 32], dtype=np.int64),
                omitted_rows=np.asarray([32], dtype=np.int64),
            )
            with self.assertRaisesRegex(RuntimeError, "schedule.*sha256"):
                RelationScheduleStore(schedule_root, cache_root, corpus)


if __name__ == "__main__":
    unittest.main()
