from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from micoformer.relation_pretraining import smoke
from micoformer.relation_pretraining.data import sha256_file
from micoformer.relation_pretraining.workflow import (
    RelationRunConfig,
    SMOKE_REQUIRED_PASS_FIELDS,
    run_relation_pretraining,
    validate_smoke_launch_authorization,
)


def _matched_arm_result() -> dict[str, object]:
    steps = []
    optimizer_steps = 0
    for index in range(smoke.MATCHED_REAL_STEPS):
        optimizer_steps += 1
        steps.append(
            {
                "scheduled_index": index,
                "schedule_batch_index": index,
                "row_ids_sha256": f"{index:064x}",
                "batch_size": 32,
                "dynamic_sequence_length": 16 + index,
                "has_relation_update": True,
                "optimizer_step_count": optimizer_steps,
                "scheduler_step_count": optimizer_steps,
                "lr_before": [1e-5 + index * 1e-7, 1e-5 + index * 1e-7],
                "lr_after": [1.01e-5 + index * 1e-7, 1.01e-5 + index * 1e-7],
            }
        )
    return {
        "optimizer_steps": optimizer_steps,
        "skipped_steps": 0,
        "steps": steps,
    }


def _authorization_step(index: int, *, sequence_length: int = 96) -> dict[str, object]:
    count = index + 1
    return {
        "scheduled_index": index,
        "schedule_batch_index": index,
        "row_ids_sha256": f"{count:064x}",
        "batch_size": 32,
        "dynamic_sequence_length": sequence_length,
        "has_relation_update": True,
        "optimizer_step_count": count,
        "scheduler_step_count": count,
        "lr_before": [1e-5, 1e-5],
        "lr_after": [1.1e-5, 1.1e-5],
        "grad_scaler_scale_before": 65536.0,
        "grad_scaler_scale_after": 65536.0,
        "loss": 0.25,
        "counters": {},
        "runtime_counters": {
            "scheduled_batches": count,
            "optimizer_steps": count,
            "skipped_batches": 0,
            "scheduler_steps": count,
        },
    }


def _authorization_manifest(
    smoke_dir: Path,
    *,
    h5ad_path: Path,
    schedule_root: Path,
    cache_root: Path,
) -> dict[str, object]:
    synthetic_step = _authorization_step(0, sequence_length=512)
    real_steps = [_authorization_step(index) for index in range(50)]
    peaks = {name: 1024**3 for name in smoke.ARM_ORDER}
    arms = {}
    for arm_name in smoke.ARM_ORDER:
        arms[arm_name] = {
            "synthetic_worst_case": {
                "shape": [32, 512],
                "step": copy.deepcopy(synthetic_step),
                "peak_allocated_bytes": 1024,
                "peak_reserved_bytes": 2048,
                "peak_reserved_gib": 2048 / 1024**3,
                "peak_reserved_limit_bytes": smoke.PEAK_RESERVED_LIMIT_BYTES,
                "peak_reserved_strictly_below_limit": True,
                "optimizer_state_finite": True,
            },
            "real_first_50": {
                "initialization": {
                    "shared_initialization_sha256": "1" * 64,
                    "full_initialization_sha256": (
                        "2" * 64 if arm_name != "pma_skip" else "3" * 64
                    ),
                },
                "scheduled_steps": 50,
                "optimizer_steps": 50,
                "scheduler_steps": 50,
                "skipped_steps": 0,
                "peak_allocated_bytes": 1024,
                "peak_reserved_bytes": 2048,
                "peak_reserved_gib": 2048 / 1024**3,
                "peak_reserved_limit_bytes": smoke.PEAK_RESERVED_LIMIT_BYTES,
                "peak_reserved_strictly_below_limit": True,
                "optimizer_state_finite": True,
                "steps": copy.deepcopy(real_steps),
            },
            "interrupted_resume": {
                "passed": True,
                "comparison": "bitwise/exact; no tolerance relaxation",
                "component_sha256": {
                    name: f"{index + 4:x}" * 64
                    for index, name in enumerate(
                        (
                            "second_step_record",
                            "model_state",
                            "optimizer_state",
                            "scheduler_state",
                            "grad_scaler_state",
                            "runtime_counters",
                            "consumed_teacher_caches",
                            "rng_after_second_step",
                        )
                    )
                },
                "second_step_loss": 0.25,
                "runtime_counters": {
                    "scheduled_batches": 2,
                    "optimizer_steps": 2,
                    "skipped_batches": 0,
                    "scheduler_steps": 2,
                },
                "checkpoint_boundary": "production hooks",
                "arm_name": arm_name,
                "first_step": copy.deepcopy(synthetic_step),
                "temporary_checkpoint_sha256": "e" * 64,
                "temporary_checkpoint_removed": True,
                "peak_allocated_bytes": 1024,
                "peak_reserved_bytes": 2048,
                "peak_reserved_gib": 2048 / 1024**3,
                "peak_reserved_limit_bytes": smoke.PEAK_RESERVED_LIMIT_BYTES,
                "peak_reserved_strictly_below_limit": True,
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": False,
                "cublas_workspace_config": ":4096:8",
            },
            "single_process_peak_host_rss_bytes": peaks[arm_name],
        }
    return {
        "schema_version": 1,
        "gate": "relation_b32_real_cuda_launch_gate",
        "status": "passed",
        "started_at": "2026-07-17T00:00:00+00:00",
        "completed_at": "2026-07-17T00:01:00+00:00",
        "contract": {
            "arms": list(smoke.ARM_ORDER),
            "synthetic_shape": [32, 512],
            "real_schedule": "train_epoch_000 fixed first 50 batches",
            "physical_batch_size": 32,
            "peak_reserved_strict_limit_bytes": smoke.PEAK_RESERVED_LIMIT_BYTES,
            "aggregate_host_rss_strict_limit_bytes": (
                smoke.AGGREGATE_HOST_RSS_LIMIT_BYTES
            ),
            "cuda_required": True,
            "cpu_fallback_allowed": False,
            "b16_implemented_or_triggered": False,
            "production_checkpoint_or_embedding_write": False,
        },
        "inputs": {
            "h5ad_path": str(h5ad_path.resolve()),
            "schedule_root": str(schedule_root.resolve()),
            "cache_root": str(cache_root.resolve()),
            "output_dir": str(smoke_dir.resolve()),
        },
        "source_assets": smoke._source_hashes(),
        "cuda": {"cpu_fallback_allowed": False},
        "determinism": {
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": False,
            "cublas_workspace_config": ":4096:8",
            "production_reference": "Lightning Trainer(deterministic=True)",
        },
        "matched_schedule": {
            "schedule_kind": "train",
            "schedule_index": 0,
            "steps": 50,
            "batch_row_id_sha256": [f"{index + 1:064x}" for index in range(50)],
        },
        "arms": arms,
        "matched_trajectory_gate": {
            "passed": True,
            "reference_arm": "main_skip",
            "scheduled_steps": 50,
            "optimizer_steps": 50,
            "skipped_steps": 0,
            "fields_checked": [],
        },
        "host_rss_gate": {
            "per_single_arm_peak_bytes": peaks,
            "conservative_aggregate_bytes": sum(peaks.values()),
            "conservative_aggregate_gib": sum(peaks.values()) / 1024**3,
            "aggregate_limit_bytes": smoke.AGGREGATE_HOST_RSS_LIMIT_BYTES,
            "aggregate_strictly_below_limit": True,
            "method": "test fixture",
        },
        "result": {
            **{name: True for name in SMOKE_REQUIRED_PASS_FIELDS},
            "production_training_authorized_by_this_gate": True,
            "b16_implemented_or_triggered": False,
        },
    }


def _publish_authorization_fixture(root: Path) -> tuple[dict[str, Path], dict[str, object]]:
    h5ad_path = root / "corpus.h5ad"
    h5ad_path.touch()
    schedule_root = root / "schedule"
    cache_root = root / "cache"
    smoke_dir = root / "smoke"
    schedule_root.mkdir()
    cache_root.mkdir()
    smoke_dir.mkdir()
    paths = {
        "h5ad_path": h5ad_path,
        "schedule_root": schedule_root,
        "cache_root": cache_root,
        "smoke_dir": smoke_dir,
    }
    manifest = _authorization_manifest(smoke_dir, **{k: paths[k] for k in (
        "h5ad_path", "schedule_root", "cache_root"
    )})
    smoke.publish_smoke_manifest(smoke_dir, manifest, passed=True)
    return paths, manifest


class RelationSmokeLogicTests(unittest.TestCase):
    def test_synthetic_batch_is_exact_b32_l512_and_float64_teachers(self) -> None:
        batch = smoke.build_synthetic_b32_l512_batch()
        self.assertEqual(batch["genus_ids"].shape, (32, 512))
        self.assertEqual(batch["genus_ids"].dtype, np.int64)
        self.assertEqual(batch["rclr"].shape, (32, 512))
        self.assertEqual(batch["rclr"].dtype, np.float32)
        self.assertTrue(np.allclose(batch["rclr"].mean(axis=1), 0.0, atol=1e-6))
        self.assertFalse(batch["padding_mask"].any())
        for value in batch["teacher_distances"].values():
            self.assertEqual(value.shape, (32, 32))
            self.assertEqual(value.dtype, np.float64)
            self.assertTrue(np.allclose(value, value.T))
            self.assertTrue(np.allclose(np.diag(value), 0.0))

    def test_cuda_requirement_fails_closed_without_cuda(self) -> None:
        with mock.patch.object(smoke.torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(smoke.RelationSmokeError, "forbids CPU fallback"):
                smoke.require_real_cuda(0)

    def test_matched_trajectory_accepts_identical_three_arms(self) -> None:
        base = _matched_arm_result()
        arms = {name: copy.deepcopy(base) for name in smoke.ARM_ORDER}
        observed = smoke.validate_matched_trajectories(arms)
        self.assertTrue(observed["passed"])
        self.assertEqual(observed["scheduled_steps"], 50)
        self.assertEqual(observed["optimizer_steps"], 50)

    def test_matched_trajectory_rejects_update_and_lr_drift(self) -> None:
        base = _matched_arm_result()
        arms = {name: copy.deepcopy(base) for name in smoke.ARM_ORDER}
        drifted = arms["main_radius"]["steps"][12]
        drifted["has_relation_update"] = False
        drifted["optimizer_step_count"] = 12
        drifted["lr_after"][0] *= 0.5
        with self.assertRaisesRegex(smoke.MatchedTrajectoryError, "not matched"):
            smoke.validate_matched_trajectories(arms)

    def test_aggregate_rss_gate_is_strict_and_conservative(self) -> None:
        peaks = {name: 2 * 1024**3 for name in smoke.ARM_ORDER}
        observed = smoke.aggregate_host_rss_gate(peaks)
        self.assertEqual(observed["conservative_aggregate_bytes"], 6 * 1024**3)
        self.assertTrue(observed["aggregate_strictly_below_limit"])

        exact_limit = {
            "main_skip": 15 * 1024**3,
            "main_radius": 15 * 1024**3,
            "pma_skip": 15 * 1024**3,
        }
        with self.assertRaisesRegex(smoke.RelationSmokeError, "not below 45 GiB"):
            smoke.aggregate_host_rss_gate(exact_limit)

    def test_counter_conservation_logic_fails_closed(self) -> None:
        values = {
            "train/mining/all/teacher_anchor_total": 64.0,
            "train/mining/all/status_total": 64.0,
            "train/mining/all/valid_relation": 60.0,
            "train/mining/all/skipped_relation": 4.0,
            "train/objective/all/teacher_anchor_total": 64.0,
            "train/objective/all/valid_anchor": 60.0,
            "train/objective/all/skipped_anchor": 4.0,
            "train/objective/all/active_hinge": 40.0,
            "train/objective/all/inactive_hinge": 20.0,
        }
        observed = smoke._validate_and_extract_counters(values)
        self.assertEqual(observed["objective_valid_anchor"], 60)
        values["train/objective/all/inactive_hinge"] = 19.0
        with self.assertRaisesRegex(smoke.RelationSmokeError, "active/inactive"):
            smoke._validate_and_extract_counters(values)

    def test_atomic_manifest_has_pass_only_complete_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            passed = {
                "schema_version": 1,
                "status": "passed",
                "completed_at": "2026-07-17T00:00:00+00:00",
                "result": {
                    **{name: True for name in SMOKE_REQUIRED_PASS_FIELDS},
                },
            }
            manifest_path = smoke.publish_smoke_manifest(output, passed, passed=True)
            complete_path = output / ".complete"
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(complete_path.is_file())
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            self.assertEqual(complete["manifest_sha256"], sha256_file(manifest_path))
            self.assertTrue(complete["resume_determinism_gate_passed"])

            failed = {
                "schema_version": 1,
                "status": "failed",
                "completed_at": "2026-07-17T00:01:00+00:00",
            }
            smoke.publish_smoke_manifest(output, failed, passed=False)
            self.assertFalse(complete_path.exists())

    def test_pass_publication_requires_resume_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            missing_resume = {
                "schema_version": 1,
                "status": "passed",
                "completed_at": "2026-07-17T00:00:00+00:00",
                "result": {
                    **{
                        name: True
                        for name in SMOKE_REQUIRED_PASS_FIELDS
                        if name != "resume_determinism_gate_passed"
                    },
                },
            }
            with self.assertRaisesRegex(ValueError, "every B32/50-step/three-arm"):
                smoke.publish_smoke_manifest(output, missing_resume, passed=True)
            self.assertFalse((output / ".complete").exists())

    def test_resume_snapshot_comparison_is_exact_and_hash_bound(self) -> None:
        snapshot = {
            "second_step_record": {
                "loss": 0.125,
                "runtime_counters": {
                    "scheduled_batches": 2,
                    "optimizer_steps": 2,
                    "skipped_batches": 0,
                    "scheduler_steps": 2,
                },
            },
            "model_state": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer_state": {
                "state": {0: {"step": torch.tensor(2.0)}},
                "param_groups": [{"lr": 1e-5, "params": [0]}],
            },
            "scheduler_state": {"last_epoch": 2},
            "grad_scaler_state": {"scale": 65536.0},
            "runtime_counters": {
                "scheduled_batches": 2,
                "optimizer_steps": 2,
                "skipped_batches": 0,
                "scheduler_steps": 2,
            },
            "consumed_teacher_caches": [
                {
                    "cache_manifest_sha256": "a" * 64,
                    "cache_sha256": "b" * 64,
                    "schedule_file_sha256": "c" * 64,
                }
            ],
            "rng_after_second_step": {
                "torch_cpu": torch.arange(8, dtype=torch.uint8),
                "numpy": np.arange(4, dtype=np.uint32),
            },
        }
        observed = smoke._compare_resume_snapshots(snapshot, copy.deepcopy(snapshot))
        self.assertTrue(observed["passed"])
        self.assertEqual(
            set(observed["component_sha256"]),
            {
                "second_step_record",
                "model_state",
                "optimizer_state",
                "scheduler_state",
                "grad_scaler_state",
                "runtime_counters",
                "consumed_teacher_caches",
                "rng_after_second_step",
            },
        )

        drifted = copy.deepcopy(snapshot)
        drifted["model_state"]["weight"][1] = 2.0001
        with self.assertRaisesRegex(smoke.ResumeDeterminismError, "model_state.weight"):
            smoke._compare_resume_snapshots(snapshot, drifted)

    def test_valid_smoke_authorization_is_hash_and_source_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, _ = _publish_authorization_fixture(Path(directory))
            observed = validate_smoke_launch_authorization(
                paths["smoke_dir"],
                h5ad_path=paths["h5ad_path"],
                schedule_root=paths["schedule_root"],
                cache_root=paths["cache_root"],
            )
            self.assertEqual(observed["directory"], str(paths["smoke_dir"].resolve()))
            self.assertEqual(
                observed["manifest"]["sha256"],
                sha256_file(paths["smoke_dir"] / "smoke_manifest.json"),
            )
            self.assertEqual(
                observed["complete"]["sha256"],
                sha256_file(paths["smoke_dir"] / ".complete"),
            )

    def test_train_bypass_without_smoke_fails_before_output_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_root = root / "production"
            config = RelationRunConfig(
                h5ad_path=root / "corpus.h5ad",
                schedule_root=root / "schedule",
                cache_root=root / "cache",
                output_root=output_root,
                arm_name="main_skip",
                disease_rows_path=root / "disease.npy",
                smoke_dir=root / "missing_smoke",
            )
            with self.assertRaisesRegex(RuntimeError, "smoke authorization directory is missing"):
                run_relation_pretraining(config)
            self.assertFalse(output_root.exists())

    def test_smoke_manifest_tamper_breaks_sentinel_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, manifest = _publish_authorization_fixture(Path(directory))
            manifest["status"] = "failed"
            (paths["smoke_dir"] / "smoke_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "sentinel manifest hash mismatch"):
                validate_smoke_launch_authorization(
                    paths["smoke_dir"],
                    h5ad_path=paths["h5ad_path"],
                    schedule_root=paths["schedule_root"],
                    cache_root=paths["cache_root"],
                )

    def test_smoke_input_path_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths, _ = _publish_authorization_fixture(root)
            different_cache = root / "cache_highs1e9"
            different_cache.mkdir()
            with self.assertRaisesRegex(RuntimeError, "input path drifted for cache_root"):
                validate_smoke_launch_authorization(
                    paths["smoke_dir"],
                    h5ad_path=paths["h5ad_path"],
                    schedule_root=paths["schedule_root"],
                    cache_root=different_cache,
                )

    def test_republished_smoke_with_source_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, manifest = _publish_authorization_fixture(Path(directory))
            workflow_source = "micoformer/relation_pretraining/workflow.py"
            manifest["source_assets"][workflow_source]["sha256"] = "0" * 64
            smoke.publish_smoke_manifest(paths["smoke_dir"], manifest, passed=True)
            with self.assertRaisesRegex(RuntimeError, "source SHA256 drifted.*workflow.py"):
                validate_smoke_launch_authorization(
                    paths["smoke_dir"],
                    h5ad_path=paths["h5ad_path"],
                    schedule_root=paths["schedule_root"],
                    cache_root=paths["cache_root"],
                )

    def test_cli_exposes_only_explicit_smoke_paths_and_device(self) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts/2.train_relation.py"
        spec = importlib.util.spec_from_file_location("relation_train_cli_for_test", script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = module.build_parser().parse_args(
            [
                "smoke",
                "--h5ad",
                "/tmp/corpus.h5ad",
                "--schedule-root",
                "/tmp/schedule",
                "--cache-root",
                "/tmp/cache",
                "--output-dir",
                "/tmp/smoke",
                "--device-index",
                "2",
            ]
        )
        self.assertEqual(args.command, "smoke")
        self.assertEqual(args.device_index, 2)
        self.assertFalse(hasattr(args, "disease_rows"))
        self.assertFalse(hasattr(args, "output_root"))

    def test_train_cli_requires_explicit_smoke_directory(self) -> None:
        script = Path(__file__).resolve().parents[2] / "scripts/2.train_relation.py"
        spec = importlib.util.spec_from_file_location(
            "relation_train_cli_smoke_required_test", script
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base = [
            "train",
            "--h5ad",
            "/tmp/corpus.h5ad",
            "--schedule-root",
            "/tmp/schedule",
            "--cache-root",
            "/tmp/cache",
            "--output-root",
            "/tmp/output",
            "--disease-rows",
            "/tmp/disease.npy",
            "--arm",
            "main_skip",
        ]
        with self.assertRaises(SystemExit):
            module.build_parser().parse_args(base)
        args = module.build_parser().parse_args(
            [*base, "--smoke-dir", "/tmp/cuda_smoke_b32_highs1e9"]
        )
        self.assertEqual(args.smoke_dir, Path("/tmp/cuda_smoke_b32_highs1e9"))


if __name__ == "__main__":
    unittest.main()
