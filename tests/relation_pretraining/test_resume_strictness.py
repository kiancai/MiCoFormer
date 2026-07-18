from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import torch

from micoformer.relation_pretraining.data import sha256_file
from micoformer.relation_pretraining.model import RelationModelConfig
from micoformer.relation_pretraining.module import (
    RELATION_SOURCE_PATHS,
    RelationPretrainingModule,
    build_relation_source_manifest,
    load_relation_checkpoint,
    restore_relation_rng_state,
)
from micoformer.relation_pretraining.workflow import (
    bind_immutable_epoch_endpoint,
    validate_matched_arm_completions,
)


def _binding() -> dict[str, object]:
    return {
        "schema_version": 1,
        "corpus_sha256": "a" * 64,
        "schedule_manifest_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "split_npz_sha256": "d" * 64,
        "physical_batch_size": 32,
        "schedule_parameters": {"batch_size": 32},
        "schedule_assets": {"corpus": "a" * 64},
        "binding_sha256": "e" * 64,
    }


def _config(dropout: float = 0.35) -> RelationModelConfig:
    return RelationModelConfig(
        vocab_size=10,
        d_model=16,
        rclr_hidden_dim=8,
        num_layers=1,
        encoder_heads=4,
        encoder_ffn_dim=32,
        decoder_heads=4,
        decoder_ffn_dim=32,
        dropout=dropout,
        max_seq_len=8,
        decoder_kind="main",
    )


def _seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _new_module(dropout: float = 0.35) -> RelationPretrainingModule:
    _seed_all(42)
    return RelationPretrainingModule(
        arm_name="main_skip",
        model_config=_config(dropout),
        data_binding=_binding(),
    ).train()


def _optimizer_scheduler(module: RelationPretrainingModule):
    configured = module.configure_optimizers()
    return configured["optimizer"], configured["lr_scheduler"]["scheduler"]


def _one_dropout_step(module, optimizer, scheduler) -> dict[str, Any]:
    python_draw = random.random()
    numpy_draw = float(np.random.random())
    genus_ids = torch.tensor(
        [[2, 3, 0], [3, 4, 5], [2, 6, 7], [4, 7, 0]], dtype=torch.long
    )
    rclr = torch.tensor(
        [[-0.2, 0.2, 0.0], [-0.3, 0.1, 0.2], [0.4, -0.1, -0.3], [0.2, -0.2, 0.0]],
        dtype=torch.float32,
    )
    optimizer.zero_grad(set_to_none=True)
    z = module.model(genus_ids, rclr, genus_ids.eq(0)).z
    target = torch.linspace(-0.5, 0.5, z.numel(), dtype=z.dtype).reshape_as(z)
    loss = (z - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    module._relation_scheduled_batch_count.add_(1)
    module._relation_optimizer_step_count.add_(1)
    module._relation_scheduler_step_count.add_(1)
    module._assert_runtime_counter_invariants()
    return {
        "python": python_draw,
        "numpy": numpy_draw,
        "z": z.detach().clone(),
        "loss": loss.detach().clone(),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


def _assert_nested_equal(test: unittest.TestCase, left: Any, right: Any) -> None:
    test.assertEqual(type(left), type(right))
    if isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(test, left_item, right_item)
    elif isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        test.assertTrue(np.array_equal(left, right))
    else:
        test.assertEqual(left, right)


class RelationResumeStrictnessTests(unittest.TestCase):
    def test_source_manifest_exactly_binds_all_training_sources(self) -> None:
        manifest = build_relation_source_manifest()
        self.assertEqual(set(manifest["files"]), set(RELATION_SOURCE_PATHS))
        self.assertEqual(len(manifest["manifest_sha256"]), 64)
        module = _new_module(dropout=0.0)
        self.assertEqual(module.relation_contract["source_manifest"], manifest)
        self.assertFalse(
            module.relation_contract["resume_determinism"]["amp_scaler"][
                "cpu_unit_coverage"
            ]
        )
        self.assertIn(
            "CUDA",
            module.relation_contract["resume_determinism"]["amp_scaler"][
                "required_verification"
            ],
        )

    def test_checkpoint_rejects_tampered_source_manifest(self) -> None:
        module = _new_module(dropout=0.0)
        checkpoint = {"state_dict": copy.deepcopy(module.state_dict()), "epoch": 0, "global_step": 0}
        with mock.patch("torch.cuda.is_available", return_value=False):
            module.on_save_checkpoint(checkpoint)
        checkpoint["relation_contract"]["source_manifest"]["manifest_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "source manifest"):
                load_relation_checkpoint(path)

    def test_interrupted_resume_next_step_is_bitwise_identical(self) -> None:
        uninterrupted = _new_module(dropout=0.35)
        optimizer, scheduler = _optimizer_scheduler(uninterrupted)
        _one_dropout_step(uninterrupted, optimizer, scheduler)
        checkpoint = {
            "state_dict": copy.deepcopy(uninterrupted.state_dict()),
            "epoch": 0,
            "global_step": 1,
            "optimizer_states": [copy.deepcopy(optimizer.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }
        with mock.patch("torch.cuda.is_available", return_value=False):
            uninterrupted.on_save_checkpoint(checkpoint)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "interrupt.ckpt"
            torch.save(checkpoint, path)
            persisted = torch.load(path, map_location="cpu", weights_only=False)

        expected = _one_dropout_step(uninterrupted, optimizer, scheduler)
        expected_model = copy.deepcopy(uninterrupted.state_dict())
        expected_optimizer = copy.deepcopy(optimizer.state_dict())
        expected_scheduler = copy.deepcopy(scheduler.state_dict())

        resumed = _new_module(dropout=0.35)
        resumed_optimizer, resumed_scheduler = _optimizer_scheduler(resumed)
        # Match Lightning's order: on_load hook first, then model/optimizer/
        # scheduler state restoration, and RNG restoration at on_train_start.
        resumed.on_load_checkpoint(persisted)
        resumed.load_state_dict(persisted["state_dict"], strict=True)
        resumed_optimizer.load_state_dict(persisted["optimizer_states"][0])
        resumed_scheduler.load_state_dict(persisted["lr_schedulers"][0])
        resumed.on_train_start()
        observed = _one_dropout_step(resumed, resumed_optimizer, resumed_scheduler)

        self.assertEqual(expected["python"], observed["python"])
        self.assertEqual(expected["numpy"], observed["numpy"])
        self.assertTrue(torch.equal(expected["z"], observed["z"]))
        self.assertTrue(torch.equal(expected["loss"], observed["loss"]))
        self.assertEqual(expected["lr"], observed["lr"])
        _assert_nested_equal(self, expected_model, resumed.state_dict())
        _assert_nested_equal(self, expected_optimizer, resumed_optimizer.state_dict())
        _assert_nested_equal(self, expected_scheduler, resumed_scheduler.state_dict())
        self.assertEqual(
            resumed.relation_runtime_counts,
            {
                "scheduled_batches": 2,
                "optimizer_steps": 2,
                "skipped_batches": 0,
                "scheduler_steps": 2,
            },
        )

    def test_true_resume_rejects_cuda_topology_change(self) -> None:
        module = _new_module(dropout=0.0)
        checkpoint = {"state_dict": copy.deepcopy(module.state_dict())}
        with mock.patch("torch.cuda.is_available", return_value=False):
            module.on_save_checkpoint(checkpoint)
        with mock.patch("torch.cuda.is_available", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "CUDA availability changed"):
                restore_relation_rng_state(checkpoint["relation_rng_state"])

    def test_immutable_endpoint_ignores_last_and_records_runtime_state(self) -> None:
        module = _new_module(dropout=0.0)
        optimizer, scheduler = _optimizer_scheduler(module)
        _one_dropout_step(module, optimizer, scheduler)
        checkpoint = {
            "state_dict": copy.deepcopy(module.state_dict()),
            "epoch": 2,
            "global_step": 1,
            "optimizer_states": [copy.deepcopy(optimizer.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }
        with mock.patch("torch.cuda.is_available", return_value=False):
            module.on_save_checkpoint(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            immutable = checkpoint_dir / "epoch02-step1.ckpt"
            torch.save(checkpoint, immutable)
            mutable = copy.deepcopy(checkpoint)
            mutable["epoch"] = 99
            torch.save(mutable, checkpoint_dir / "last.ckpt")
            run_manifest = root / "run_manifest.json"
            run_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "arm_name": "main_skip",
                        "source_manifest": module.relation_contract["source_manifest"],
                        "relation_contract": module.relation_contract,
                    }
                )
            )
            endpoint_manifest = root / "endpoints/arm_completion.json"
            selected = bind_immutable_epoch_endpoint(
                checkpoint_dir=checkpoint_dir,
                endpoint_manifest_path=endpoint_manifest,
                run_manifest_path=run_manifest,
                expected_relation_contract=module.relation_contract,
                expected_data_binding=_binding(),
                arm_name="main_skip",
                endpoint_name="arm_completion",
                expected_epoch=2,
                require_amp_scaler=False,
            )
            self.assertEqual(selected, immutable.resolve())
            record = json.loads(endpoint_manifest.read_text())
            self.assertFalse(record["checkpoint"]["last_ckpt_used"])
            self.assertEqual(record["checkpoint"]["sha256"], sha256_file(immutable))
            self.assertEqual(record["runtime_counters"]["optimizer_steps"], 1)
            self.assertEqual(record["runtime_counters"]["scheduler_steps"], 1)
            self.assertIsInstance(record["runtime_counters"]["current_lr"], float)

            run_manifest_payload = json.loads(run_manifest.read_text())
            run_manifest_payload["arm_name"] = "pma_skip"
            run_manifest.write_text(json.dumps(run_manifest_payload))
            endpoint_manifest.unlink()
            with self.assertRaisesRegex(RuntimeError, "exact arm/source/run contract"):
                bind_immutable_epoch_endpoint(
                    checkpoint_dir=checkpoint_dir,
                    endpoint_manifest_path=endpoint_manifest,
                    run_manifest_path=run_manifest,
                    expected_relation_contract=module.relation_contract,
                    expected_data_binding=_binding(),
                    arm_name="main_skip",
                    endpoint_name="arm_completion",
                    expected_epoch=2,
                    require_amp_scaler=False,
                )

    def test_immutable_endpoint_rejects_versioned_or_ambiguous_epoch_files(self) -> None:
        module = _new_module(dropout=0.0)
        optimizer, scheduler = _optimizer_scheduler(module)
        _one_dropout_step(module, optimizer, scheduler)
        checkpoint = {
            "state_dict": copy.deepcopy(module.state_dict()),
            "epoch": 2,
            "global_step": 1,
            "optimizer_states": [copy.deepcopy(optimizer.state_dict())],
            "lr_schedulers": [copy.deepcopy(scheduler.state_dict())],
        }
        with mock.patch("torch.cuda.is_available", return_value=False):
            module.on_save_checkpoint(checkpoint)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            torch.save(checkpoint, checkpoint_dir / "epoch02-step1.ckpt")
            torch.save(checkpoint, checkpoint_dir / "epoch02-step1-v1.ckpt")
            run_manifest = root / "run_manifest.json"
            run_manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "arm_name": "main_skip",
                        "source_manifest": module.relation_contract["source_manifest"],
                        "relation_contract": module.relation_contract,
                    }
                )
            )
            with self.assertRaisesRegex(RuntimeError, "unexpected non-last checkpoint"):
                bind_immutable_epoch_endpoint(
                    checkpoint_dir=checkpoint_dir,
                    endpoint_manifest_path=root / "endpoint.json",
                    run_manifest_path=run_manifest,
                    expected_relation_contract=module.relation_contract,
                    expected_data_binding=_binding(),
                    arm_name="main_skip",
                    endpoint_name="arm_completion",
                    expected_epoch=2,
                    require_amp_scaler=False,
                )

    def test_matched_arm_completion_gate_rejects_counter_or_lr_drift(self) -> None:
        runtime = {
            "scheduled_batches": 10,
            "optimizer_steps": 8,
            "skipped_batches": 2,
            "scheduler_steps": 8,
            "current_lr": 1.5e-4,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm, (architecture, no_next_mode) in {
                "main_skip": ("main", "skip"),
                "main_radius": ("main", "closest_radius_inside"),
                "pma_skip": ("pma", "skip"),
            }.items():
                path = root / arm / "endpoints/arm_completion.json"
                path.parent.mkdir(parents=True)
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "endpoint_kind": "frozen_relation_arm_completion",
                            "arm_name": arm,
                            "architecture": architecture,
                            "no_next_mode": no_next_mode,
                            "physical_batch_size": 32,
                            "runtime_counters": runtime,
                            "checkpoint": {"path": f"/{arm}.ckpt", "sha256": arm},
                        }
                    )
                )
            gate = validate_matched_arm_completions(root)
            self.assertIsNotNone(gate)
            assert gate is not None
            self.assertEqual(json.loads(gate.read_text())["runtime_counters"], runtime)

            pma_path = root / "pma_skip/endpoints/arm_completion.json"
            pma = json.loads(pma_path.read_text())
            pma["runtime_counters"] = {**runtime, "current_lr": 1.4e-4}
            pma_path.write_text(json.dumps(pma))
            with self.assertRaisesRegex(RuntimeError, "runtime counters/LR diverged"):
                validate_matched_arm_completions(root)


if __name__ == "__main__":
    unittest.main()
