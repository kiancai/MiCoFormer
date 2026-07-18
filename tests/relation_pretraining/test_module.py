from __future__ import annotations

import unittest
from unittest import mock

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from micoformer.relation_pretraining.model import RelationModelConfig
from micoformer.relation_pretraining.module import RelationPretrainingModule
from micoformer.relation_pretraining.workflow import build_relation_module


def _small_config(decoder_kind: str = "main") -> RelationModelConfig:
    return RelationModelConfig(
        vocab_size=10,
        d_model=16,
        rclr_hidden_dim=8,
        num_layers=1,
        encoder_heads=4,
        encoder_ffn_dim=32,
        decoder_heads=4,
        decoder_ffn_dim=32,
        dropout=0.0,
        max_seq_len=8,
        decoder_kind=decoder_kind,
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


def _batch() -> dict[str, object]:
    teacher = torch.tensor(
        [
            [0.0, 0.1, 0.6, 0.9],
            [0.1, 0.0, 0.8, 0.5],
            [0.6, 0.8, 0.0, 0.2],
            [0.9, 0.5, 0.2, 0.0],
        ],
        dtype=torch.float64,
    )
    valid = ~torch.eye(4, dtype=torch.bool)
    return {
        "genus_ids": torch.tensor(
            [[2, 3, 0], [3, 4, 5], [2, 6, 7], [4, 7, 0]], dtype=torch.long
        ),
        "rclr": torch.tensor(
            [[-0.2, 0.2, 0.0], [-0.3, 0.1, 0.2], [0.4, -0.1, -0.3], [0.2, -0.2, 0.0]],
            dtype=torch.float32,
        ),
        "padding_mask": torch.tensor(
            [[False, False, True], [False, False, False], [False, False, False], [False, False, True]]
        ),
        "row_ids": torch.tensor([10, 20, 30, 40], dtype=torch.long),
        "project_ids": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "site_ids": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "teacher_distances": {
            "protein": teacher,
            "unifrac": teacher.flip((0, 1)).clone(),
        },
        "teacher_validity": {
            "protein": valid,
            "unifrac": valid,
        },
        "protein_valid_mass": torch.ones(4, dtype=torch.float64),
        "protein_borrowed_mass": torch.zeros(4, dtype=torch.float64),
        "cache_manifest_sha256": "1" * 64,
        "cache_sha256": "2" * 64,
        "schedule_file_sha256": "3" * 64,
        "schedule_kind": "train",
        "schedule_index": torch.tensor(0),
        "schedule_batch_index": torch.tensor(0),
    }


class RelationModuleTests(unittest.TestCase):
    def test_step_keeps_teacher_float64_and_student_float32(self) -> None:
        module = RelationPretrainingModule(
            arm_name="main_skip",
            model_config=_small_config(),
            data_binding=_binding(),
        )
        module.log = mock.Mock()  # type: ignore[method-assign]

        from micoformer.relation_pretraining import module as module_namespace

        original = module_namespace.mine_relations

        def checked_mining(z, teacher_distances, *args, **kwargs):
            self.assertEqual(z.dtype, torch.float32)
            for distance in teacher_distances.values():
                self.assertEqual(distance.dtype, torch.float64)
            return original(z, teacher_distances, *args, **kwargs)

        with mock.patch.object(module_namespace, "mine_relations", side_effect=checked_mining):
            result = module._shared_step(_batch(), "train")
            loss = result.loss
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in module.model.parameters()))

    def test_singleton_eval_batch_is_explicitly_skipped(self) -> None:
        module = RelationPretrainingModule(
            arm_name="main_skip",
            model_config=_small_config(),
            data_binding=_binding(),
        )
        module.log = mock.Mock()  # type: ignore[method-assign]
        batch = _batch()
        for key in ("genus_ids", "rclr", "padding_mask", "row_ids", "project_ids", "site_ids"):
            batch[key] = batch[key][:1]
        for key in ("protein_valid_mass", "protein_borrowed_mass"):
            batch[key] = batch[key][:1]
        for mapping_key in ("teacher_distances", "teacher_validity"):
            batch[mapping_key] = {
                name: value[:1, :1] for name, value in batch[mapping_key].items()
            }
        loss = module.validation_step(batch, 0)
        self.assertEqual(float(loss), 0.0)
        logged_names = {call.args[0] for call in module.log.call_args_list}
        self.assertIn("val/health/singleton_batch", logged_names)

    def test_none_present_skips_backward_optimizer_and_scheduler(self) -> None:
        module = RelationPretrainingModule(
            arm_name="main_skip",
            model_config=_small_config(),
            data_binding=_binding(),
        )
        module.log = mock.Mock()  # type: ignore[method-assign]
        batch = _batch()
        invalid = torch.zeros((4, 4), dtype=torch.bool)
        batch["teacher_validity"] = {"protein": invalid, "unifrac": invalid}
        optimizer = mock.Mock()
        scheduler = mock.Mock()
        with (
            mock.patch.object(module, "optimizers", return_value=optimizer),
            mock.patch.object(module, "lr_schedulers", return_value=scheduler),
            mock.patch.object(module, "manual_backward") as backward,
        ):
            loss = module.training_step(batch, 0)
        self.assertEqual(float(loss), 0.0)
        backward.assert_not_called()
        optimizer.zero_grad.assert_not_called()
        optimizer.step.assert_not_called()
        scheduler.step.assert_not_called()
        self.assertEqual(module.relation_optimizer_step_count, 0)

    def test_one_teacher_key_cannot_be_silently_omitted(self) -> None:
        module = RelationPretrainingModule(
            arm_name="main_skip",
            model_config=_small_config(),
            data_binding=_binding(),
        )
        module.log = mock.Mock()  # type: ignore[method-assign]
        batch = _batch()
        batch["teacher_distances"] = {"protein": batch["teacher_distances"]["protein"]}
        batch["teacher_validity"] = {"protein": batch["teacher_validity"]["protein"]}
        with self.assertRaisesRegex(RuntimeError, "exactly the two frozen teacher"):
            module._shared_step(batch, "train")

    def test_relation_update_advances_optimizer_and_scheduler_once(self) -> None:
        module = RelationPretrainingModule(
            arm_name="main_skip",
            model_config=_small_config(),
            data_binding=_binding(),
        )
        module.log = mock.Mock()  # type: ignore[method-assign]
        optimizer = mock.Mock()
        scheduler = mock.Mock()
        with (
            mock.patch.object(module, "optimizers", return_value=optimizer),
            mock.patch.object(module, "lr_schedulers", return_value=scheduler),
            mock.patch.object(
                module,
                "manual_backward",
                side_effect=lambda loss: loss.backward(),
            ) as backward,
        ):
            module.training_step(_batch(), 0)
        backward.assert_called_once()
        optimizer.zero_grad.assert_called_once()
        optimizer.step.assert_called_once()
        scheduler.step.assert_called_once()
        self.assertEqual(module.relation_optimizer_step_count, 1)

    def test_shared_initialization_is_identical_across_all_three_arms(self) -> None:
        modules = {
            arm: build_relation_module(
                arm_name=arm,
                data_binding=_binding(),
                seed=42,
                model_config_overrides={
                    "vocab_size": 10,
                    "d_model": 16,
                    "rclr_hidden_dim": 8,
                    "num_layers": 1,
                    "encoder_heads": 4,
                    "encoder_ffn_dim": 32,
                    "decoder_heads": 4,
                    "decoder_ffn_dim": 32,
                    "dropout": 0.0,
                    "max_seq_len": 8,
                },
            )
            for arm in ("main_skip", "main_radius", "pma_skip")
        }
        shared_hashes = {module.shared_initialization_sha256 for module in modules.values()}
        self.assertEqual(len(shared_hashes), 1)
        self.assertEqual(
            modules["main_skip"].full_initialization_sha256,
            modules["main_radius"].full_initialization_sha256,
        )
        self.assertNotEqual(
            modules["main_skip"].full_initialization_sha256,
            modules["pma_skip"].full_initialization_sha256,
        )

    def test_adamw_excludes_all_bias_and_layernorm_parameters_from_decay(self) -> None:
        module = RelationPretrainingModule(
            arm_name="main_skip",
            model_config=_small_config(),
            data_binding=_binding(),
        )
        configured = module.configure_optimizers()
        optimizer = configured["optimizer"]
        group_by_id = {
            id(parameter): group["weight_decay"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        names = dict(module.model.named_parameters())
        self.assertEqual(
            group_by_id[id(names["decoder.cross_attention.in_proj_bias"])], 0.0
        )
        self.assertEqual(group_by_id[id(names["final_token_norm.weight"])], 0.0)
        self.assertEqual(group_by_id[id(names["input_stem.genus_embedding.weight"])], 0.01)

    def test_lightning_manual_optimization_single_step_smoke(self) -> None:
        class OneBatch(Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, index):
                del index
                return _batch()

        module = RelationPretrainingModule(
            arm_name="main_skip",
            model_config=_small_config(),
            data_binding=_binding(),
        )
        trainer = L.Trainer(
            accelerator="cpu",
            devices=1,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=0,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
        )
        trainer.fit(module, train_dataloaders=DataLoader(OneBatch(), batch_size=None))
        self.assertEqual(module.relation_optimizer_step_count, 1)


if __name__ == "__main__":
    unittest.main()
