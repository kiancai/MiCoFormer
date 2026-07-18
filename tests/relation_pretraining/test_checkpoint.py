from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

from micoformer.relation_pretraining.data import sha256_file
from micoformer.relation_pretraining.extract import (
    _load_embedding_checkpoint,
    extract_relation_embeddings,
)
from micoformer.relation_pretraining.model import RelationModelConfig
from micoformer.relation_pretraining.module import (
    RelationPretrainingModule,
    load_relation_checkpoint,
)
from micoformer.relation_pretraining.workflow import _write_or_verify_epoch0


def _config() -> RelationModelConfig:
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
        decoder_kind="main",
    )


def _binding(corpus: str = "a") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "corpus_sha256": corpus * 64,
        "schedule_manifest_sha256": "b" * 64,
        "split_manifest_sha256": "c" * 64,
        "split_npz_sha256": "d" * 64,
        "physical_batch_size": 32,
        "schedule_parameters": {"batch_size": 32},
        "schedule_assets": {"corpus": corpus * 64},
        "binding_sha256": "e" * 64,
    }
    return payload


class RelationCheckpointTests(unittest.TestCase):
    def _write_checkpoint(self, path: Path, module: RelationPretrainingModule) -> dict:
        checkpoint = {
            "state_dict": module.state_dict(),
            "epoch": 2,
            "global_step": 5,
        }
        module.on_save_checkpoint(checkpoint)
        torch.save(checkpoint, path)
        return checkpoint

    def test_strict_reload_has_output_parity_and_no_teacher_arrays(self) -> None:
        torch.manual_seed(42)
        module = RelationPretrainingModule(
            arm_name="main_skip", model_config=_config(), data_binding=_binding()
        ).eval()
        module._record_cache_provenance(
            {
                "cache_manifest_sha256": "1" * 64,
                "cache_sha256": "2" * 64,
                "schedule_file_sha256": "3" * 64,
            }
        )
        genus_ids = torch.tensor([[2, 3, 0], [4, 5, 6]], dtype=torch.long)
        rclr = torch.tensor([[-0.2, 0.2, 0.0], [-0.3, 0.1, 0.2]], dtype=torch.float32)
        mask = genus_ids.eq(0)
        with torch.no_grad():
            before = module(genus_ids, rclr, mask).z

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            checkpoint = self._write_checkpoint(path, module)
            reloaded = load_relation_checkpoint(
                path, expected_data_binding=_binding(), map_location="cpu"
            ).eval()
            with torch.no_grad():
                after = reloaded(genus_ids, rclr, mask).z
            self.assertTrue(torch.equal(before, after))
            self.assertEqual(reloaded.consumed_teacher_caches, module.consumed_teacher_caches)

            def walk(value):
                if isinstance(value, dict):
                    for key, child in value.items():
                        self.assertNotIn("teacher_distances", str(key))
                        yield from walk(child)
                elif isinstance(value, (list, tuple)):
                    for child in value:
                        yield from walk(child)
                elif isinstance(value, torch.Tensor):
                    yield value

            tensors = list(walk(checkpoint))
            self.assertFalse(any(tensor.ndim == 2 and tensor.shape == (32, 32) for tensor in tensors))

    def test_strict_reload_rejects_data_binding_drift(self) -> None:
        module = RelationPretrainingModule(
            arm_name="main_skip", model_config=_config(), data_binding=_binding()
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            self._write_checkpoint(path, module)
            with self.assertRaisesRegex(RuntimeError, "data binding"):
                load_relation_checkpoint(
                    path, expected_data_binding=_binding("f"), map_location="cpu"
                )

    def test_epoch0_architecture_artifact_strict_reload(self) -> None:
        torch.manual_seed(42)
        module = RelationPretrainingModule(
            arm_name="main_skip", model_config=_config(), data_binding=_binding()
        ).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "main.ckpt"
            _write_or_verify_epoch0(path, module)
            loaded = _load_embedding_checkpoint(path)
            genus_ids = torch.tensor([[2, 3, 0]], dtype=torch.long)
            rclr = torch.tensor([[-0.2, 0.2, 0.0]], dtype=torch.float32)
            with torch.no_grad():
                expected = module.model(genus_ids, rclr, genus_ids.eq(0)).z
                observed = loaded.model(genus_ids, rclr, genus_ids.eq(0)).z
            self.assertTrue(torch.equal(expected, observed))
            self.assertEqual(loaded.checkpoint_kind, "relation_architecture_epoch0")

    def test_cpu_embedding_export_preserves_row_order_and_unit_norm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / "toy.h5ad"
            matrix = np.asarray(
                [
                    [0.0, 0.2, 0.3, 0.5, 0.0, 0.0, 0.0, 0.0],
                    [0.4, 0.0, 0.1, 0.0, 0.5, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.2, 0.3, 0.0, 0.5, 0.0],
                ],
                dtype=np.float32,
            )
            row_index, column_index = np.nonzero(matrix)
            indices = column_index.astype(np.int32)
            data = matrix[row_index, column_index]
            indptr = np.concatenate(
                [[0], np.cumsum((matrix != 0).sum(axis=1))]
            ).astype(np.int32)
            with h5py.File(corpus, "w") as handle:
                group = handle.create_group("X")
                group.attrs["encoding-type"] = "csr_matrix"
                group.attrs["shape"] = matrix.shape
                group.create_dataset("indices", data=indices)
                group.create_dataset("indptr", data=indptr)
                group.create_dataset("data", data=data)

            binding = _binding()
            binding["corpus_sha256"] = sha256_file(corpus)
            binding["schedule_assets"] = {"corpus": binding["corpus_sha256"]}
            module = RelationPretrainingModule(
                arm_name="main_skip", model_config=_config(), data_binding=binding
            ).eval()
            checkpoint = root / "model.ckpt"
            self._write_checkpoint(checkpoint, module)
            rows = np.asarray([2, 0], dtype=np.int64)
            output = root / "embedding.npz"
            extract_relation_embeddings(
                checkpoint_path=checkpoint,
                h5ad_path=corpus,
                row_ids=rows,
                output_path=output,
                device="cpu",
                batch_size=2,
                require_d_model_256=False,
            )
            with np.load(output, allow_pickle=False) as archive:
                self.assertEqual(archive.files, ["row_ids", "z"])
                self.assertTrue(np.array_equal(archive["row_ids"], rows))
                self.assertEqual(archive["z"].dtype, np.float32)
                self.assertTrue(
                    np.allclose(np.linalg.norm(archive["z"], axis=1), 1.0, atol=1e-6)
                )
            self.assertTrue(output.with_suffix(".npz.manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
