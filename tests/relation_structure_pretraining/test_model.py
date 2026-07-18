from __future__ import annotations

import unittest

import torch

from micoformer.relation_pretraining.model import RelationModelConfig, RelationOnlyModel
from micoformer.relation_structure_pretraining.model import (
    STRUCTURE_ARMS,
    StructureRelationModel,
    masked_token_mean,
)


def _small_config() -> RelationModelConfig:
    return RelationModelConfig(
        vocab_size=32,
        d_model=256,
        rclr_hidden_dim=16,
        num_layers=1,
        encoder_heads=8,
        encoder_ffn_dim=64,
        decoder_heads=4,
        decoder_ffn_dim=64,
        dropout=0.0,
        max_seq_len=8,
        decoder_kind="main",
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    genus_ids = torch.tensor([[2, 3, 0], [4, 5, 6]], dtype=torch.long)
    rclr = torch.tensor([[0.5, -0.5, 0.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
    return genus_ids, rclr, genus_ids.eq(0)


class StructureRelationModelTest(unittest.TestCase):
    def test_masked_token_mean(self) -> None:
        hidden = torch.tensor(
            [[[1.0, 3.0], [5.0, 7.0], [99.0, 99.0]], [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]]
        )
        padding = torch.tensor([[False, False, True], [False, False, False]])
        expected = torch.tensor([[3.0, 5.0], [6.0, 8.0]])
        self.assertTrue(torch.equal(masked_token_mean(hidden, padding), expected))

    def test_arm_readout_contracts_and_c0_parity(self) -> None:
        config = _small_config()
        genus_ids, rclr, padding = _inputs()

        torch.manual_seed(42)
        historical = RelationOnlyModel(config).eval()
        torch.manual_seed(42)
        c0 = StructureRelationModel(RelationOnlyModel(config), "c0_decoder").eval()
        with torch.inference_mode():
            historical_output = historical(genus_ids, rclr, padding)
            c0_output = c0(genus_ids, rclr, padding)
        self.assertTrue(torch.equal(c0_output.z, historical_output.z))
        self.assertTrue(torch.equal(c0_output.downstream_z, c0_output.z))

        for arm in STRUCTURE_ARMS[1:]:
            torch.manual_seed(42)
            model = StructureRelationModel(RelationOnlyModel(config), arm).eval()
            with torch.inference_mode():
                output = model(genus_ids, rclr, padding)
            self.assertEqual(tuple(output.z.shape), (2, 256))
            self.assertTrue(torch.allclose(output.z.norm(dim=1), torch.ones(2), atol=1e-6))
            if arm == "c1_token_mean":
                self.assertTrue(torch.equal(output.z, output.backbone_z))
                self.assertTrue(torch.equal(output.downstream_z, output.backbone_z))
            else:
                self.assertIsNotNone(output.projector_z)
                self.assertTrue(torch.equal(output.z, output.projector_z))
                self.assertTrue(torch.equal(output.downstream_z, output.backbone_z))


if __name__ == "__main__":
    unittest.main()
