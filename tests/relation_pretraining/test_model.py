from __future__ import annotations

import unittest

import torch

from micoformer.relation_pretraining.model import (
    DEFAULT_VOCAB_SIZE,
    PAD_TOKEN_ID,
    REAL_GENUS_COUNT,
    RelationModelConfig,
    RelationOnlyModel,
)


def _small_config(*, decoder_kind: str = "main") -> RelationModelConfig:
    return RelationModelConfig(
        vocab_size=32,
        d_model=32,
        rclr_hidden_dim=8,
        num_layers=2,
        encoder_heads=4,
        encoder_ffn_dim=64,
        decoder_heads=4,
        decoder_ffn_dim=64,
        dropout=0.0,
        max_seq_len=16,
        decoder_kind=decoder_kind,
    )


class RelationModelTest(unittest.TestCase):
    def test_locked_default_shape_and_no_forbidden_parameters(self) -> None:
        config = RelationModelConfig()
        self.assertEqual(config.vocab_size, DEFAULT_VOCAB_SIZE)
        self.assertEqual(DEFAULT_VOCAB_SIZE, REAL_GENUS_COUNT + 2)
        self.assertEqual(DEFAULT_VOCAB_SIZE, 8_116)
        self.assertEqual(PAD_TOKEN_ID, 0)
        self.assertEqual((config.num_layers, config.d_model, config.encoder_heads), (6, 256, 8))
        self.assertEqual(
            (config.encoder_ffn_dim, config.decoder_heads, config.decoder_ffn_dim),
            (1_024, 4, 1_024),
        )

        model = RelationOnlyModel(config)
        forbidden = ("mask_token", "mlm", "prior", "register", "phylo", "protein", "position")
        parameter_names = [name.lower() for name, _ in model.named_parameters()]
        self.assertFalse(any(term in name for name in parameter_names for term in forbidden))
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(parameter_count, 7_500_000)
        self.assertLess(parameter_count, 7_800_000)

    def test_padding_and_token_permutation_invariance(self) -> None:
        short_ids = torch.tensor([[2, 9, 4]], dtype=torch.long)
        short_rclr = torch.tensor([[0.4, -0.1, -0.3]], dtype=torch.float32)
        padded_ids = torch.tensor([[2, 9, 4, 0, 0, 0]], dtype=torch.long)
        padded_rclr = torch.tensor([[0.4, -0.1, -0.3, 0.0, 0.0, 0.0]], dtype=torch.float32)
        permuted_ids = torch.tensor([[4, 2, 9, 0, 0, 0]], dtype=torch.long)
        permuted_rclr = torch.tensor([[-0.3, 0.4, -0.1, 0.0, 0.0, 0.0]], dtype=torch.float32)

        for decoder_kind in ("main", "pma"):
            with self.subTest(decoder_kind=decoder_kind):
                torch.manual_seed(4)
                model = RelationOnlyModel(_small_config(decoder_kind=decoder_kind)).eval()
                with torch.no_grad():
                    short = model(short_ids, short_rclr).z
                    padded = model(padded_ids, padded_rclr).z
                    permuted = model(permuted_ids, permuted_rclr).z
                torch.testing.assert_close(short, padded, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(padded, permuted, rtol=2e-5, atol=2e-6)

    def test_final_z_unit_norm_and_finite_backward(self) -> None:
        genus_ids = torch.tensor(
            [
                [2, 3, 4, 0, 0],
                [5, 6, 7, 8, 0],
                [9, 10, 11, 12, 13],
            ],
            dtype=torch.long,
        )
        rclr = torch.tensor(
            [
                [0.4, -0.1, -0.3, 0.0, 0.0],
                [0.7, 0.2, -0.4, -0.5, 0.0],
                [0.8, 0.4, 0.0, -0.3, -0.9],
            ],
            dtype=torch.float32,
        )
        for decoder_kind in ("main", "pma"):
            with self.subTest(decoder_kind=decoder_kind):
                torch.manual_seed(9)
                model = RelationOnlyModel(_small_config(decoder_kind=decoder_kind)).train()
                output = model(genus_ids, rclr)
                self.assertEqual(output.z.shape, (3, 32))
                self.assertEqual(output.z.dtype, torch.float32)
                self.assertEqual(output.z_raw.shape, (3, 32))
                self.assertEqual(output.token_embeddings.shape, (3, 5, 32))
                torch.testing.assert_close(
                    output.z.norm(dim=-1),
                    torch.ones(3),
                    rtol=1e-5,
                    atol=1e-6,
                )
                loss = output.z[:, :5].sum()
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_pad_contract_and_empty_sample_fail_closed(self) -> None:
        model = RelationOnlyModel(_small_config()).eval()
        ids = torch.tensor([[2, 3, 0]], dtype=torch.long)
        rclr = torch.tensor([[0.1, -0.1, 0.0]])
        wrong_mask = torch.tensor([[False, False, False]])
        with self.assertRaisesRegex(ValueError, "exactly"):
            model(ids, rclr, wrong_mask)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(torch.zeros((1, 3), dtype=torch.long), torch.zeros((1, 3)))


if __name__ == "__main__":
    unittest.main()
