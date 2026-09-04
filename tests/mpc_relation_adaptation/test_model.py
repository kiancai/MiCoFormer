from __future__ import annotations

import unittest

import torch

from micoformer.mpc_relation_adaptation.model import (
    FrozenMPCResidualAdapter,
    ResidualAdapter,
    ResidualAdapterConfig,
)


class _FakeConfig:
    d_model = 8


class _FakeMPC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = _FakeConfig()
        self.dropout = torch.nn.Dropout(p=0.9)
        self.projection = torch.nn.Linear(1, 8)

    def unmasked_representations(
        self,
        genus_ids: torch.Tensor,
        rclr: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del genus_ids
        token = self.dropout(self.projection(rclr.unsqueeze(-1)))
        valid = (~padding_mask).to(token.dtype).unsqueeze(-1)
        mean = (token * valid).sum(1) / valid.sum(1)
        return token, mean


class ResidualAdapterTest(unittest.TestCase):
    def test_zero_initialization_has_exact_parity(self) -> None:
        adapter = ResidualAdapter(ResidualAdapterConfig(d_model=8, bottleneck_dim=3))
        h = torch.randn(5, 8)
        output = adapter(h)
        self.assertTrue(torch.equal(output.h_raw, output.z_raw))
        self.assertTrue(torch.equal(output.h_unit, output.z_unit))
        self.assertEqual(sum(parameter.numel() for parameter in adapter.parameters()), 75)

    def test_only_adapter_receives_gradients_and_mpc_stays_eval(self) -> None:
        model = FrozenMPCResidualAdapter(
            _FakeMPC(), ResidualAdapterConfig(d_model=8, bottleneck_dim=3)
        ).train()
        self.assertFalse(model.mpc.training)
        self.assertTrue(model.adapter.training)
        self.assertTrue(all(name.startswith("adapter.") for name in model.trainable_parameter_names()))
        genus = torch.tensor([[2, 3, 0], [4, 5, 6]], dtype=torch.long)
        rclr = torch.tensor([[0.5, -0.5, 0.0], [1.0, 0.0, -1.0]])
        output = model(genus, rclr, genus.eq(0))
        output.z_unit[:, 0].sum().backward()
        model.assert_gradient_boundary()
        self.assertTrue(all(parameter.grad is None for parameter in model.mpc.parameters()))

    def test_rejects_non_finite_embedding(self) -> None:
        adapter = ResidualAdapter(ResidualAdapterConfig(d_model=8, bottleneck_dim=3))
        with self.assertRaisesRegex(ValueError, "finite"):
            adapter(torch.full((2, 8), float("nan")))


if __name__ == "__main__":
    unittest.main()
