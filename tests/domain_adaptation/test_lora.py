import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from micoformer.models.attn_bias import BiasedTransformerEncoderLayer, BiasedTransformerEncoder
from micoformer.domain_adaptation.lora import (
    LoRAConfig, LoRALinear, inject_lora, frozen_sha256, parameter_groups, trainable_inventory,
)
from micoformer.domain_adaptation.workflow import state_equal, make_optimizer, capture_rng, restore_rng


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_layers=2)
        self.backbone = nn.Module()
        self.backbone.encoder = BiasedTransformerEncoder(BiasedTransformerEncoderLayer(16, 4, 32, 0.), 2)
        self.backbone.input_stem = nn.Linear(16, 16)
        self.backbone.final_token_norm = nn.LayerNorm(16)
        self.backbone.mlm_head = nn.Linear(16, 1)
        self.query_head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.backbone.final_token_norm(self.backbone.encoder(self.backbone.input_stem(x)))
        return h, self.backbone.mlm_head(h).mean() + self.query_head(h).mean()


class LoRATest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_zero_delta_and_updated_linear_formula(self):
        linear = nn.Linear(7, 9)
        wrapper = LoRALinear(copy.deepcopy(linear), LoRAConfig(3, 6, 0))
        x = torch.randn(2, 4, 7)
        self.assertTrue(torch.equal(linear(x), wrapper(x)))
        with torch.no_grad():
            wrapper.lora_B.normal_()
        expected = nn.functional.linear(x, linear.weight + 2 * wrapper.lora_B @ wrapper.lora_A, linear.bias)
        torch.testing.assert_close(wrapper(x), expected, rtol=2e-6, atol=2e-6)

    def test_only_adapters_and_heads_change_embedding(self):
        model = TinyModel().eval()
        x = torch.randn(3, 7, 16)
        before = model(x)[0].detach().clone()
        inventory = inject_lora(model, LoRAConfig(4, 8, .05))
        self.assertEqual(len(inventory["parameters"]["lora"]), 16)
        model.eval()
        self.assertTrue(torch.equal(before, model(x)[0]))
        frozen = frozen_sha256(model)
        opt = torch.optim.AdamW(parameter_groups(model, lora_lr=1e-3, head_lr=1e-4))
        model.train()
        for _ in range(2):
            opt.zero_grad()
            _, score = model(x)
            (score - 1).square().backward()
            self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
            opt.step()
        self.assertEqual(frozen, frozen_sha256(model))
        model.eval()
        self.assertFalse(torch.equal(before, model(x)[0]))
        with self.assertRaises(RuntimeError):
            inject_lora(model, LoRAConfig())
        model.backbone.input_stem.weight.requires_grad_(True)
        with self.assertRaises(RuntimeError):
            trainable_inventory(model, LoRAConfig(4, 8, .05))

    def test_head_learning_rate_ratio_and_schedule(self):
        model = TinyModel()
        inject_lora(model, LoRAConfig(4, 8, .05))
        arm = dict(learning_rate=1e-4, head_learning_rate=1e-5, weight_decay=.01,
                   total_steps=100, warmup_ratio=.02, minimum_lr=1e-6)
        opt, sched = make_optimizer(model, arm)
        for _ in range(100):
            self.assertAlmostEqual(opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"], .1)
            opt.step(); sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-6)
        self.assertAlmostEqual(opt.param_groups[1]["lr"], 1e-7)

    def test_rng_replay_and_exact_equality(self):
        rng = capture_rng(torch.device("cpu"))
        a = torch.rand(20)
        restore_rng(rng, torch.device("cpu"))
        self.assertTrue(torch.equal(a, torch.rand(20)))
        self.assertFalse(state_equal(torch.tensor([1.]), torch.tensor([1.0000001])))


if __name__ == "__main__":
    unittest.main()
