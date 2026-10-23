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

    def test_only_adapters_change_embedding_through_frozen_heads(self):
        model = TinyModel().eval()
        x = torch.randn(3, 7, 16)
        before = model(x)[0].detach().clone()
        inventory = inject_lora(model, LoRAConfig(4, 8, .05))
        self.assertEqual(len(inventory["parameters"]["lora"]), 16)
        model.eval()
        self.assertTrue(torch.equal(before, model(x)[0]))
        frozen = frozen_sha256(model)
        opt = torch.optim.AdamW(parameter_groups(model, lora_lr=1e-3, head_lr=0.0))
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

    def test_only_adapter_optimizer_and_schedule(self):
        model = TinyModel()
        inject_lora(model, LoRAConfig(4, 8, .05))
        arm = dict(learning_rate=1e-4, head_learning_rate=0.0, weight_decay=.01,
                   total_steps=100, warmup_ratio=.02, minimum_lr=1e-6)
        opt, sched = make_optimizer(model, arm)
        for _ in range(100):
            self.assertEqual(len(opt.param_groups), 1)
            opt.step(); sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 1e-6)
        self.assertTrue(all(not p.requires_grad for p in model.query_head.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.mlm_head.parameters()))

    def test_rng_replay_and_exact_equality(self):
        rng = capture_rng(torch.device("cpu"))
        a = torch.rand(20)
        restore_rng(rng, torch.device("cpu"))
        self.assertTrue(torch.equal(a, torch.rand(20)))
        self.assertFalse(state_equal(torch.tensor([1.]), torch.tensor([1.0000001])))

    def test_recomputation_preserves_dropout_outputs_gradients_and_rng(self):
        model = TinyModel()
        inject_lora(model, LoRAConfig(4, 8, .05))
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.p = .1
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith('lora_B'):
                    parameter.normal_(std=.01)
        reference = copy.deepcopy(model).train()
        recompute = copy.deepcopy(model).train()
        recompute.backbone.encoder.grad_checkpointing = True
        x = torch.randn(3, 7, 16)
        expected = None
        for candidate in (reference, recompute):
            torch.manual_seed(71)
            h, score = candidate(x)
            score.square().backward()
            result = (h.detach(), {n: p.grad.clone() for n,p in candidate.named_parameters()
                                  if p.requires_grad}, torch.get_rng_state())
            if expected is None:
                expected = result
            else:
                self.assertTrue(torch.equal(expected[0], result[0]))
                self.assertTrue(torch.equal(expected[2], result[2]))
                for name in expected[1]:
                    torch.testing.assert_close(expected[1][name], result[1][name], rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
