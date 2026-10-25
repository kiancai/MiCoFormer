import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from micoformer.domain_adaptation.lora import LoRALinear, LoRAConfig
from micoformer.domain_method_adaptation.modules import BottleneckAdapter, DoRALinear, inject, parameter_groups
from micoformer.models.attn_bias import BiasedTransformerEncoderLayer
from micoformer.mpc_unifrac.workflow import _make_scheduler


class MethodsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

    def test_adapter_identity_then_both_projections_learn(self):
        adapter = BottleneckAdapter(12, 5, 0.)
        x = torch.randn(3, 7, 12)
        self.assertTrue(torch.equal(adapter(x), x))
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        (adapter(x).square().mean()).backward()
        self.assertEqual(float(adapter.down.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(adapter.up.weight.grad.abs().sum()), 0.)
        optimizer.step(); optimizer.zero_grad()
        adapter(x).square().mean().backward()
        self.assertGreater(float(adapter.down.weight.grad.abs().sum()), 0.)

    def test_adapter_preln_branch_placement_and_base_parameters(self):
        layer = BiasedTransformerEncoderLayer(512, 16, 2048, .1).eval()
        original = copy.deepcopy(layer)
        model = nn.Module(); model.backbone = nn.Module(); model.backbone.encoder = nn.Module()
        model.backbone.encoder.layers = nn.ModuleList([layer])
        model.config = SimpleNamespace(num_layers=1)
        base = {k: v.detach().clone() for k, v in model.state_dict().items()}
        result = inject(model, dict(method='adapter', bottleneck=256, dropout=.05))
        self.assertEqual(result['counts']['adaptation'], 525824)
        for name, value in base.items():
            self.assertTrue(torch.equal(value, model.state_dict()[name]))
        model.eval()
        x = torch.randn(2, 9, 512); mask = torch.zeros(2, 9, dtype=torch.bool); mask[1, 5:] = True
        self.assertTrue(torch.equal(original(x, mask), layer(x, mask)))

    def test_dora_dense_forward_and_detached_norm_gradients(self):
        module = DoRALinear(nn.Linear(9, 7), LoRAConfig(3, 6., 0.))
        x = torch.randn(2, 4, 9)
        self.assertTrue(torch.equal(module(x), module.base(x)))
        with torch.no_grad(): module.lora_B.normal_(0, .1); module.magnitude.mul_(1.1)
        a = module.lora_A.detach().clone().requires_grad_()
        b = module.lora_B.detach().clone().requires_grad_()
        magnitude = module.magnitude.detach().clone().requires_grad_()
        direction = module.base.weight.detach() + 2 * (b @ a)
        norm = direction.detach().norm(dim=1)
        reference = nn.functional.linear(x, direction * (magnitude / norm)[:, None], module.base.bias)
        actual = module(x)
        torch.testing.assert_close(actual, reference, atol=5e-7, rtol=2e-6)
        target = torch.randn_like(actual)
        (actual * target).sum().backward(); (reference * target).sum().backward()
        for actual_grad, expected in [(module.lora_A.grad, a.grad), (module.lora_B.grad, b.grad), (module.magnitude.grad, magnitude.grad)]:
            torch.testing.assert_close(actual_grad, expected, atol=3e-6, rtol=5e-6)
        self.assertIsNone(module.base.weight.grad)
        self.assertIsNone(module.base.bias.grad)

    def test_dora_dropout_correction_bias_unscaled(self):
        module = DoRALinear(nn.Linear(9, 7), LoRAConfig(3, 6., .2)).train()
        with torch.no_grad(): module.lora_B.normal_(0, .1); module.magnitude.mul_(.8)
        x = torch.randn(2, 4, 9)
        torch.manual_seed(17); actual = module(x)
        torch.manual_seed(17); dropped = nn.functional.dropout(x, .2, True)
        direction = module.base.weight.detach() + 2 * (module.lora_B.detach() @ module.lora_A.detach())
        scale = module.magnitude / direction.norm(dim=1)
        expected = module.base(x) + (scale-1)*nn.functional.linear(dropped, module.base.weight) + scale*2*nn.functional.linear(nn.functional.linear(dropped, module.lora_A), module.lora_B)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_dora_bf16_keeps_small_magnitude_corrections_in_fp32(self):
        module=DoRALinear(nn.Linear(9,7),LoRAConfig(3,6.,0.)).eval()
        with torch.no_grad():module.magnitude.mul_(1.002)
        x=torch.randn(32,9)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            actual=module(x)
            base=module.base(x)
            with torch.autocast('cpu',enabled=False):
                scale=module.magnitude/module.base.weight.detach().norm(dim=1)
            expected=(base+(scale-1)*nn.functional.linear(x,module.base.weight)).to(base.dtype)
            prematurely_rounded=(base+(scale.to(base.dtype)-1)*nn.functional.linear(x,module.base.weight)).to(base.dtype)
        self.assertTrue(torch.equal(actual,expected))
        self.assertFalse(torch.equal(expected,prematurely_rounded))

    def test_loraplus_ratio1_reproduces_adamw_and_ratios_survive_resume(self):
        first = LoRALinear(nn.Linear(9, 7), LoRAConfig(3, 6., 0.))
        second = copy.deepcopy(first)
        arm = dict(adaptation=dict(method='loraplus'), learning_rate=1e-4, lr_ratio=1.)
        original = torch.optim.AdamW([p for p in first.parameters() if p.requires_grad], lr=1e-4, weight_decay=.01)
        plus = torch.optim.AdamW(parameter_groups(second, arm), weight_decay=.01)
        x = torch.randn(4, 9)
        for _ in range(3):
            original.zero_grad(); plus.zero_grad()
            first(x).square().sum().backward(); second(x).square().sum().backward()
            original.step(); plus.step()
        for a, b in zip(first.parameters(), second.parameters()): self.assertTrue(torch.equal(a, b))
        for ratio in [4., 16.]:
            arm['lr_ratio'] = ratio
            opt = torch.optim.AdamW(parameter_groups(second, arm), weight_decay=.01)
            scheduler = _make_scheduler(opt, total_steps=100, warmup_ratio=.1, base_lr=1e-4, minimum_lr=1e-6)
            for _ in range(101):
                self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'], ratio)
                opt.step(); scheduler.step()
            saved = (copy.deepcopy(opt.state_dict()), copy.deepcopy(scheduler.state_dict()))
            opt.load_state_dict(saved[0]); scheduler.load_state_dict(saved[1])
            self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'], ratio)


if __name__ == '__main__': unittest.main()
