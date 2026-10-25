import copy
from types import SimpleNamespace
import unittest
import torch
from torch import nn
from micoformer.models.attn_bias import BiasedTransformerEncoderLayer
from micoformer.domain_method_adaptation.modules import inject, parameter_groups
from micoformer.domain_long_adaptation.eva import Moments
from micoformer.domain_long_adaptation.prepare import specifications
from micoformer.mpc_unifrac.workflow import _make_scheduler


def model():
    value=nn.Module();value.backbone=nn.Module();value.backbone.encoder=nn.Module()
    value.backbone.encoder.layers=nn.ModuleList([BiasedTransformerEncoderLayer(512,16,2048,.1)])
    value.config=SimpleNamespace(num_layers=1)
    return value


class Next50Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_qkv_preserves_initialization_rng_and_matches_dense_independent_update(self):
        packed=model();independent=copy.deepcopy(packed)
        spec=dict(method='loraplus',rank=64,alpha=128.,dropout=.05)
        torch.manual_seed(42);inject(packed,spec);rng=torch.get_rng_state().clone()
        torch.manual_seed(42);result=inject(independent,dict(spec,qkv_independent=True))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(result['counts']['adaptation'],589824)
        p=packed.backbone.encoder.layers[0];q=independent.backbone.encoder.layers[0]
        for suffix in ('self_attn.out_proj','ff.0','ff.3'):
            for name,value in p.get_submodule(suffix).state_dict().items():
                self.assertTrue(torch.equal(value,q.get_submodule(suffix).state_dict()[name]))
        wrapper=q.self_attn.in_proj
        for branch in wrapper.branches.values():
            self.assertTrue(torch.equal(branch.lora_A,p.self_attn.in_proj.lora_A))
        with torch.no_grad():
            for branch in wrapper.branches.values():branch.lora_B.normal_(0,.02)
        wrapper.eval();x=torch.randn(3,7,512)
        dense=torch.cat([b.lora_B@b.lora_A for b in wrapper.branches.values()])
        expected=nn.functional.linear(x,wrapper.base.weight+2*dense,wrapper.base.bias)
        torch.testing.assert_close(wrapper(x),expected,atol=2e-6,rtol=2e-5)
        wrapper(x).square().mean().backward()
        grads=[b.lora_A.grad for b in wrapper.branches.values()]
        self.assertTrue(all(g is not None and g.norm()>0 for g in grads))
        self.assertFalse(torch.equal(grads[0],grads[1]))
        self.assertIsNone(wrapper.base.weight.grad)

    def test_all_six_counts_and_heads_are_frozen(self):
        expected=[589824,589824,524288,1048576,524288,526336]
        for (name,spec),count in zip(specifications().items(),expected,strict=True):
            value=model();value.query_head=nn.Linear(512,1)
            result=inject(value,spec)
            self.assertEqual(result['counts']['adaptation'],count,name)
            self.assertTrue(all(not p.requires_grad for p in value.query_head.parameters()))

    def test_norm_scope_and_lr_survive_optimizer_restore(self):
        value=model();value.backbone.layer_norm=nn.LayerNorm(512)
        spec=specifications()['norm_loraplus_r64'];inject(value,spec)
        self.assertTrue(all(not p.requires_grad for p in value.backbone.layer_norm.parameters()))
        arm=dict(adaptation=spec,learning_rate=5e-5,lr_ratio=16.,norm_learning_rate=1e-5)
        groups=parameter_groups(value,arm)
        found={g['group_name']:g for g in groups}
        self.assertEqual(found['A']['lr'],5e-5);self.assertEqual(found['B']['lr'],8e-4)
        self.assertEqual(found['norm']['lr'],1e-5);self.assertEqual(found['norm']['weight_decay'],0.)
        self.assertEqual(sum(p.numel() for p in found['norm']['params']),2048)
        optimizer=torch.optim.AdamW(groups,weight_decay=.01)
        saved=copy.deepcopy(optimizer.state_dict());optimizer.load_state_dict(saved)
        self.assertEqual(optimizer.param_groups[0]['group_name'],groups[0]['group_name'])

    def test_eva_principal_subspace_against_known_covariance(self):
        values=torch.tensor([[6.,0.,0.],[-6.,0.,0.],[0.,2.,0.],[0.,-2.,0.],[0.,0.,.1],[0.,0.,-.1]])
        sketch=Moments(3,'cpu');sketch.add(values[:3]);sketch.add(values[3:])
        a,record=sketch.principal(2)
        torch.testing.assert_close(a.T@a,torch.diag(torch.tensor([1.,1.,0.])),atol=1e-6,rtol=0)
        self.assertEqual(record['tokens'],6)
        self.assertGreater(record['explained_variance_ratio'],.999)

    def test_eva_rejects_empty_or_degenerate_statistics(self):
        sketch=Moments(4,'cpu')
        with self.assertRaises(RuntimeError):sketch.principal(2)
        sketch.add(torch.ones(6,4))
        with self.assertRaises(RuntimeError):sketch.principal(2)

    def test_independent_loraplus_routes_every_b_to_high_lr(self):
        value=model();spec=specifications()['qkv_loraplus_r64'];inject(value,spec)
        groups=parameter_groups(value,dict(adaptation=spec,learning_rate=5e-5,lr_ratio=16.))
        named=dict(value.named_parameters())
        high={id(p) for g in groups if g['group_name']=='B' for p in g['params']}
        for name,p in named.items():
            if p.requires_grad:self.assertEqual(id(p) in high,name.endswith('.lora_B'),name)

    def test_ratio32_keeps_entire_b_schedule_and_halves_a(self):
        value=model();spec=specifications()['loraplus_r64_ratio32'];inject(value,spec)
        schedules=[]
        for lr,ratio,minimum in [(5e-5,16.,1e-6),(2.5e-5,32.,5e-7)]:
            opt=torch.optim.AdamW(parameter_groups(value,dict(adaptation=spec,learning_rate=lr,lr_ratio=ratio)))
            schedule=_make_scheduler(opt,total_steps=43350,warmup_ratio=.02,base_lr=lr,minimum_lr=minimum)
            rates=[]
            import warnings
            for step in [0,867,8670,26010,43350]:
                opt.step()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',UserWarning);schedule.step(step)
                rates.append([g['lr'] for g in opt.param_groups])
            schedules.append(rates)
        for old,new in zip(*schedules,strict=True):
            self.assertEqual(new[0],old[0]/2)
            self.assertEqual(new[1],old[1])


if __name__=='__main__':unittest.main()
