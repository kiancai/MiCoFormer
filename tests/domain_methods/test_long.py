import copy
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from micoformer.models.attn_bias import BiasedTransformerEncoderLayer
from micoformer.domain_adaptation.lora import inject_lora,LoRAConfig
from micoformer.domain_method_adaptation.modules import inject,parameter_groups
from micoformer.domain_long_adaptation.prepare import bind_manifest
from micoformer.domain_long_adaptation.extraction import validate_prefix
from micoformer.mpc_unifrac.workflow import _make_scheduler


class LongRunTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_portable_rebinding_does_not_change_sealed_data(self):
        raw=dict(assets={'corpus':dict(path='assets/corpus.h5ad')},contract=dict(path='PLAN.md'),arms={'a':dict(output='runs/a',data=dict(paths=dict(train='assets/train.npy',val='assets/val.npy'),observed_genus_ids='assets/observed.npy',smoke_rows='assets/smoke.npy'))})
        original=copy.deepcopy(raw);a=bind_manifest(raw,'/cml/task');b=bind_manifest(raw,'/h100/task')
        self.assertEqual(raw,original);self.assertEqual(a['arms']['a']['data']['paths']['train'],'/cml/task/assets/train.npy');self.assertEqual(b['assets']['corpus']['path'],'/h100/task/assets/corpus.h5ad')

    def test_ordinary_lora_matches_previous_implementation_exactly(self):
        torch.manual_seed(42);model=nn.Module();model.backbone=nn.Module();model.backbone.encoder=nn.Module()
        model.backbone.encoder.layers=nn.ModuleList([BiasedTransformerEncoderLayer(512,16,2048,.1)])
        model.config=SimpleNamespace(num_layers=1);old=copy.deepcopy(model)
        torch.manual_seed(42);inject_lora(old,LoRAConfig(128,256.,.05))
        torch.manual_seed(42);result=inject(model,dict(method='lora',rank=128,alpha=256.,dropout=.05))
        self.assertEqual(result['counts']['adaptation'],1048576)
        self.assertEqual(set(old.state_dict()),set(model.state_dict()))
        for key,value in old.state_dict().items():self.assertTrue(torch.equal(value,model.state_dict()[key]),key)
        for key,p in old.named_parameters():self.assertEqual(p.requires_grad,dict(model.named_parameters())[key].requires_grad,key)

    def test_long_schedule_keeps_loraplus_ratio_and_reaches_minimum(self):
        model=nn.Module();model.lora_A=nn.Parameter(torch.ones(1));model.lora_B=nn.Parameter(torch.ones(1))
        arm=dict(adaptation=dict(method='loraplus'),learning_rate=5e-5,lr_ratio=16.)
        opt=torch.optim.AdamW(parameter_groups(model,arm));schedule=_make_scheduler(opt,total_steps=43350,warmup_ratio=.02,base_lr=5e-5,minimum_lr=1e-6)
        rates={}
        import warnings
        for step in [0,867,8670,17340,26010,43350]:
            opt.step()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning);schedule.step(step)
            rates[step]=[g['lr'] for g in opt.param_groups]
            self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'],16.)
        self.assertGreater(rates[8670][0],rates[43350][0]*20)
        self.assertAlmostEqual(rates[43350][0],1e-6);self.assertAlmostEqual(rates[43350][1],1.6e-5)

    def test_milestones_require_complete_prefix_but_allow_running_training(self):
        arm=dict(steps_per_epoch=867)
        epochs=[dict(epoch=i,global_step=i*867,exact_train_row_coverage=True,frozen_weights_unchanged=True,validation=dict(exact_row_coverage=True),samples=110868) for i in range(1,21)]
        validate_prefix(epochs,arm,10)
        with self.assertRaises(RuntimeError):validate_prefix(epochs[:9],arm,10)
        changed=copy.deepcopy(epochs);changed[2]['frozen_weights_unchanged']=False
        with self.assertRaises(RuntimeError):validate_prefix(changed,arm,20)
        changed=copy.deepcopy(epochs);changed[9]['global_step']-=1
        with self.assertRaises(RuntimeError):validate_prefix(changed,arm,10)


if __name__=='__main__':unittest.main()
