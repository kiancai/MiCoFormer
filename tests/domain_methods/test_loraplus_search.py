import copy
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from micoformer.domain_method_adaptation.modules import parameter_groups
from micoformer.domain_long_adaptation.prepare import configurations, specifications, NAMES
from micoformer.domain_long_adaptation.extraction import validate_prefix
from micoformer.mpc_unifrac.workflow import _make_scheduler


def rates(cfg, steps):
    model=nn.Module()
    model.lora_A=nn.Parameter(torch.ones(1));model.lora_B=nn.Parameter(torch.ones(1))
    arm=dict(cfg,adaptation=dict(method='loraplus'))
    opt=torch.optim.AdamW(parameter_groups(model,arm))
    sch=_make_scheduler(opt,total_steps=cfg['epochs']*867,warmup_ratio=.02,
        base_lr=cfg['learning_rate'],minimum_lr=cfg['minimum_lr'])
    out=[]
    for step in steps:
        opt.step()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning);sch.step(step)
        out.append([g['lr'] for g in opt.param_groups])
    return torch.tensor(out,dtype=torch.float64)


class SearchTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_lr_scaling_and_ratio8_preserve_full_intended_trajectories(self):
        cfgs=configurations();base=dict(epochs=50,learning_rate=5e-5,minimum_lr=1e-6,lr_ratio=16.)
        steps=[0,1,867,8670,21675,43350];ref=rates(base,steps)
        torch.testing.assert_close(rates(cfgs[NAMES[0]],steps),ref*.5,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(rates(cfgs[NAMES[1]],steps),ref*2,rtol=1e-12,atol=1e-14)
        ratio8=rates(cfgs[NAMES[2]],steps)
        torch.testing.assert_close(ratio8[:,0],ref[:,0]*2,rtol=1e-12,atol=1e-14)
        torch.testing.assert_close(ratio8[:,1],ref[:,1],rtol=1e-12,atol=1e-14)

    def test_100_epoch_schedule_retains_lr_at_50_and_reaches_floor_at_100(self):
        cfg=configurations()[NAMES[-1]]
        result=rates(cfg,[1734,43350,86700])
        self.assertAlmostEqual(result[0,0].item(),5e-5)
        self.assertGreater(result[1,0].item(),2.5e-5)
        self.assertAlmostEqual(result[2,0].item(),1e-6)
        self.assertAlmostEqual(result[2,1].item(),1.6e-5)

    def test_dropout_arm_changes_only_adapter_dropout(self):
        from micoformer.domain_adaptation.lora import LoRALinear
        from micoformer.domain_adaptation.lora import LoRAConfig
        torch.manual_seed(42);base=nn.Linear(16,24)
        torch.manual_seed(99);a=LoRALinear(copy.deepcopy(base),LoRAConfig(4,8.,.05))
        torch.manual_seed(99);b=LoRALinear(copy.deepcopy(base),LoRAConfig(4,8.,0.))
        for key,value in a.state_dict().items():self.assertTrue(torch.equal(value,b.state_dict()[key]),key)
        torch.testing.assert_close(a.eval()(torch.ones(3,16)),b.eval()(torch.ones(3,16)),rtol=0,atol=0)
        self.assertEqual(specifications()[NAMES[3]],dict(method='loraplus',rank=64,alpha=128.,dropout=0.))

    def test_100_endpoint_requires_complete_valid_prefix(self):
        rows=[dict(epoch=i,global_step=i*867,exact_train_row_coverage=True,
            frozen_weights_unchanged=True,validation=dict(exact_row_coverage=True),samples=110868) for i in range(1,101)]
        validate_prefix(rows,dict(steps_per_epoch=867),100)
        with self.assertRaises(RuntimeError):validate_prefix(rows[:-1],dict(steps_per_epoch=867),100)

    def test_queue_waits_for_occupant_and_two_idle_observations(self):
        from micoformer.domain_long_adaptation import runner
        with tempfile.TemporaryDirectory() as folder,patch.object(runner,'RUN',Path(folder)),\
                patch.object(runner.subprocess,'check_output',side_effect=['123\n','','']) as query,\
                patch.object(runner.time,'sleep') as sleep:
            runner.wait_for_idle(0,NAMES[-1],'gate',timeout=30)
            self.assertEqual(query.call_count,3)
            self.assertEqual([c.args[0] for c in sleep.call_args_list],[15,2])


if __name__=='__main__':unittest.main()
