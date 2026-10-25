import copy
import unittest

import torch

from micoformer.domain_gut_adaptation.prepare import ARM, UUID, WEIGHTS, check_contract
from micoformer.domain_method_adaptation.modules import parameter_groups
from micoformer.mpc_unifrac.workflow import _make_scheduler


def recipe():
    return dict(method='gut_h100_loraplus_long',execution=dict(gpu_uuid=UUID,world_size=1),
        assets=dict(corpus=dict(sha256='1bacb2ee3fc88006ad472c500df585997380a98bf88b37e52da04df4cedc0b41'),
                    origin=dict(sha256='0eeef0cc83fae83217a5d6808629b840c919bc77de7824a4fe580479fced289c')),
        arms={ARM:dict(domain='gut',epochs=50,microbatch=128,accumulate=1,seed=42,
            window=4096,sampler='window4096',steps_per_epoch=2304,total_steps=115200,
            learning_rate=5e-5,lr_ratio=16.,head_learning_rate=0.,minimum_lr=1e-6,
            warmup_ratio=.02,weight_decay=.01,gradient_clip=1.,workers=4,
            extract_on_milestone=False,
            milestones=[0,10,20,30,50],weights=copy.deepcopy(WEIGHTS),
            adaptation=dict(method='loraplus',rank=64,alpha=128.,dropout=.05),
            trainable_inventory=dict(counts=dict(adaptation=6291456,heads=559747,frozen=37864576)),
            data=dict(train=294906,val=1539,rows=296445))})


class GutLongContractTests(unittest.TestCase):
    def test_domain_transfer_keeps_method_and_uses_new_update_budget(self):
        raw=recipe();check_contract(raw)
        self.assertEqual((294906+127)//128,2304)
        for key,value in [('steps_per_epoch',867),('total_steps',43350),('accumulate',3),
                          ('domain','respiratory'),('extract_on_milestone',True),('head_learning_rate',1e-5)]:
            with self.subTest(key=key):
                bad=copy.deepcopy(raw);bad['arms'][ARM][key]=value
                with self.assertRaises(RuntimeError):check_contract(bad)

    def test_free_card_or_new_model_is_not_an_approved_substitute(self):
        raw=recipe();raw['execution']['gpu_uuid']='GPU-3d8a1781-fc8e-1855-bff0-6bda65635a07'
        with self.assertRaises(RuntimeError):check_contract(raw)
        raw=recipe();raw['assets']['origin']['sha256']='respiratory_adapted_checkpoint'
        with self.assertRaises(RuntimeError):check_contract(raw)
        raw=recipe();raw['arms'][ARM]['trainable_inventory']['counts']['adaptation']+=25600
        with self.assertRaises(RuntimeError):check_contract(raw)

    def test_new_schedule_keeps_ratio_for_full_gut_budget(self):
        model=torch.nn.Module();model.lora_A=torch.nn.Parameter(torch.ones(1));model.lora_B=torch.nn.Parameter(torch.ones(1))
        arm=recipe()['arms'][ARM];opt=torch.optim.AdamW(parameter_groups(model,arm))
        scheduler=_make_scheduler(opt,total_steps=115200,warmup_ratio=.02,base_lr=5e-5,minimum_lr=1e-6)
        import warnings
        for step in [2304,23040,46080,69120,115200]:
            opt.step()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning);scheduler.step(step)
            self.assertAlmostEqual(opt.param_groups[1]['lr']/opt.param_groups[0]['lr'],16.)
        self.assertAlmostEqual(opt.param_groups[0]['lr'],1e-6)
        self.assertAlmostEqual(opt.param_groups[1]['lr'],1.6e-5)


if __name__=='__main__':unittest.main()
