import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from micoformer.domain_bucket_adaptation.sampling import BucketSampler
from micoformer.domain_bucket_adaptation.workflow import split_backward
from micoformer.mpc_unifrac.teacher import weighted_unifrac,neighborhood_details
from micoformer.mpc_unifrac.model import Output


class SamplingTest(unittest.TestCase):
    def test_full_resp_roster_and_resume(self):
        lengths=np.random.default_rng(7).integers(5,513,110868)
        for mode in ("random","window4096"):
            sampler=BucketSampler(lengths,mode=mode,epoch=0)
            self.assertEqual(len(sampler),867)
            self.assertEqual(sorted(map(len,sampler.batches)),[20]+[128]*866)
            self.assertTrue(np.array_equal(np.sort(np.concatenate(sampler.batches)),np.arange(len(lengths))))
            resumed=BucketSampler(lengths,mode=mode,epoch=0,cursor=500)
            self.assertEqual(list(resumed),sampler.batches[500:])
            self.assertEqual(resumed.order_sha256,sampler.order_sha256)
            self.assertNotEqual(BucketSampler(lengths,mode=mode,epoch=1).order_sha256,sampler.order_sha256)

    def test_singleton_tail_and_finished_cursor(self):
        for mode in ("random","window4096"):
            sampler=BucketSampler(np.full(4097,8),mode=mode,epoch=0)
            self.assertEqual(sorted(map(len,sampler.batches)),[2,127]+[128]*31)
            self.assertEqual(len(BucketSampler(np.full(4097,8),mode=mode,epoch=0,cursor=len(sampler.batches))),0)

    def test_window_membership_and_padding_reduction(self):
        lengths=np.random.default_rng(8).integers(5,513,16384)
        order=torch.randperm(len(lengths),generator=torch.Generator().manual_seed(42)).numpy()
        windows=[set(x) for x in order.reshape(-1,4096)]
        grouped=BucketSampler(lengths,mode="window4096",epoch=0)
        self.assertTrue(all(any(set(batch)<=window for window in windows) for batch in grouped.batches))
        random=BucketSampler(lengths,mode="random",epoch=0)
        cost=lambda sampler:sum(len(b)*max(lengths[b]) for b in sampler.batches)
        self.assertLess(cost(grouped),.65*cost(random))

    def test_illegal_resume_or_mode_rejected(self):
        for kwargs in (dict(mode="other",cursor=0),dict(mode="random",cursor=9)):
            with self.assertRaises(ValueError):BucketSampler([5,8],epoch=0,**kwargs)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(3,5),nn.Tanh(),nn.Dropout(.15))
        self.mlm_head=nn.Linear(5,1)
        self.query_head=nn.Linear(5,1)
        self.mlm_head.requires_grad_(False)
        self.query_head.requires_grad_(False)
        self.register_buffer("unifrac_lengths",torch.ones(3))

    def _mlm_loss(self,ids,rclr,padding,mask):
        return self.mlm_head(self.encoder(rclr)).square().mean(),len(ids)

    def _query_output(self,ids,rclr,padding,mask,negative):
        return SimpleNamespace(sample_h=self.encoder(rclr))

    def _query_losses(self,q):
        logits=self.query_head(q.sample_h)
        return logits.square().mean(),(logits-1).square().mean(),len(logits),len(logits)*2

    def forward(self,**args):
        m,mc=self._mlm_loss(args['genus_ids'],args['rclr'],args['padding_mask'],args['mlm_mask'])
        q=self._query_output(args['genus_ids'],args['rclr'],args['padding_mask'],args['query_mask'],args['negative_genus_ids'])
        c,p,cc,pc=self._query_losses(q)
        u,e,k=neighborhood_details(q.sample_h,weighted_unifrac(args['edge_mass'],self.unifrac_lengths))
        return Output(m,c,p,u,mc,cc,pc,len(q.sample_h),e,k)


class SplitTest(unittest.TestCase):
    def test_all_four_losses_frozen_heads_and_dropout_gradient_sum(self):
        torch.set_num_threads(1)
        torch.manual_seed(41)
        model=Tiny().train()
        args={name:torch.ones(4,3) for name in ('genus_ids','padding_mask','mlm_mask','query_mask','negative_genus_ids')}
        args['rclr']=torch.randn(4,3)
        args['edge_mass']=torch.rand(4,3)+.2
        weights=dict(mlm=1.,continuous=.534399128791011,presence=.24296011504612833,unifrac=.00005)
        torch.manual_seed(52)
        output=model(**args)
        sum(weights[n]*getattr(output,n+'_loss') for n in weights).backward()
        expected=[p.grad.clone() for p in model.encoder.parameters()]
        model.zero_grad(set_to_none=True)
        torch.manual_seed(52)
        actual=split_backward(model,args,weights,bf16=False)
        for a,b in zip(expected,[p.grad for p in model.encoder.parameters()]):
            torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        for name in weights:torch.testing.assert_close(getattr(output,name+'_loss'),getattr(actual,name+'_loss'))
        self.assertTrue(all(p.grad is None for p in list(model.mlm_head.parameters())+list(model.query_head.parameters())))


if __name__=='__main__':unittest.main()
