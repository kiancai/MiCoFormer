"""Add UniFrac supervision to the Query sample representation.
在 Query 样本表示上增加 UniFrac 邻域监督。
"""
from typing import NamedTuple
import numpy as np
import torch
from torch import Tensor
from micoformer.mpc_fixed_pretraining.model import MPCFixedPretrainingModel,MPCFixedModelConfig
from .teacher import ASSETS,weighted_unifrac,neighborhood_loss,neighborhood_details
class Output(NamedTuple):
    mlm_loss:Tensor
    continuous_loss:Tensor
    presence_loss:Tensor
    unifrac_loss:Tensor
    mlm_count:int
    continuous_count:int
    presence_count:int
    unifrac_count:int
    unifrac_teacher_entropy:Tensor
    unifrac_kl:Tensor
class UniFracModel(MPCFixedPretrainingModel):
    def __init__(self,candidate_table,config=None):
        super().__init__(candidate_table,config or MPCFixedModelConfig())
        self.register_buffer('unifrac_lengths',torch.from_numpy(np.load(ASSETS/'branch_lengths.npy')).float(),persistent=False)
    def forward(self,genus_ids,rclr,padding_mask,query_mask,mlm_mask,negative_genus_ids,edge_mass):
        padding=self.backbone._validate_inputs(genus_ids,rclr,padding_mask)
        if negative_genus_ids.shape!=genus_ids.shape or negative_genus_ids.dtype!=torch.long:raise TypeError('negative IDs must be aligned long tensor')
        m,mc=self._mlm_loss(genus_ids,rclr,padding,mlm_mask)
        q=self._query_output(genus_ids,rclr,padding,query_mask,negative_genus_ids)
        c,p,cc,pc=self._query_losses(q)
        u,entropy,kl=neighborhood_details(q.sample_h,weighted_unifrac(edge_mass,self.unifrac_lengths))
        return Output(m,c,p,u,mc,cc,pc,len(genus_ids),entropy,kl)
    def component_loss(self,component,**arguments):
        mass=arguments.pop('edge_mass')
        if component!='unifrac':return super().component_loss(component,**arguments)
        q=self._query_output(arguments['genus_ids'],arguments['rclr'],arguments['padding_mask'],arguments['query_mask'],arguments['negative_genus_ids'])
        return neighborhood_loss(q.sample_h,weighted_unifrac(mass,self.unifrac_lengths))
