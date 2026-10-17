"""Normalized branch-length weighted UniFrac and local soft-neighborhood CE."""
from pathlib import Path
import torch
from torch import nn
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
ASSETS=ROOT/'tmp/20260715_relation_geometry_pretraining/audit/gg2_genus_unifrac'
TEACHER_T=.03
STUDENT_T=.05

def weighted_unifrac(mass,length):
    with torch.autocast(device_type=mass.device.type,enabled=False):
        mass=mass.detach().float();length=length.detach().float()
        if mass.ndim!=2 or length.shape!=(mass.shape[1],):raise ValueError('tree mass shape')
        if not torch.isfinite(mass).all() or (mass<0).any() or not torch.isfinite(length).all() or (length<0).any():raise ValueError('invalid tree mass/length')
        weighted=mass*length;total=weighted.sum(1)
        if (total<=0).any():raise ValueError('zero weighted tree mass')
        numerator=torch.cdist(weighted,weighted,p=1)
        result=numerator/(total[:,None]+total[None,:])
        if not torch.isfinite(result).all():raise ValueError('nonfinite UniFrac')
        return result

def neighborhood_details(h,distances):
    if len(h)<2:raise ValueError('neighborhood needs at least two anchors')
    with torch.autocast(device_type=h.device.type,enabled=False):
        diagonal=torch.eye(len(h),device=h.device,dtype=torch.bool)
        teacher_logits=(-distances.detach().float()/TEACHER_T).masked_fill(diagonal,-torch.inf)
        prob=torch.softmax(teacher_logits,1)
        z=torch.nn.functional.normalize(h.float(),dim=1)
        logits=((z@z.T)/STUDENT_T).masked_fill(diagonal,-torch.inf)
        logp=torch.log_softmax(logits,1).masked_fill(diagonal,0.)
        loss=-(prob*logp).sum(1).mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite neighborhood loss')
        with torch.no_grad():
            teacher_logp=torch.log_softmax(teacher_logits,1).masked_fill(diagonal,0.)
            entropy=-(prob*teacher_logp).sum(1).mean()
            kl=(prob*(teacher_logp-logp.detach())).sum(1).mean()
        return loss,entropy,kl


def neighborhood_loss(h,distances):
    return neighborhood_details(h,distances)[0]
