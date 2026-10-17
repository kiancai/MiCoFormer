"""Keep original corruption; add full-support normalized branch masses."""
import numpy as np
import torch
from scipy import sparse
from micoformer.mpc_pretraining.data import FullSupportRowDataset as BaseDataset,DualCorruptionCollator as BaseCollator,SharedEpoch,batch_to_device as base_to_device
from .teacher import ASSETS
class FullSupportRowDataset(BaseDataset):
    def __getitem__(self,position):
        sample=super().__getitem__(position)
        x=self._h5()['X'];row=int(sample['row_id']);left,right=x['indptr'][row:row+2]
        raw=np.asarray(x['data'][left:right],dtype=np.float64)
        if not np.isfinite(raw).all() or (raw<=0).any() or raw.sum()<=0:raise ValueError('invalid full composition')
        sample['full_relative']=raw/raw.sum()
        return sample
class DualCorruptionCollator(BaseCollator):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.incidence=sparse.load_npz(ASSETS/'genus_to_edge.npz').tocsr().astype(np.float64)
    def __call__(self,samples):
        out=super().__call__(samples)
        indptr=np.r_[0,np.cumsum([len(s['full_indices']) for s in samples])]
        x=sparse.csr_matrix((np.concatenate([s['full_relative'] for s in samples]),np.concatenate([s['full_indices'] for s in samples]),indptr),shape=(len(samples),8114))
        out['edge_mass']=(x@self.incidence).toarray().astype(np.float32)
        return out

def batch_to_device(batch,device):
    out=base_to_device(batch,device);out['edge_mass']=torch.as_tensor(batch['edge_mass'],device=device,dtype=torch.float32)
    return out
