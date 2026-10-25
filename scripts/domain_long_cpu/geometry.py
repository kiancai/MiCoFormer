"""Paired geometry: distinguish raw drift from changes surviving alignment."""
import numpy as np
def quantiles(v):return {str(q):float(np.quantile(v,q)) for q in [0,.05,.5,.95,1]}
def compare(x,y,k=20,neighbors=True):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 assert x.shape==y.shape and x.ndim==2 and np.isfinite(x).all() and np.isfinite(y).all()
 mx=x.mean(0);my=y.mean(0);xc=x-mx;yc=y-my
 nx=np.linalg.norm(x,axis=1);ny=np.linalg.norm(y,axis=1)
 xx=xc.T@xc;yy=yc.T@yc;xy=xc.T@yc
 ssx=float(np.trace(xx));ssy=float(np.trace(yy));sing=np.linalg.svd(xy,compute_uv=False)
 cka=float(np.square(xy).sum()/np.sqrt(np.square(xx).sum()*np.square(yy).sum()))
 residual=max(0.,ssy-float(sing.sum())**2/ssx)
 r=dict(rows=len(x),raw_paired_cosine=quantiles(np.sum(x*y,1)/(nx*ny)),raw_relative_rms_displacement=float(np.linalg.norm(y-x)/np.linalg.norm(x)),centered_relative_rms_displacement=float(np.linalg.norm(yc-xc)/np.sqrt(ssx)),mean_shift_relative_to_centered_rms=float(np.linalg.norm(my-mx)/np.sqrt(ssx/len(x))),centered_scale_ratio=float(np.sqrt(ssy/ssx)),linear_cka=cka,procrustes_relative_residual=float(np.sqrt(residual/ssy)),optimal_uniform_scale=float(sing.sum()/ssx))
 if neighbors:
  ix=np.linspace(0,len(x)-1,min(1024,len(x)),dtype=np.int64);out={}
  for name,ax,ay in [('raw',x[ix],y[ix]),('centered',xc[ix],yc[ix])]:
   ax=ax/np.linalg.norm(ax,axis=1,keepdims=True);ay=ay/np.linalg.norm(ay,axis=1,keepdims=True)
   sx=ax@ax.T;sy=ay@ay.T;np.fill_diagonal(sx,-np.inf);np.fill_diagonal(sy,-np.inf)
   left=np.argsort(-sx,axis=1,kind='stable')[:,:k];right=np.argsort(-sy,axis=1,kind='stable')[:,:k]
   overlap=np.array([len(set(a)&set(b))/k for a,b in zip(left,right)])
   tri=np.triu_indices(len(ix),1);distance_correlation=float(np.corrcoef((1-sx)[tri],(1-sy)[tri])[0,1])
   out[name]=dict(probe_rows=len(ix),k=k,topk_retention_mean=float(overlap.mean()),topk_retention_quantiles=quantiles(overlap),pairwise_cosine_distance_correlation=distance_correlation)
  r['neighbors']=out
 return r
