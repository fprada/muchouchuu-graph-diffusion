#!/usr/bin/env python3
import argparse,time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import sparse
try:
    from sparse_dot_mkl import dot_product_mkl
    HAVE_MKL=True
except ImportError:
    dot_product_mkl=None; HAVE_MKL=False

def periodic_delta(x,c,L):
    d=x-c; d-=L*np.rint(d/L); return d

def select_cube(xyz,c,side,L,chunk=1000000):
    out=[]; h=side/2
    for s in range(0,xyz.shape[0],chunk):
        e=min(s+chunk,xyz.shape[0]); p=np.asarray(xyz[s:e],float)
        m=np.all(np.abs(periodic_delta(p,c[None,:],L))<=h,axis=1)
        if np.any(m): out.append(s+np.flatnonzero(m))
    if not out: raise RuntimeError('No halos in subvolume')
    return np.concatenate(out)

def choose_start(xyz,idx,c,L):
    d=periodic_delta(np.asarray(xyz[idx],float),c[None,:],L)
    return int(idx[np.argmin(np.sum(d*d,axis=1))])

def propagate(G,deg,start,times,backend,progress):
    n=G.shape[0]; times=sorted(set(times)); mx=max(times)
    a=np.zeros(n,np.float32); b=np.zeros(n,np.float32); a[start]=1
    saved={}; t0=time.time()
    for t in range(1,mx+1):
        if backend=='mkl':
            b.fill(0); r=dot_product_mkl(G,a,cast=False,out=b,out_scalar=0.0)
            if r is not b: b[:]=r
        else: b[:]=G@a
        b/=deg; a,b=b,a
        if t in times: saved[t]=a.copy()
        if t%progress==0 or t in times:
            print(f't={t} mean_step={(time.time()-t0)/t:.4f}s',flush=True)
    return saved

def local_edges(G,idx,xy,max_edges,rng):
    mp={int(g):i for i,g in enumerate(idx)}; seg=[]
    for li,gi in enumerate(idx):
        for gj in G.indices[G.indptr[gi]:G.indptr[gi+1]]:
            lj=mp.get(int(gj))
            if lj is not None and lj>li: seg.append([xy[li],xy[lj]])
    if len(seg)>max_edges:
        keep=rng.choice(len(seg),max_edges,replace=False); seg=[seg[i] for i in keep]
    return np.asarray(seg,float)

def logvals(v):
    p=v[v>0]
    if len(p)==0:return np.full_like(v,np.nan)
    lo=np.quantile(p,.02); hi=p.max(); return np.log10(np.clip(v,lo,hi))

def main(a):
    xyz=np.memmap(a.xyz_binary,'r',np.float32,shape=(a.n,3))
    G=sparse.load_npz(a.graph).tocsr().astype(np.float32,copy=False); G.sort_indices()
    deg=np.maximum(np.asarray(G.sum(axis=1)).ravel().astype(np.float32),1e-20)
    c=np.array(a.center if a.center else [a.box_size/2]*3,float)
    idx=select_cube(xyz,c,a.subvolume_size,a.box_size,a.io_chunk_rows)
    print(f'Selected {len(idx):,} halos',flush=True)
    start=choose_start(xyz,idx,c,a.box_size); print('Starting node:',start,flush=True)
    backend=a.backend if a.backend!='auto' else ('mkl' if HAVE_MKL else 'scipy')
    if backend=='mkl' and not HAVE_MKL: raise RuntimeError('sparse_dot_mkl unavailable')
    states=propagate(G,deg,start,[a.early_time,a.late_time],backend,a.progress_every)
    rng=np.random.default_rng(a.seed)
    coords=np.asarray(xyz[idx],float); xy=periodic_delta(coords,c[None,:],a.box_size)[:,:2]
    disp=np.arange(len(idx)) if len(idx)<=a.max_nodes else rng.choice(len(idx),a.max_nodes,replace=False)
    gdisp=idx[disp]; xyd=xy[disp]
    sxy=xy[np.where(idx==start)[0][0]]
    seg=local_edges(G,idx,xy,a.max_edges,rng)
    le, ll = logvals(states[a.early_time][gdisp]), logvals(states[a.late_time][gdisp])
    finite=np.r_[le[np.isfinite(le)],ll[np.isfinite(ll)]]; vmin,vmax=finite.min(),finite.max()
    fig,ax=plt.subplots(2,2,figsize=(13.5,11),constrained_layout=True)
    ax[0,0].scatter(xyd[:,0],xyd[:,1],s=1.3,alpha=.55,rasterized=True); ax[0,0].set_title('(a) Halo distribution')
    if len(seg): ax[0,1].add_collection(LineCollection(seg,linewidths=.22,alpha=.16))
    ax[0,1].scatter(xyd[:,0],xyd[:,1],s=1,alpha=.45,rasterized=True); ax[0,1].scatter(*sxy,s=55,marker='*',label='Starting node'); ax[0,1].legend(frameon=False); ax[0,1].set_title('(b) Periodic kNN graph')
    sc=ax[1,0].scatter(xyd[:,0],xyd[:,1],c=le,s=4,vmin=vmin,vmax=vmax,rasterized=True); ax[1,0].scatter(*sxy,s=45,marker='*'); ax[1,0].set_title(rf'(c) Early diffusion, $t={a.early_time}$')
    sc=ax[1,1].scatter(xyd[:,0],xyd[:,1],c=ll,s=4,vmin=vmin,vmax=vmax,rasterized=True); ax[1,1].scatter(*sxy,s=45,marker='*'); ax[1,1].set_title(rf'(d) Later diffusion, $t={a.late_time}$')
    h=a.subvolume_size/2
    for q in ax.ravel():
        q.set(xlim=(-h,h),ylim=(-h,h),xlabel=r'$\Delta x\,[h^{-1}\,\mathrm{Mpc}]$',ylabel=r'$\Delta y\,[h^{-1}\,\mathrm{Mpc}]$'); q.set_aspect('equal'); q.tick_params(direction='in',top=True,right=True)
    cb=fig.colorbar(sc,ax=[ax[1,0],ax[1,1]],shrink=.86,pad=.02); cb.set_label(r'$\log_{10}P(i,t)$')
    fig.suptitle('Cosmic-web graph and diffusion on the BigUchuu halo network',fontsize=16)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=250); plt.close(fig); print('Wrote:',out)

def parse():
    p=argparse.ArgumentParser(); p.add_argument('xyz_binary'); p.add_argument('n',type=int); p.add_argument('graph')
    p.add_argument('--box-size',type=float,default=4000.); p.add_argument('--center',nargs=3,type=float); p.add_argument('--subvolume-size',type=float,default=300.)
    p.add_argument('--early-time',type=int,default=128); p.add_argument('--late-time',type=int,default=4096); p.add_argument('--backend',choices=['auto','mkl','scipy'],default='auto')
    p.add_argument('--max-nodes',type=int,default=120000); p.add_argument('--max-edges',type=int,default=180000); p.add_argument('--io-chunk-rows',type=int,default=1000000)
    p.add_argument('--progress-every',type=int,default=128); p.add_argument('--seed',type=int,default=12345); p.add_argument('--output',default='figure1_cosmic_web_diffusion.png'); return p.parse_args()
if __name__=='__main__': main(parse())
