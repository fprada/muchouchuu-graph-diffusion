#!/usr/bin/env python3
"""Safe full-box random-walk diffusion sketch and Euclidean metric diagnostics.

Critical safety property: checkpoint files are immutable. Resume loads a
checkpoint read-only and copies it into separate mutable work buffers before
continuing, so no saved state can be overwritten by buffer swapping.
"""
from __future__ import print_function
import argparse, csv, hashlib, json, os, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import linregress
try:
    from sparse_dot_mkl import dot_product_mkl
    HAVE_MKL = True
except Exception:
    dot_product_mkl = None
    HAVE_MKL = False

def periodic_distances(xyz, i_idx, j_idx, box_size):
    d = np.abs(np.asarray(xyz[i_idx], dtype=np.float64) - np.asarray(xyz[j_idx], dtype=np.float64))
    d = np.minimum(d, box_size - d)
    return np.sqrt(np.sum(d*d, axis=1))

def sample_pairs_stratified(xyz, box_size, rng, n_bins, pairs_per_bin, dmin, dmax, max_rounds, outdir):
    pair_file = outdir / 'sampled_pairs.npz'
    if pair_file.exists():
        z = np.load(str(pair_file))
        print('Reusing saved pair sample:', pair_file, flush=True)
        return z['i'], z['j'], z['distance']
    n = len(xyz); edges = np.linspace(dmin, dmax, n_bins+1)
    si=[[] for _ in range(n_bins)]; sj=[[] for _ in range(n_bins)]; sd=[[] for _ in range(n_bins)]
    counts=np.zeros(n_bins,dtype=np.int64); batch=max(1_000_000,n_bins*pairs_per_bin*10)
    for r in range(max_rounds):
        if np.all(counts>=pairs_per_bin): break
        i=rng.randint(0,n,size=batch).astype(np.int64); j=rng.randint(0,n,size=batch).astype(np.int64)
        m=i!=j; i=i[m]; j=j[m]
        dist=periodic_distances(xyz,i,j,box_size); bid=np.searchsorted(edges,dist,side='right')-1
        for b in range(n_bins):
            need=pairs_per_bin-counts[b]
            if need<=0: continue
            found=np.flatnonzero(bid==b)[:need]
            if len(found):
                si[b].append(i[found]); sj[b].append(j[found]); sd[b].append(dist[found]); counts[b]+=len(found)
        print('pair round {} minimum bin count {}'.format(r,int(counts.min())),flush=True)
    if np.any(counts<pairs_per_bin): raise RuntimeError('Pair sampling incomplete: {}'.format(counts.tolist()))
    ia=np.concatenate([np.concatenate(x) for x in si]); ja=np.concatenate([np.concatenate(x) for x in sj]); da=np.concatenate([np.concatenate(x) for x in sd])
    np.savez(str(pair_file),i=ia,j=ja,distance=da)
    return ia,ja,da

def fit_linearity(d,delta,lo,hi,bins):
    e=np.linspace(lo,hi,bins+1); xs=[]; ys=[]
    for a,b in zip(e[:-1],e[1:]):
        m=(d>=a)&(d<b)
        if m.sum()>=50: xs.append(np.median(d[m])); ys.append(np.median(delta[m]))
    x=np.asarray(xs); y=np.asarray(ys)
    if len(x)<5: return (np.nan,)*5
    scale=np.dot(x,y)/max(np.dot(x,x),1e-30); pred=scale*x
    err=np.sqrt(np.mean((y-pred)**2))/max(np.sqrt(np.mean(y*y)),1e-30)
    r2=linregress(x,y).rvalue**2
    return float(x.min()),float(x.max()),float(scale),float(err),float(r2)

def create_random_sketch(path,n,dim,seed,chunk):
    rng=np.random.RandomState(seed); y=np.lib.format.open_memmap(str(path),mode='w+',dtype=np.float32,shape=(n,dim)); scale=np.float32(1.0/np.sqrt(dim))
    for s in range(0,n,chunk):
        q=min(s+chunk,n); block=rng.standard_normal((q-s,dim)).astype(np.float32); block*=scale; y[s:q]=block
        if q==n or q%(10*chunk)==0: print('random sketch rows {}/{}'.format(q,n),flush=True)
    y.flush(); return y

def newest_checkpoint(outdir):
    pat=re.compile(r'state_t(\d{7})\.npy$'); found=[]
    for p in outdir.glob('state_t*.npy'):
        m=pat.match(p.name)
        if m: found.append((int(m.group(1)),p))
    return max(found,key=lambda x:x[0]) if found else None

def copy_array(src,dst,chunk):
    for s in range(0,src.shape[0],chunk):
        q=min(s+chunk,src.shape[0]); dst[s:q]=src[s:q]
    dst.flush()

def save_checkpoint(y,step,outdir,chunk):
    final=outdir/('state_t{:07d}.npy'.format(step)); tmp=outdir/('state_t{:07d}.tmp.npy'.format(step))
    if final.exists():
        print('Checkpoint already exists; preserving:',final,flush=True); return
    print('Saving immutable checkpoint:',final,flush=True)
    z=np.lib.format.open_memmap(str(tmp),mode='w+',dtype=np.float32,shape=y.shape); copy_array(y,z,chunk); del z; os.replace(str(tmp),str(final))

def checksum(path,chunk_bytes=64*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while True:
            block=f.read(chunk_bytes)
            if not block: break
            h.update(block)
    return h.hexdigest()

def append_rows(path,fields,rows):
    exists=path.exists()
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        if not exists: w.writeheader()
        for row in rows: w.writerow(row)

def completed_times(path):
    if not path.exists(): return set()
    return set(pd.read_csv(str(path))['diffusion_time'].astype(int).tolist())

def diagnostics(graph,step,y,i,j,physical,windows,fit_bins,cmin,cmax,cbins):
    delta=np.linalg.norm(np.asarray(y[i],dtype=np.float64)-np.asarray(y[j],dtype=np.float64),axis=1)
    delta/=max(np.sqrt(np.mean(delta*delta)),1e-30)
    dr=[]
    for k in range(0,len(windows),2):
        lo,hi=windows[k],windows[k+1]; xmin,xmax,scale,err,r2=fit_linearity(physical,delta,lo,hi,fit_bins)
        dr.append(dict(graph=graph,diffusion_time=step,requested_window_min=lo,requested_window_max=hi,window_min_mpc_h=xmin,window_max_mpc_h=xmax,linear_scale=scale,relative_rmse=err,linear_r2=r2))
    cr=[]; edges=np.linspace(cmin,cmax,cbins+1)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(physical>=lo)&(physical<hi)
        if np.any(m):
            cr.append(dict(graph=graph,diffusion_time=step,distance_bin_center_mpc_h=0.5*(lo+hi),pair_count=int(m.sum()),median_diffusion_distance=float(np.median(delta[m])),q16=float(np.quantile(delta[m],.16)),q84=float(np.quantile(delta[m],.84))))
    return dr,cr

class Multiplier:
    def __init__(self,A,backend):
        self.A=A; self.backend=backend
        if backend=='mkl' and not HAVE_MKL: raise RuntimeError('sparse_dot_mkl is unavailable')
    def multiply(self,x,out):
        if self.backend=='mkl':
            out.fill(0.0); result=dot_product_mkl(self.A,x,cast=False,out=out,out_scalar=0.0)
            if result is not out: out[:]=result
        else: out[:]=self.A@x

def main(a):
    outdir=Path(a.output_dir).expanduser().resolve(); outdir.mkdir(parents=True,exist_ok=True)
    xyz=np.memmap(str(Path(a.xyz_binary).resolve()),mode='r',dtype=np.float32,shape=(a.n,3))
    rng=np.random.RandomState(a.seed+1)
    i,j,physical=sample_pairs_stratified(xyz,a.box_size,rng,a.pair_distance_bins,a.pairs_per_bin,a.curve_distance_min,a.curve_distance_max,a.max_pair_rounds,outdir)
    graph_path=Path(a.graph).resolve(); graph_name=graph_path.stem
    print('Loading graph:',graph_path,flush=True)
    A=sparse.load_npz(str(graph_path)).tocsr().astype(np.float32,copy=False); A.sort_indices()
    if A.shape != (a.n,a.n): raise ValueError('Graph shape {} does not match N={}'.format(A.shape,a.n))
    degree=np.asarray(A.sum(axis=1)).ravel().astype(np.float32); degree=np.maximum(degree,np.float32(1e-20))
    diag_path=outdir/'euclidean_diagnostics.csv'; curve_path=outdir/'diffusion_distance_vs_physical_separation.csv'
    done=completed_times(diag_path); times=sorted(set(a.times)); tmax=max(times)
    ck=newest_checkpoint(outdir) if a.resume else None
    if ck:
        start,checkpoint_path=ck; print('Resuming safely from immutable checkpoint t={}: {}'.format(start,checkpoint_path),flush=True)
        checkpoint=np.load(str(checkpoint_path),mmap_mode='r')
    else:
        start=0; checkpoint_path=outdir/'state_t0000000.npy'
        if not checkpoint_path.exists(): create_random_sketch(checkpoint_path,a.n,a.sketch_dim,a.seed,a.io_chunk_rows)
        checkpoint=np.load(str(checkpoint_path),mmap_mode='r')
    if checkpoint.shape != (a.n,a.sketch_dim): raise ValueError('Checkpoint shape mismatch: {}'.format(checkpoint.shape))
    src=np.lib.format.open_memmap(str(outdir/'work_buffer_a.npy'),mode='w+',dtype=np.float32,shape=checkpoint.shape)
    dst=np.lib.format.open_memmap(str(outdir/'work_buffer_b.npy'),mode='w+',dtype=np.float32,shape=checkpoint.shape)
    print('Copying checkpoint into mutable work buffer...',flush=True); copy_array(checkpoint,src,a.io_chunk_rows); del checkpoint
    backend=a.backend
    if backend=='auto': backend='mkl' if HAVE_MKL else 'scipy'
    mult=Multiplier(A,backend); print('Using backend:',backend,flush=True)
    meta=vars(a).copy(); meta.update(graph_name=graph_name,selected_backend=backend,mkl_available=HAVE_MKL,start_step=start,resume_checkpoint=str(checkpoint_path),resume_checkpoint_sha256=checksum(checkpoint_path))
    (outdir/'metadata.json').write_text(json.dumps(meta,indent=2))
    df=['graph','diffusion_time','requested_window_min','requested_window_max','window_min_mpc_h','window_max_mpc_h','linear_scale','relative_rmse','linear_r2']
    cf=['graph','diffusion_time','distance_bin_center_mpc_h','pair_count','median_diffusion_distance','q16','q84']
    t0=time.time()
    for step in range(start+1,tmax+1):
        s0=time.time(); mult.multiply(src,dst); dst/=degree[:,None]; src,dst=dst,src
        if step%a.progress_every==0 or step in times:
            mean=(time.time()-t0)/(step-start); rem=mean*(tmax-step)
            print('t={} step_seconds={:.3f} mean_seconds={:.3f} estimated_remaining_seconds={:.1f}'.format(step,time.time()-s0,mean,rem),flush=True)
        if step in times and step not in done:
            dr,cr=diagnostics(graph_name,step,src,i,j,physical,a.fit_windows,a.fit_bins,a.curve_distance_min,a.curve_distance_max,a.pair_distance_bins)
            append_rows(diag_path,df,dr); append_rows(curve_path,cf,cr); done.add(step); print('Wrote diagnostics for t={}'.format(step),flush=True)
        if (a.checkpoint_every>0 and step%a.checkpoint_every==0) or step==tmax: save_checkpoint(src,step,outdir,a.io_chunk_rows)
    print('Finished:',outdir,flush=True)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('xyz_binary'); p.add_argument('n',type=int); p.add_argument('graph')
    p.add_argument('--box-size',type=float,default=4000.0); p.add_argument('--output-dir',default='biguchuu_diffusion_safe')
    p.add_argument('--times',nargs='+',type=int,default=[256,512,1024]); p.add_argument('--sketch-dim',type=int,default=32)
    p.add_argument('--backend',choices=['auto','mkl','scipy'],default='auto'); p.add_argument('--checkpoint-every',type=int,default=256); p.add_argument('--resume',action='store_true')
    p.add_argument('--progress-every',type=int,default=16); p.add_argument('--io-chunk-rows',type=int,default=500000)
    p.add_argument('--pair-distance-bins',type=int,default=70); p.add_argument('--pairs-per-bin',type=int,default=5000); p.add_argument('--curve-distance-min',type=float,default=50.0); p.add_argument('--curve-distance-max',type=float,default=1200.0); p.add_argument('--max-pair-rounds',type=int,default=2000)
    p.add_argument('--fit-windows',nargs='+',type=float,default=[75,150,150,300,300,600,600,1200]); p.add_argument('--fit-bins',type=int,default=16); p.add_argument('--seed',type=int,default=12345)
    a=p.parse_args()
    if len(a.fit_windows)%2: p.error('--fit-windows requires min/max pairs')
    return a
if __name__=='__main__': main(parse_args())
