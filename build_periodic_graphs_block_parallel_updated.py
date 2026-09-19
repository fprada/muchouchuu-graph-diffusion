#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

_CFG = {}

def _init_worker(xyz_path,n,L,B,order_path,offsets_path,knn_i_path,knn_d_path,k,radius,radius_dir,build_radius):
    os.environ['OMP_NUM_THREADS']='1'; os.environ['MKL_NUM_THREADS']='1'; os.environ['OPENBLAS_NUM_THREADS']='1'
    _CFG.update(
        xyz=np.memmap(xyz_path, mode='r', dtype=np.float32, shape=(n, 3)),
        order=np.memmap(order_path, mode='r', dtype=np.int64, shape=(n,)),
        offsets=np.memmap(offsets_path, mode='r', dtype=np.int64, shape=(B**3 + 1,)),
        knn_i=np.memmap(knn_i_path, mode='r+', dtype=np.int32, shape=(n, k)),
        knn_d=np.memmap(knn_d_path, mode='r+', dtype=np.float32, shape=(n, k)),
        n=n,L=float(L),B=int(B),side=float(L/B),k=int(k),radius=float(radius),
        radius_dir=Path(radius_dir),build_radius=bool(build_radius))

def _bid(ix,iy,iz,B): return (ix*B+iy)*B+iz

def _decode(b,B):
    ix=b//(B*B); rem=b%(B*B); iy=rem//B; iz=rem%B; return ix,iy,iz

def _members(b):
    a=int(_CFG['offsets'][b]); c=int(_CFG['offsets'][b+1]); return np.asarray(_CFG['order'][a:c])

def _query(tree, pts, k):
    try: return tree.query(pts,k=k,workers=1)
    except TypeError:
        try: return tree.query(pts,k=k,n_jobs=1)
        except TypeError: return tree.query(pts,k=k)

def _candidates(ix,iy,iz):
    xyz=_CFG['xyz']; B=_CFG['B']; L=_CFG['L']; pp=[]; ii=[]
    for dx in (-1,0,1):
        ux=ix+dx; bx=ux%B; sx=(ux//B)*L
        for dy in (-1,0,1):
            uy=iy+dy; by=uy%B; sy=(uy//B)*L
            for dz in (-1,0,1):
                uz=iz+dz; bz=uz%B; sz=(uz//B)*L
                ids=_members(_bid(bx,by,bz,B))
                if len(ids)==0: continue
                p=np.asarray(xyz[ids],dtype=np.float64)
                p[:,0]+=sx; p[:,1]+=sy; p[:,2]+=sz
                pp.append(p); ii.append(ids.astype(np.int32,copy=False))
    return np.concatenate(pp), np.concatenate(ii)

def _process_block(b):
    xyz=_CFG['xyz']; k=_CFG['k']; side=_CFG['side']; radius=_CFG['radius']; B=_CFG['B']
    ix,iy,iz=_decode(b,B); central=_members(b).astype(np.int32,copy=False)
    if len(central)==0: return {'block':b,'central':0,'max_kth':0.0,'radius_edges':0}
    cp=np.asarray(xyz[central],dtype=np.float64)
    candp,candid=_candidates(ix,iy,iz)
    tree=cKDTree(candp,balanced_tree=True,compact_nodes=True)
    d,loc=_query(tree,cp,k+1); ids=candid[loc]
    out_i=np.empty((len(central),k),np.int32); out_d=np.empty((len(central),k),np.float32)
    for r in range(len(central)):
        keep=ids[r]!=central[r]
        ri=ids[r][keep][:k]; rd=d[r][keep][:k]
        if len(ri)!=k: raise RuntimeError(f'block {b}: insufficient non-self neighbors')
        out_i[r]=ri; out_d[r]=rd
    maxk=float(out_d[:,-1].max())
    if maxk>=side: raise RuntimeError(f'block {b}: kth distance {maxk} >= block side {side}; reduce blocks-per-dim')
    _CFG['knn_i'][central]=out_i; _CFG['knn_d'][central]=out_d
    redges=0
    if _CFG['build_radius']:
        ctree=cKDTree(cp,balanced_tree=True,compact_nodes=True)
        coo=tree.sparse_distance_matrix(ctree,radius,output_type='coo_matrix')
        rows=central[coo.col].astype(np.int32,copy=False); cols=candid[coo.row].astype(np.int32,copy=False)
        keep=rows!=cols; rows=rows[keep]; cols=cols[keep]; redges=len(rows)
        np.savez(_CFG['radius_dir']/f'block_{b:06d}.npz',rows=rows,cols=cols)
    return {'block':b,'central':len(central),'max_kth':maxk,'radius_edges':redges}

def prepare_blocks(xyz_path,n,L,B,work):
    xyz=np.memmap(xyz_path, mode='r', dtype=np.float32, shape=(n, 3))
    if np.any(xyz<0) or np.any(xyz>=L): raise ValueError('coordinates must be in [0,L)')
    side=L/B; cell=np.floor(xyz/side).astype(np.int16); np.minimum(cell,B-1,out=cell)
    bid=((cell[:,0].astype(np.int32)*B+cell[:,1].astype(np.int32))*B+cell[:,2].astype(np.int32))
    counts=np.bincount(bid,minlength=B**3).astype(np.int64)
    print('Sorting points by block...',flush=True)
    order=np.argsort(bid,kind='stable').astype(np.int64,copy=False)
    op=work/'block_order.i64'; mm=np.memmap(op, mode='w+', dtype=np.int64, shape=(n,)); mm[:]=order; mm.flush(); del mm
    offsets=np.empty(B**3+1,np.int64); offsets[0]=0; np.cumsum(counts,out=offsets[1:])
    fp=work/'block_offsets.i64'; offsets.tofile(fp)
    return op,fp,counts

def build_knn_graphs(out,n,k,ip,dp):
    idx=np.memmap(ip, mode='r', dtype=np.int32, shape=(n, k)); dist=np.memmap(dp, mode='r', dtype=np.float32, shape=(n, k))
    indptr=np.arange(0,n*k+1,k,dtype=np.int64)
    directed=sparse.csr_matrix((np.ones(n*k,np.float32),idx.reshape(-1),indptr),shape=(n,n))
    g=directed.maximum(directed.T).tocsr(); g.setdiag(0); g.eliminate_zeros(); sparse.save_npz(out/'knn_periodic.npz',g,compressed=False); e=int(g.nnz//2)
    eps=float(np.median(dist[:,-1])); w=np.exp(-(dist.reshape(-1)**2)/np.float32(eps**2)).astype(np.float32)
    gd=sparse.csr_matrix((w,idx.reshape(-1),indptr),shape=(n,n)); gg=gd.maximum(gd.T).tocsr(); gg.setdiag(0); gg.eliminate_zeros(); sparse.save_npz(out/'gaussian_knn_periodic.npz',gg,compressed=False)
    return e,eps

def merge_radius(radius_dir,out,n,work):
    files=sorted(radius_dir.glob('block_*.npz')); counts=[]
    for p in files:
        with np.load(p) as z: counts.append(len(z['rows']))
    total=int(np.sum(counts,dtype=np.int64)); rp=work/'radius_rows.i32'; cp=work/'radius_cols.i32'
    r=np.memmap(rp, mode='w+', dtype=np.int32, shape=(total,)); c=np.memmap(cp, mode='w+', dtype=np.int32, shape=(total,)); pos=0
    for j,p in enumerate(files):
        with np.load(p) as z:
            m=len(z['rows']); r[pos:pos+m]=z['rows']; c[pos:pos+m]=z['cols']; pos+=m
        if (j+1)%100==0: print(f'Merged {j+1}/{len(files)} radius chunks',flush=True)
    r.flush(); c.flush(); g=sparse.coo_matrix((np.ones(total,np.float32),(r,c)),shape=(n,n)).tocsr(); g=g.maximum(g.T).tocsr(); g.setdiag(0); g.eliminate_zeros(); sparse.save_npz(out/'radius_periodic.npz',g,compressed=False); return int(g.nnz//2)

def main():
    p=argparse.ArgumentParser(); p.add_argument('xyz_binary'); p.add_argument('n',type=int); p.add_argument('--box-size',type=float,default=4000.0); p.add_argument('--blocks-per-dim',type=int,default=12); p.add_argument('--processes',type=int,default=64); p.add_argument('--k',type=int,default=12); p.add_argument('--target-degree',type=float,default=14.0); p.add_argument('--output-dir',default='biguchuu_graphs_block_parallel'); p.add_argument('--work-dir',default='biguchuu_graph_work'); p.add_argument('--skip-radius',action='store_true'); p.add_argument('--keep-radius-chunks',action='store_true'); a=p.parse_args()
    xyz=Path(a.xyz_binary).resolve(); out=Path(a.output_dir).resolve(); work=Path(a.work_dir).resolve(); rdir=work/'radius_chunks'; out.mkdir(parents=True,exist_ok=True); work.mkdir(parents=True,exist_ok=True); rdir.mkdir(parents=True,exist_ok=True)
    expected=a.n*3*4
    if xyz.stat().st_size!=expected: raise ValueError(f'coordinate file bytes={xyz.stat().st_size}, expected={expected}')
    density=a.n/a.box_size**3; radius=(3*a.target_degree/(4*np.pi*density))**(1/3); side=a.box_size/a.blocks_per_dim
    if radius>=side: raise ValueError('radius must be smaller than block side')
    print('Block-parallel builder version 1.1 (memmap keyword fix)', flush=True)
    print(f'N={a.n:,}\ndensity={density}\nradius={radius}\nblocks={a.blocks_per_dim}^3\nblock side={side}\nprocesses={a.processes}',flush=True)
    op,fp,counts=prepare_blocks(xyz,a.n,a.box_size,a.blocks_per_dim,work); print(f'block population min={counts.min()} median={int(np.median(counts))} max={counts.max()}',flush=True)
    ip=work/'knn_indices.i32'; dp=work/'knn_distances.f32'; np.memmap(ip, mode='w+', dtype=np.int32, shape=(a.n, a.k)).flush(); np.memmap(dp, mode='w+', dtype=np.float32, shape=(a.n, a.k)).flush()
    ctx=mp.get_context('fork'); init=(str(xyz),a.n,a.box_size,a.blocks_per_dim,str(op),str(fp),str(ip),str(dp),a.k,radius,str(rdir),not a.skip_radius)
    total=a.blocks_per_dim**3; maxk=0.0; red=0; done=0
    with ctx.Pool(a.processes,initializer=_init_worker,initargs=init,maxtasksperchild=20) as pool:
        for res in pool.imap_unordered(_process_block,range(total),chunksize=1):
            done+=1; maxk=max(maxk,res['max_kth']); red+=res['radius_edges']
            if done%10==0 or done==total: print(f'Completed {done}/{total}; max kth={maxk:.5f}; radius directed={red:,}',flush=True)
    print('Building CSR kNN graphs...',flush=True); edges,eps=build_knn_graphs(out,a.n,a.k,ip,dp)
    redges=None
    if not a.skip_radius: print('Merging radius chunks...',flush=True); redges=merge_radius(rdir,out,a.n,work)
    meta={'n':a.n,'box_size_mpc_h':a.box_size,'density_h3_mpc_minus3':density,'k':a.k,'target_degree':a.target_degree,'radius_mpc_h':radius,'blocks_per_dim':a.blocks_per_dim,'block_side_mpc_h':side,'processes':a.processes,'max_kth_neighbor_distance_mpc_h':maxk,'validation_passed':maxk<side,'knn_undirected_edges':edges,'gaussian_epsilon_mpc_h':eps,'radius_undirected_edges':redges,'periodic':True,'algorithm':'block-parallel local cKDTree with one periodic ghost-block layer'}
    (out/'parameters.json').write_text(json.dumps(meta,indent=2))
    if not a.keep_radius_chunks and not a.skip_radius:
        for pth in rdir.glob('block_*.npz'): pth.unlink()
    print(f'Finished: {out}',flush=True)
if __name__=='__main__': main()
