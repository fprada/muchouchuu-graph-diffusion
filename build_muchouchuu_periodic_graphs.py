#!/usr/bin/env python3
"""
Build periodic MuchoUchuu kNN, optional Gaussian-kNN, and optional radius graphs.

Input
-----
Raw float32 XYZ binary with C-order shape (N, 3), produced by
select_muchouchuu_vpeak_v4.py.

For the standard 6 Gpc/h MuchoUchuu selection:
    L = 6000 h^-1 Mpc
    N = 108,000,000
    number density = 5e-4 h^3 Mpc^-3

The builder uses a block decomposition and one periodic ghost-block layer.
It writes disk-backed directed kNN indices/distances, then constructs
symmetrized SciPy CSR graphs.

Important
---------
This is an HPC-scale calculation. The final symmetrization of a graph with
approximately 1.6 billion directed kNN entries requires a large-memory node.
Gaussian-kNN and radius graphs can be skipped initially with --skip-gaussian --skip-radius.
"""

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

def build_knn_graphs(out,n,k,ip,dp,skip_gaussian):
    idx=np.memmap(ip, mode='r', dtype=np.int32, shape=(n, k)); dist=np.memmap(dp, mode='r', dtype=np.float32, shape=(n, k))
    indptr=np.arange(0,n*k+1,k,dtype=np.int64)
    directed = sparse.csr_matrix(
        (np.ones(n * k, np.float32), idx.reshape(-1), indptr),
        shape=(n, n),
    )
    # Do not call sort_indices() or sum_duplicates() on this directed
    # matrix: its index array is a read-only view of the disk memmap.
    # The directed kNN rows contain exactly k distinct non-self neighbours.
    # maximum(A, A.T) creates a new, writable CSR matrix that can safely be
    # canonicalized below.

    # Self-neighbours were explicitly removed in _process_block().
    # Therefore maximum(A, A.T) cannot introduce diagonal entries.
    # Avoid setdiag(0): inserting absent diagonal positions into CSR
    # changes its sparsity structure and is extremely expensive here.
    g = directed.maximum(directed.T).tocsr(copy=True)
    g.sum_duplicates()
    g.eliminate_zeros()
    g.sort_indices()

    sparse.save_npz(
        out / 'knn_periodic.npz',
        g,
        compressed=False,
    )
    e = int(g.nnz // 2)
    eps = None
    if not skip_gaussian:
        eps = float(np.median(dist[:, -1]))
        w = np.exp(
            -(dist.reshape(-1) ** 2) / np.float32(eps ** 2)
        ).astype(np.float32)
        gd = sparse.csr_matrix(
            (w, idx.reshape(-1), indptr),
            shape=(n, n),
        )
        # gd also uses the read-only neighbour-index memmap. Canonicalize
        # only the newly allocated symmetrized output.
        gg = gd.maximum(gd.T).tocsr(copy=True)
        gg.sum_duplicates()
        gg.eliminate_zeros()
        gg.sort_indices()
        sparse.save_npz(
            out / 'gaussian_knn_periodic.npz',
            gg,
            compressed=False,
        )
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
    r.flush()
    c.flush()
    g = sparse.coo_matrix(
        (np.ones(total, np.float32), (r, c)),
        shape=(n, n),
    ).tocsr()
    g.sum_duplicates()
    g.sort_indices()
    g = g.maximum(g.T).tocsr()
    g.sum_duplicates()
    g.eliminate_zeros()
    g.sort_indices()
    sparse.save_npz(
        out / 'radius_periodic.npz',
        g,
        compressed=False,
    )
    return int(g.nnz // 2)

def main():
    p = argparse.ArgumentParser(
        description=(
            "Build periodic MuchoUchuu graphs from a raw float32 XYZ file."
        )
    )
    p.add_argument("xyz_binary")
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Number of halos. By default infer it from file_size / 12."
        ),
    )
    p.add_argument("--box-size", type=float, default=6000.0)
    p.add_argument(
        "--blocks-per-dim",
        type=int,
        default=24,
        help=(
            "Number of spatial blocks per dimension. The one-layer ghost "
            "construction is valid only when every kth-neighbor distance "
            "is smaller than the block side."
        ),
    )
    p.add_argument("--processes", type=int, default=64)
    p.add_argument("--k", type=int, default=12)
    p.add_argument("--target-degree", type=float, default=14.0)
    p.add_argument(
        "--output-dir",
        default="muchouchuu_graphs_block_parallel",
    )
    p.add_argument(
        "--work-dir",
        default="muchouchuu_graph_work",
    )
    p.add_argument("--skip-gaussian", action="store_true")
    p.add_argument(
        "--reuse-knn-work",
        action="store_true",
        help=(
            "Reuse existing knn_indices.i32 and knn_distances.f32 in "
            "--work-dir and skip the spatial block search. Use this after "
            "a run that completed all blocks but failed during CSR assembly."
        ),
    )
    p.add_argument("--skip-radius", action="store_true")
    p.add_argument("--keep-radius-chunks", action="store_true")
    a = p.parse_args()

    xyz = Path(a.xyz_binary).expanduser().resolve()
    if not xyz.exists():
        raise FileNotFoundError(xyz)

    byte_count = xyz.stat().st_size
    bytes_per_row = 3 * np.dtype(np.float32).itemsize

    if byte_count % bytes_per_row != 0:
        raise ValueError(
            f"Coordinate file size {byte_count:,} is not divisible by "
            f"{bytes_per_row}; expected raw float32 XYZ rows."
        )

    inferred_n = byte_count // bytes_per_row
    if a.n is None:
        a.n = inferred_n
    elif a.n != inferred_n:
        raise ValueError(
            f"--n={a.n:,}, but file size implies N={inferred_n:,}."
        )

    out = Path(a.output_dir).expanduser().resolve()
    work = Path(a.work_dir).expanduser().resolve()
    rdir = work / "radius_chunks"

    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    density=a.n/a.box_size**3; radius=(3*a.target_degree/(4*np.pi*density))**(1/3); side=a.box_size/a.blocks_per_dim
    if radius>=side: raise ValueError('radius must be smaller than block side')
    print('MuchoUchuu block-parallel periodic builder version 1.2 (read-only memmap fix and work reuse)', flush=True)
    print(f'N={a.n:,}\ndensity={density}\nradius={radius}\nblocks={a.blocks_per_dim}^3\nblock side={side}\nprocesses={a.processes}',flush=True)
    ip = work / 'knn_indices.i32'
    dp = work / 'knn_distances.f32'

    if a.reuse_knn_work:
        expected_index_bytes = a.n * a.k * np.dtype(np.int32).itemsize
        expected_distance_bytes = a.n * a.k * np.dtype(np.float32).itemsize

        if not ip.exists() or not dp.exists():
            raise FileNotFoundError(
                "--reuse-knn-work requires existing files "
                f"{ip} and {dp}."
            )

        if ip.stat().st_size != expected_index_bytes:
            raise ValueError(
                f"{ip} has {ip.stat().st_size:,} bytes; expected "
                f"{expected_index_bytes:,}."
            )

        if dp.stat().st_size != expected_distance_bytes:
            raise ValueError(
                f"{dp} has {dp.stat().st_size:,} bytes; expected "
                f"{expected_distance_bytes:,}."
            )

        print(
            "Reusing completed directed kNN work arrays; "
            "skipping all spatial block searches.",
            flush=True,
        )

        distance_memmap = np.memmap(
            dp,
            mode='r',
            dtype=np.float32,
            shape=(a.n, a.k),
        )
        maxk = float(np.max(distance_memmap[:, -1]))
        del distance_memmap
        red = 0
        counts = None
    else:
        op, fp, counts = prepare_blocks(
            xyz,
            a.n,
            a.box_size,
            a.blocks_per_dim,
            work,
        )
        print(
            f'block population min={counts.min()} '
            f'median={int(np.median(counts))} max={counts.max()}',
            flush=True,
        )

        np.memmap(
            ip,
            mode='w+',
            dtype=np.int32,
            shape=(a.n, a.k),
        ).flush()
        np.memmap(
            dp,
            mode='w+',
            dtype=np.float32,
            shape=(a.n, a.k),
        ).flush()

        ctx = mp.get_context('fork')
        init = (
            str(xyz),
            a.n,
            a.box_size,
            a.blocks_per_dim,
            str(op),
            str(fp),
            str(ip),
            str(dp),
            a.k,
            radius,
            str(rdir),
            not a.skip_radius,
        )
        total = a.blocks_per_dim ** 3
        maxk = 0.0
        red = 0
        done = 0

        with ctx.Pool(
            a.processes,
            initializer=_init_worker,
            initargs=init,
            maxtasksperchild=20,
        ) as pool:
            for res in pool.imap_unordered(
                _process_block,
                range(total),
                chunksize=1,
            ):
                done += 1
                maxk = max(maxk, res['max_kth'])
                red += res['radius_edges']

                if done % 10 == 0 or done == total:
                    print(
                        f'Completed {done}/{total}; '
                        f'max kth={maxk:.5f}; '
                        f'radius directed={red:,}',
                        flush=True,
                    )

    print('Building CSR kNN graph...', flush=True)
    edges, eps = build_knn_graphs(
        out,
        a.n,
        a.k,
        ip,
        dp,
        a.skip_gaussian,
    )

    redges=None
    if not a.skip_radius: print('Merging radius chunks...',flush=True); redges=merge_radius(rdir,out,a.n,work)
    meta = {
        'input_xyz_binary': str(xyz),
        'input_bytes': byte_count,
        'input_dtype': 'float32',
        'input_shape': [a.n, 3],
        'n': a.n,
        'box_size_mpc_h': a.box_size,
        'density_h3_mpc_minus3': density,
        'k': a.k,
        'target_degree': a.target_degree,
        'skip_gaussian': a.skip_gaussian,
        'skip_radius': a.skip_radius,
        'reused_knn_work_arrays': a.reuse_knn_work,
        'radius_mpc_h': radius,
        'blocks_per_dim': a.blocks_per_dim,
        'number_of_blocks': a.blocks_per_dim ** 3,
        'block_side_mpc_h': side,
        'processes': a.processes,
        'max_kth_neighbor_distance_mpc_h': maxk,
        'validation_passed': maxk < side,
        'knn_directed_entries': a.n * a.k,
        'knn_undirected_edges': edges,
        'mean_symmetrized_knn_degree': 2.0 * edges / a.n,
        'gaussian_epsilon_mpc_h': eps,
        'radius_undirected_edges': redges,
        'periodic': True,
        'graph_files': {
            'knn': str(out / 'knn_periodic.npz'),
            'gaussian_knn': (
                None if a.skip_gaussian
                else str(out / 'gaussian_knn_periodic.npz')
            ),
            'radius': (
                None if a.skip_radius else str(out / 'radius_periodic.npz')
            ),
        },
        'algorithm': (
            'block-parallel local cKDTree with one periodic ghost-block layer'
        ),
    }
    (out/'parameters.json').write_text(json.dumps(meta,indent=2))
    if not a.keep_radius_chunks and not a.skip_radius:
        for pth in rdir.glob('block_*.npz'): pth.unlink()
    print(f'Finished: {out}',flush=True)
if __name__=='__main__': main()
