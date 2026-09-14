#!/usr/bin/env python3
import argparse, hashlib
from pathlib import Path
import numpy as np
from scipy import sparse

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(64*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main(a):
    xyz=Path(a.xyz); graph=Path(a.graph)
    expected=a.n*3*np.dtype(np.float32).itemsize
    print('XYZ:',xyz.resolve()); print('bytes:',xyz.stat().st_size,'expected:',expected)
    if xyz.stat().st_size!=expected: raise ValueError('Coordinate file size mismatch')
    mm=np.memmap(xyz,mode='r',dtype=np.float32,shape=(a.n,3)); print('coordinate min:',np.min(mm,axis=0)); print('coordinate max:',np.max(mm,axis=0))
    A=sparse.load_npz(graph).tocsr(); print('graph shape:',A.shape,'nnz:',A.nnz,'dtype:',A.dtype)
    if A.shape!=(a.n,a.n): raise ValueError('Graph shape mismatch')
    degree=np.diff(A.indptr); print('stored row nnz min/mean/max:',degree.min(),degree.mean(),degree.max()); print('xyz sha256:',sha256(xyz)); print('graph sha256:',sha256(graph))

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('xyz'); p.add_argument('n',type=int); p.add_argument('graph'); return p.parse_args()
if __name__=='__main__': main(parse_args())
