#!/usr/bin/env python3
"""Compute finite-difference effective spectral dimension from return probability."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main(a):
    d=pd.read_csv(a.input_csv).sort_values('diffusion_time').drop_duplicates('diffusion_time',keep='last').reset_index(drop=True)
    required={'diffusion_time','mean_return_probability'}; missing=required-set(d.columns)
    if missing: raise ValueError('Missing columns: {}'.format(sorted(missing)))
    t=d.diffusion_time.to_numpy(float); p=d.mean_return_probability.to_numpy(float)
    if len(t)<3 or np.any(t<=0) or np.any(p<=0): raise ValueError('Need >=3 positive times and probabilities')
    ds=-2*np.gradient(np.log(p),np.log(t),edge_order=2)
    if 'mean_return_probability_se' in d.columns: sigma_log=d.mean_return_probability_se.to_numpy(float)/p
    elif 'relative_trace_se' in d.columns: sigma_log=d.relative_trace_se.to_numpy(float)
    else: sigma_log=np.full_like(p,np.nan)
    err=np.full_like(ds,np.nan)
    if np.all(np.isfinite(sigma_log)):
        for i in range(1,len(t)-1): err[i]=2*np.hypot(sigma_log[i-1],sigma_log[i+1])/(np.log(t[i+1])-np.log(t[i-1]))
        err[0]=2*np.hypot(sigma_log[0],sigma_log[1])/(np.log(t[1])-np.log(t[0]))
        err[-1]=2*np.hypot(sigma_log[-2],sigma_log[-1])/(np.log(t[-1])-np.log(t[-2]))
    d['effective_spectral_dimension']=ds; d['effective_spectral_dimension_se']=err
    d['deviation_from_three']=ds-3.; d['within_one_percent_of_three']=np.abs(ds-3.)<=a.tolerance
    out=Path(a.output_csv); out.parent.mkdir(parents=True,exist_ok=True); d.to_csv(out,index=False)
    print(d[['diffusion_time','mean_return_probability','effective_spectral_dimension','effective_spectral_dimension_se','within_one_percent_of_three']].to_string(index=False)); print('\nWrote:',out)

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('input_csv'); p.add_argument('--output-csv',required=True); p.add_argument('--tolerance',type=float,default=.03); return p.parse_args()
if __name__=='__main__': main(parse_args())
