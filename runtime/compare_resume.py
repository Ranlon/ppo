"""Compare real uninterrupted and interrupted/resumed PPO numerical state."""
from pathlib import Path
import json
import numpy as np
import torch
import argparse

ROOT=Path(__file__).resolve().parent


def compare(a,b,path):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert set(a)==set(b),path
        for key in a:compare(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)


