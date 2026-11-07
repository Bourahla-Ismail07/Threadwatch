"""Window feature extraction: mean, sample standard deviation, min, max."""
import numpy as np
from .windows import windows

def features_py(x,w=256,s=1,chunk=100_000):
    x=np.asarray(x)
    nw=(len(x)-w)//s+1 if len(x)>=w and w>0 and s>0 else 0
    if nw<=0: raise ValueError("invalid window/stride or series too short")
    if chunk<1: raise ValueError("chunk must be positive")
    out=np.empty((nw,4),dtype=np.float32)
    for a in range(0,nw,chunk):
        b=min(a+chunk,nw)
        win=windows(x[a*s:(b-1)*s+w],w,s).astype(np.float64,copy=False)
        out[a:b,0]=win.mean(axis=1)
        out[a:b,1]=win.std(axis=1,ddof=1) if w>1 else 0
        out[a:b,2]=win.min(axis=1)
        out[a:b,3]=win.max(axis=1)
    return out
