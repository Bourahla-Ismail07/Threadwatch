"""Sequential rolling median absolute deviation detector."""
import numpy as np
from .windows import windows

def mad_py(x, w=256, s=1, chunk=50_000):
    x=np.asarray(x)
    nw=(len(x)-w)//s+1 if len(x)>=w and w>0 and s>0 else 0
    if nw<=0: raise ValueError("invalid window/stride or series too short")
    if chunk<1: raise ValueError("chunk must be positive")
    out=np.empty(nw,dtype=np.float32)
    for a in range(0,nw,chunk):
        b=min(a+chunk,nw)
        win=windows(x[a*s:(b-1)*s+w],w,s).astype(np.float64,copy=False)
        med=np.median(win,axis=1)
        mad=np.median(np.abs(win-med[:,None]),axis=1)
        out[a:b]=np.abs(win[:,-1]-med)/(1.4826*mad+1e-12)
    return out
