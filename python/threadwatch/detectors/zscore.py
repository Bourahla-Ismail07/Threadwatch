"""Sequential rolling sample z-score detector."""
import numpy as np
from .windows import windows

def zscore_py(x, w=256, s=1, chunk=100_000):
    x = np.asarray(x)
    if chunk < 1: raise ValueError("chunk must be positive")
    nw = (len(x)-w)//s+1 if len(x)>=w and w>0 and s>0 else 0
    if nw <= 0: raise ValueError("invalid window/stride or series too short")
    out = np.empty(nw, dtype=np.float32)
    for a in range(0, nw, chunk):
        b = min(a+chunk, nw)
        win = windows(x[a*s:(b-1)*s+w], w, s).astype(np.float64, copy=False)
        mu = win.mean(axis=1)
        sd = win.std(axis=1, ddof=1) if w > 1 else np.zeros(len(win))
        out[a:b] = np.divide(np.abs(win[:,-1]-mu), sd, out=np.zeros_like(mu), where=sd>1e-12)
    return out
