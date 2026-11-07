"""Small, repeatable sequential timing helper."""
from time import perf_counter
import statistics

def time_call(fn,*args,warmup=1,repeats=3,**kwargs):
    if warmup<0 or repeats<1: raise ValueError("warmup must be >=0 and repeats >=1")
    for _ in range(warmup): fn(*args,**kwargs)
    samples=[]
    result=None
    for _ in range(repeats):
        start=perf_counter(); result=fn(*args,**kwargs); samples.append((perf_counter()-start)*1000)
    return result, {"runs_ms":samples,"median_ms":statistics.median(samples),"min_ms":min(samples),"max_ms":max(samples)}
