"""Point-wise and event-wise anomaly detection metrics."""
import numpy as np

def prf(pred, labels):
    pred=np.asarray(pred,dtype=bool); labels=np.asarray(labels,dtype=bool)
    if pred.shape!=labels.shape: raise ValueError("pred and labels must have the same shape")
    tp=int(np.count_nonzero(pred&labels)); fp=int(np.count_nonzero(pred&~labels)); fn=int(np.count_nonzero(~pred&labels))
    precision=tp/(tp+fp) if tp+fp else 0.0
    recall=tp/(tp+fn) if tp+fn else 0.0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
    return precision,recall,f1

def event_recall(pred,labels,tau=0):
    pred=np.asarray(pred,dtype=bool); labels=np.asarray(labels,dtype=bool)
    if pred.shape!=labels.shape: raise ValueError("pred and labels must have the same shape")
    if tau<0: raise ValueError("tau must be non-negative")
    starts=np.flatnonzero(labels & np.r_[True,~labels[:-1]])
    ends=np.flatnonzero(labels & np.r_[~labels[1:],True])
    if len(starts)==0:return 0.0
    found=0
    for start,end in zip(starts,ends):
        lo=max(0,start-tau); hi=min(len(pred),end+tau+1)
        found += bool(np.any(pred[lo:hi]))
    return found/len(starts)

def evaluate(flags,labels,w=256,s=1,tau=None):
    flags=np.asarray(flags,dtype=bool); labels=np.asarray(labels,dtype=bool)
    if w<1 or s<1: raise ValueError("w and s must be positive")
    if len(flags)!=(len(labels)-w)//s+1: raise ValueError("flag count does not match labels/window parameters")
    t=np.arange(len(flags))*s+(w-1)
    pred=np.zeros(len(labels),dtype=bool); pred[t[flags]]=True
    p,r,f1=prf(pred,labels)
    return {"precision":p,"recall":r,"f1":f1,"event_recall":event_recall(pred,labels,w//2 if tau is None else tau)}
