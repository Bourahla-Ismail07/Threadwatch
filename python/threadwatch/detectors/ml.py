"""Fit and score sequential scikit-learn anomaly models."""
from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

def fit_models(F_sample, k=8, seed=0):
    F=np.asarray(F_sample,dtype=np.float32)
    if F.ndim!=2 or len(F)==0: raise ValueError("F_sample must be a non-empty 2D array")
    k=min(int(k),len(F))
    if k<1: raise ValueError("k must be positive")
    std=F.astype(np.float64).std(axis=0)
    scale=np.divide(1.0,std,out=np.ones_like(std),where=std>1e-12).astype(np.float32)
    km=KMeans(n_clusters=k,n_init=10,random_state=seed).fit(F*scale)
    iso=IsolationForest(n_estimators=100,max_samples=min(256,len(F)),random_state=seed,n_jobs=1).fit(F)
    return km.cluster_centers_.astype(np.float32)/scale,scale,iso

def kmeans_score_py(F,centroids,scale,chunk=50_000):
    F=np.asarray(F,dtype=np.float32); C=np.asarray(centroids,dtype=np.float32); scale=np.asarray(scale,dtype=np.float32)
    if F.ndim!=2 or C.ndim!=2 or F.shape[1]!=C.shape[1] or scale.shape!=(F.shape[1],):
        raise ValueError("incompatible feature, centroid, or scale shapes")
    if chunk<1: raise ValueError("chunk must be positive")
    out=np.empty(len(F),dtype=np.float32)
    for a in range(0,len(F),chunk):
        b=min(a+chunk,len(F))
        d=(F[a:b,None,:]-C[None,:,:])*scale
        out[a:b]=np.sqrt(np.min(np.sum(d*d,axis=2),axis=1))
    return out

def isolation_score_py(F,iso,chunk=100_000):
    F=np.asarray(F,dtype=np.float32)
    return np.concatenate([-iso.score_samples(F[a:a+chunk]) for a in range(0,len(F),chunk)]).astype(np.float32) if len(F) else np.empty(0,dtype=np.float32)

def threshold(scores_sample,rho=0.001):
    if not 0<rho<1: raise ValueError("rho must be between 0 and 1")
    scores=np.asarray(scores_sample)
    if scores.size==0: raise ValueError("scores_sample cannot be empty")
    return float(np.quantile(scores,1-rho))
