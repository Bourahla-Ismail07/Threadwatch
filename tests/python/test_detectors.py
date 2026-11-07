import numpy as np
from threadwatch.detectors.windows import windows
from threadwatch.detectors.zscore import zscore_py
from threadwatch.detectors.mad import mad_py
from threadwatch.detectors.features import features_py
from threadwatch.detectors.ml import kmeans_score_py, threshold

def test_windows_and_features():
    x=np.arange(8,dtype=np.float32)
    assert windows(x,3,2).shape==(3,3)
    f=features_py(x,3,2)
    np.testing.assert_allclose(f[0],[1,1,0,2])

def test_spike_zscore():
    x=np.zeros(31,dtype=np.float32); x[15]=20
    scores=zscore_py(x,w=7)
    assert scores[15-7+1] > 2

def test_mad_finite_and_kmeans():
    x=np.array([0,0,0,0,10,0,0,0],dtype=np.float32)
    assert np.isfinite(mad_py(x,3)).all()
    F=np.array([[0,0],[1,1]],dtype=np.float32)
    np.testing.assert_allclose(kmeans_score_py(F,F,np.ones(2)),[0,0])
    assert threshold(np.array([1,2,3]),rho=.1)>=2
