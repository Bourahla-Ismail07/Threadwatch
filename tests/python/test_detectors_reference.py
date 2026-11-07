"""Reference checks added in review: detectors against pandas/scipy, chunk invariance, acceptance spike."""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import median_abs_deviation

from threadwatch.detectors.features import features_py
from threadwatch.detectors.mad import mad_py
from threadwatch.detectors.ml import fit_models, kmeans_score_py, threshold
from threadwatch.detectors.zscore import zscore_py
from threadwatch.metrics import evaluate, event_recall, prf


def test_spike_acceptance_hand_made_series():
    """README: score above 6 at the spike and below 3 elsewhere."""
    x = np.sin(np.arange(400) * 0.2).astype(np.float32)
    x[200] += 10
    w = 64
    z = zscore_py(x, w=w)
    i = 200 - (w - 1)                       # window whose last sample is the spike
    assert z[i] > 6
    assert np.delete(z, i).max() < 3


def test_zscore_matches_pandas_rolling():
    x = np.random.default_rng(1).normal(5, 2, 3000)
    r = pd.Series(x).rolling(64)
    ref = (np.abs(x - r.mean()) / r.std(ddof=1)).to_numpy()[63:]
    np.testing.assert_allclose(zscore_py(x, w=64), ref, atol=1e-5)


def test_mad_matches_scipy_with_stride():
    x = np.random.default_rng(2).normal(5, 2, 3000)
    w, s = 64, 3
    got = mad_py(x, w=w, s=s)
    ref = [abs(x[i + w - 1] - np.median(x[i:i + w])) /
           (1.4826 * median_abs_deviation(x[i:i + w]) + 1e-12) for i in range(0, len(x) - w + 1, s)]
    np.testing.assert_allclose(got, ref, rtol=1e-4)


def test_chunking_does_not_change_results():
    x = np.random.default_rng(3).normal(size=2000).astype(np.float32)
    np.testing.assert_array_equal(zscore_py(x, w=50, chunk=7), zscore_py(x, w=50, chunk=10**6))
    np.testing.assert_array_equal(mad_py(x, w=50, chunk=5), mad_py(x, w=50, chunk=10**6))
    np.testing.assert_array_equal(features_py(x, w=50, chunk=9), features_py(x, w=50, chunk=10**6))


def test_features_values_with_stride():
    x = np.random.default_rng(4).normal(size=500)
    f = features_py(x, w=32, s=5)
    win = x[50:82]                          # window index 10
    np.testing.assert_allclose(f[10], [win.mean(), win.std(ddof=1), win.min(), win.max()], rtol=1e-5)


def test_fit_is_deterministic_and_centroids_in_raw_space():
    x = np.random.default_rng(5).normal(50, 3, 4000).astype(np.float32)
    F = features_py(x, w=64)
    c1, s1, _ = fit_models(F, k=4, seed=0)
    c2, s2, _ = fit_models(F, k=4, seed=0)
    np.testing.assert_array_equal(c1, c2)
    assert 40 < c1[:, 0].min() and c1[:, 0].max() < 60      # mean column is on the raw scale
    assert kmeans_score_py(F, c1, s1).shape == (len(F),)


def test_evaluate_rejects_wrong_flag_count_and_counts_events_once():
    y = np.zeros(200, bool); y[100:110] = True
    with pytest.raises(ValueError):
        evaluate(np.zeros(10, bool), y, w=64)
    flags = np.zeros(200 - 64 + 1, bool); flags[100 - 63 + 3: 100 - 63 + 7] = True
    r = evaluate(flags, y, w=64)
    assert r["event_recall"] == 1.0 and r["precision"] == 1.0


def test_mad_plateau_gives_huge_score_documented_behaviour():
    """MAD = 0 on a flat window: the +1e-12 in the formula makes any change score ~1e12.
    The C kernel must reproduce this; the threshold step makes it harmless for flagging."""
    x = np.zeros(300, np.float32); x[200:] = 1
    assert mad_py(x, w=64).max() > 1e9
