"""C kernels (threads=1) against the D-03 NumPy reference: rtol 1e-5, same F1."""
import numpy as np
import pytest

from threadwatch.data import list_series, load_series
from threadwatch.data.loader import release
from threadwatch.detectors.features import features_py
from threadwatch.detectors.mad import mad_py
from threadwatch.detectors.ml import fit_models, kmeans_score_py, threshold
from threadwatch.detectors.zscore import zscore_py
from threadwatch.metrics import evaluate
from threadwatch.sdk import kmeans_score, mad, window_features, zscore

TOL = dict(rtol=1e-5, atol=1e-6)


def test_feature_equivalence():
    x = np.random.default_rng(10).normal(size=800).astype(np.float32)
    actual, st = window_features(x, 31, 4)
    np.testing.assert_allclose(actual, features_py(x, 31, 4), **TOL)
    assert st.windows == len(actual) and st.threads == 1 and st.status == 0 and st.ms >= 0


def test_zscore_equivalence_with_stride():
    x = np.random.default_rng(11).normal(size=800).astype(np.float32)
    np.testing.assert_allclose(zscore(x, 31, 4)[0], zscore_py(x, 31, 4), **TOL)


def test_mad_equivalence_odd_and_even_window():
    x = np.random.default_rng(12).normal(size=500).astype(np.float32)
    for w in (19, 20):
        np.testing.assert_allclose(mad(x, w, 3)[0], mad_py(x, w, 3), **TOL)


def test_kmeans_equivalence():
    rng = np.random.default_rng(13)
    F = rng.normal(size=(150, 4)).astype(np.float32)
    Cn = rng.normal(size=(5, 4)).astype(np.float32)
    sc = np.array([1, .5, 2, 3], np.float32)
    np.testing.assert_allclose(kmeans_score(F, Cn, sc)[0], kmeans_score_py(F, Cn, sc), **TOL)


def test_progress_counter_reaches_nw():
    x = np.random.default_rng(14).normal(size=20_000).astype(np.float32)
    prog = np.zeros(8, np.int64)
    out, st = zscore(x, 64, 1, progress=prog)
    assert prog[0] == len(out) == st.windows


def test_flat_and_sorted_windows_match_reference():
    x = np.concatenate([np.zeros(300), np.arange(300.0), np.full(300, 5.0), [9.0]]).astype(np.float32)
    np.testing.assert_allclose(mad(x, 64)[0], mad_py(x, 64), **TOL)
    np.testing.assert_allclose(zscore(x, 64)[0], zscore_py(x, 64), **TOL)


DATA = pytest.mark.skipif(not list_series(prefix="syn_"), reason="run: python -m threadwatch.data all")
SERIES = ["syn_1M", "nab_ec2_cpu_utilization_24ae8d", "nab_ec2_cpu_utilization_5f5533"]


@DATA
@pytest.mark.parametrize("name", SERIES)
def test_real_series_equivalence_and_same_f1(name):
    x, y, _ = load_series(name)
    x, y = np.array(x[:300_000]), np.array(y[:300_000])          # syn_1M: first 300k samples (keeps pytest fast)
    release()
    w, s = 256, 1
    z_c, z_p = zscore(x, w, s)[0], zscore_py(x, w, s)
    m_c, m_p = mad(x, w, s)[0], mad_py(x, w, s)
    F_c, F_p = window_features(x, w, s)[0], features_py(x, w, s)
    np.testing.assert_allclose(z_c, z_p, **TOL)
    np.testing.assert_allclose(m_c, m_p, **TOL)
    np.testing.assert_allclose(F_c, F_p, **TOL)
    idx = np.sort(np.random.default_rng(0).choice(len(F_p), min(100_000, len(F_p)), replace=False))
    cen, sc, _ = fit_models(F_p[idx], k=8, seed=0)
    k_c, k_p = kmeans_score(F_p, cen, sc)[0], kmeans_score_py(F_p, cen, sc)
    np.testing.assert_allclose(k_c, k_p, **TOL)
    for c_scores, p_scores in ((z_c, z_p), (m_c, m_p), (k_c, k_p)):
        f_c = evaluate(c_scores >= threshold(c_scores[idx], 0.001), y, w, s)
        f_p = evaluate(p_scores >= threshold(p_scores[idx], 0.001), y, w, s)
        assert abs(f_c["f1"] - f_p["f1"]) < 1e-6
