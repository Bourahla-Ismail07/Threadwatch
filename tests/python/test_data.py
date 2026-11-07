import hashlib
import json

import numpy as np
import pytest

from threadwatch.data import list_series, load_series, nab, smd, synthetic
from threadwatch.data.loader import release, write_series


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_same_bytes(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    synthetic.generate_to_disk("s", 300_000, seed=7, root=a)
    synthetic.generate_to_disk("s", 300_000, seed=7, root=b)
    synthetic.generate_to_disk("s", 300_000, seed=8, root=c)
    assert _sha(a / "s.f32") == _sha(b / "s.f32")
    assert _sha(a / "s.labels.u8") == _sha(b / "s.labels.u8")
    assert _sha(a / "s.f32") != _sha(c / "s.f32")


def test_blocks_continue_phase_and_noise(tmp_path):
    """A small block size must still give a finite, continuous series with the same labels."""
    synthetic.generate_to_disk("s", 120_000, seed=3, root=tmp_path, block=50_000)
    x, y, m = load_series("s", tmp_path)
    assert np.isfinite(x).all() and len(x) == len(y) == m["n"] == 120_000
    assert np.abs(np.diff(x[49_990:50_010])).max() < 15      # no jump at the block border
    release(x, y)


@pytest.mark.parametrize("n", [1_000_000, 2_000_000])
def test_rate_within_20_percent(tmp_path, n):
    m = synthetic.generate_to_disk("s", n, seed=7, root=tmp_path)
    assert abs(m["rate_actual"] - 0.001) <= 0.0002
    x, y, meta = load_series("s", tmp_path)
    assert len(x) == len(y) == n and y.dtype == np.uint8 and set(np.unique(y)) <= {0, 1}
    release(x, y)


def test_loader_roundtrip_and_length_check(tmp_path):
    write_series("t", np.arange(10, dtype=np.float64), np.zeros(10), {"source": "test"}, tmp_path)
    x, y, m = load_series("t", tmp_path)
    assert x.dtype == np.float32 and list(x[:3]) == [0, 1, 2] and m["source"] == "test"
    release(x, y)
    (tmp_path / "t.labels.u8").write_bytes(b"\0" * 9)
    with pytest.raises(ValueError):
        load_series("t", tmp_path)


def test_nab_windows_and_missing(tmp_path):
    csv = tmp_path / "f.csv"
    csv.write_text("timestamp,value\n2014-01-01 00:00:00,1\n2014-01-01 00:05:00,\n"
                   "2014-01-01 00:10:00,3\n2014-01-01 00:15:00,4\n")
    x, y, miss, step = nab.convert_file(
        csv, [["2014-01-01 00:05:00.000000", "2014-01-01 00:10:00.000000"]])
    assert list(y) == [0, 1, 1, 0] and miss == 1 and list(x) == [1, 1, 3, 4]
    assert step.total_seconds() == 300


def test_smd_metric_rule(tmp_path):
    f = tmp_path / "m.txt"
    f.write_text("1-5:1,2\n10-20:2,3\n30-40:2\n")
    assert smd.pick_metric(f) == 1          # dimension 2 (1-based) appears in 3 segments


REAL = pytest.mark.skipif(not list_series(prefix="nab_"), reason="run: python -m threadwatch.data all")


@REAL
def test_real_datasets_load_and_align():
    for prefix in ("nab_", "smd_"):
        for name in list_series(prefix=prefix):
            x, y, m = load_series(name)
            assert len(x) == len(y) == m["n"] and np.isfinite(x).all()
            if prefix == "smd_":
                assert y.any()
            release(x, y)


@REAL
def test_most_nab_series_have_labels():
    with_labels = 0
    for name in list_series(prefix="nab_"):
        x, y, _ = load_series(name)
        with_labels += bool(y.any())
        release(x, y)
    assert with_labels >= 15


@REAL
def test_all_sizes_exist():
    names = set(list_series(prefix="syn_"))
    assert {"syn_1M", "syn_2M", "syn_5M", "syn_10M", "syn_20M"} <= names
