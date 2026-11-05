import psutil
from threadwatch.lib import load


def test_thread_count_matches_logical_cores_or_less():
    logical = psutil.cpu_count(logical=True)
    physical = psutil.cpu_count(logical=False) or logical
    assert 1 <= physical <= logical
    assert 1 <= load().tw_max_threads() <= logical
