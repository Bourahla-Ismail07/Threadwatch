import ctypes as C
from threadwatch.lib import load

def test_hello_dll():
    lib = load()
    assert lib.tw_version() == 100
    assert lib.tw_max_threads() >= 1

def test_threads_run():
    lib = load()
    seen = C.c_int32(0)
    assert lib.tw_hello_threads(4, C.byref(seen)) == 0
    assert seen.value == 4
