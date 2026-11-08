import ctypes as C, os, pathlib
import numpy as np

F32 = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")


class TwStats(C.Structure):
    _fields_ = [("windows", C.c_int64), ("ms", C.c_double),
                ("threads", C.c_int32), ("status", C.c_int32)]


def default_path():
    return pathlib.Path(__file__).resolve().parents[2] / "build" / "threadwatch.dll"


def _prepare(mingw_bin):
    if os.name == "nt" and os.path.isdir(mingw_bin):
        os.add_dll_directory(mingw_bin)        # finds libgomp-1.dll, libwinpthread-1.dll


def load(path=None, mingw_bin=r"C:\msys64\ucrt64\bin"):
    dll = pathlib.Path(path) if path else default_path()
    _prepare(mingw_bin)
    if not dll.is_file():
        raise FileNotFoundError(f"{dll} not found. Build it first: make build/threadwatch.dll (MSYS2 UCRT64 shell)")
    lib = C.CDLL(str(dll))                     # CDLL releases the GIL during calls
    # D-01
    lib.tw_version.restype = C.c_int
    lib.tw_max_threads.restype = C.c_int
    lib.tw_hello_threads.argtypes = [C.c_int32, C.POINTER(C.c_int32)]
    lib.tw_hello_threads.restype = C.c_int
    # D-04
    p64, pst = C.POINTER(C.c_int64), C.POINTER(TwStats)
    for name in ("tw_window_features", "tw_zscore", "tw_mad"):
        fn = getattr(lib, name)
        fn.argtypes = [F32, C.c_int64, C.c_int32, C.c_int32, F32, C.c_int32, p64, pst]
        fn.restype = C.c_int
    lib.tw_kmeans_score.argtypes = [F32, C.c_int64, C.c_int32, F32, F32, C.c_int32,
                                    F32, C.c_int32, p64, pst]
    lib.tw_kmeans_score.restype = C.c_int
    return lib
