import ctypes as C, os, pathlib

def load(path=None, mingw_bin=r"C:\msys64\ucrt64\bin"):
    root = pathlib.Path(__file__).resolve().parents[2]
    dll = pathlib.Path(path) if path else root / "build" / "threadwatch.dll"
    if os.name == "nt" and os.path.isdir(mingw_bin):
        os.add_dll_directory(mingw_bin)        # finds libgomp-1.dll, libwinpthread-1.dll
    lib = C.CDLL(str(dll))                     # CDLL releases the GIL during calls
    lib.tw_version.restype = C.c_int
    lib.tw_max_threads.restype = C.c_int
    lib.tw_hello_threads.argtypes = [C.c_int32, C.POINTER(C.c_int32)]
    lib.tw_hello_threads.restype = C.c_int
    return lib
