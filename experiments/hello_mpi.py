from mpi4py import MPI
from threadwatch.lib import load

comm = MPI.COMM_WORLD
lib = load()
print(f"rank {comm.Get_rank()}/{comm.Get_size()} dll v{lib.tw_version()} max_threads={lib.tw_max_threads()}")
