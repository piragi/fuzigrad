import ctypes
import numpy as np
import math
import time
import tensor.ops.mse as ops

# Load the shared library
libmatmul = ctypes.CDLL('/home/piragi/projects/fuzigrad/build/libkernel_debug.so')
# Define the argument types for the matmul_2d_benchmark function
libmatmul.reduce_kernel.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
]

# Define the return type
libmatmul.reduce_kernel.restype = ctypes.c_float

BM = 256
NUMBER_OF_THREADS = BM / 4 
NUMBER_OF_WARPS = NUMBER_OF_THREADS / 32


def reduce_benchmark(M):
    np.random.seed(0)
    a = np.ones(M, dtype=float)

    times = []
    for _ in range(1): 
        start = time.time()
        result = ops.reduce(a)
        times.append(time.time()-start)
    print(f'(GPU) time[ms] average of 1 in 100 iterations: {np.average(times) * 1e3}')

    assert result[0] == M, f'result[0] = {result[0]}'

    return _

reduce_benchmark(127)
reduce_benchmark(128)
reduce_benchmark(256)
reduce_benchmark(1024)
reduce_benchmark(16384)