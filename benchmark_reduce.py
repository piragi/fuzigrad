import ctypes
import numpy as np
import math
import time

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


def reduce_benchmark():
    np.random.seed(0)
    M = 16384
    a = np.ones(M, dtype=float)
    a = np.array(a, dtype=np.float32, order='C')
    result = np.zeros((int) ((M / BM) * NUMBER_OF_WARPS), dtype=float) 
    result = np.array(result, dtype=np.float32, order='C')

    times = []
    for _ in range(1): 
        start = time.time()
        libmatmul.reduce_kernel(a, M, result)
        times.append(time.time()-start)
    print(f'time[ms] average of 1 in 100 iterations: {np.average(times) * 1e3}')

    print(result.size)
    assert a[0] == 128, f'a[0] = {a[0]}'

    return _

reduce_benchmark()