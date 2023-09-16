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
libmatmul.reduce_kernel.restype = ctypes.c_float

def reduce_benchmark(M):
    np.random.seed(0)
    a = np.random.uniform(1, 100, M)
    times = []
    for _ in range(1): 
        start = time.time()
        result = ops.reduce(a)
        times.append(time.time()-start)
    print(f'(GPU) time[ms] average of 1 in 100 iterations: {np.average(times) * 1e3}')

    tolerance = 1e-5    
    abs_error = np.abs(a.sum()- result[0])
    rel_error = abs_error / np.abs(a.sum())
    assert abs_error <= tolerance or rel_error <= tolerance, f'Array of {M}; absolute error: {abs_error}, relative error: {rel_error}'

    return _

reduce_benchmark(127)
reduce_benchmark(128)
reduce_benchmark(256)
reduce_benchmark(1024)
reduce_benchmark(16384)