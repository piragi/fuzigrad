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
]

# Define the return type
libmatmul.reduce_kernel.restype = ctypes.c_float

def reduce_benchmark():
    np.random.seed(0)
    a = np.ones(256, dtype=float)
    a = np.array(a, dtype=np.float32, order='C')
    

    times = []
    for _ in range(1): 
        start = time.time()
        _ = libmatmul.reduce_kernel(a, 256)
        times.append(time.time()-start)
    print(f'time[ms] average of 1 in 100 iterations: {np.average(times) * 1e3}')

    print(a)
    assert a[0] == 128, f'a[0] = {a[0]}'

    return _

reduce_benchmark()