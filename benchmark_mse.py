import ctypes
import numpy as np
import math

# Load the shared library
libmatmul = ctypes.CDLL('./build/libkernels.so')
# Define the argument types for the matmul_2d_benchmark function
libmatmul.mse.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
]

# Define the return type
libmatmul.mse.restype = ctypes.c_float

def mse_benchmark(n_rows, n_cols):
    np.random.seed(0)
    a = np.random.uniform(1, 100, (n_rows, n_cols))
    b = np.random.uniform(1, 100, (n_rows, n_cols))
    a = np.array(a, dtype=np.float32, order='C')
    b = np.array(b, dtype=np.float32, order='C')
    M, K = a.shape
    K_, N = b.shape
    assert K == K_

    block_dims = (math.ceil(M/128) * math.ceil(N/16))

    mse_gpu = np.zeros((block_dims), dtype=np.float32, order='C')

    _ = libmatmul.mse(a, b, mse_gpu, M, N)    
    mse_cpu = (np.square(a - b).mean())
    mse_gpu = (mse_gpu.sum() / (M*N))

    tolerance = 1e-5
    abs_error = np.abs(mse_cpu - mse_gpu)
    rel_error = abs_error / np.abs(mse_cpu)
    assert abs_error <= tolerance or rel_error <= tolerance


    return _


mse_benchmark(128, 128) 
mse_benchmark(256, 256) 
mse_benchmark(512, 512) 
mse_benchmark(1024, 1024) 
mse_benchmark(2048, 2048) 
mse_benchmark(4096, 4096) 