import ctypes
import numpy as np

# Load the shared library
libmatmul = ctypes.CDLL('./build/libkernel.so')
# Define the argument types for the matmul_2d_benchmark function
libmatmul.matmul_benchmark_wrapper.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

# Define the return type
libmatmul.matmul_benchmark_wrapper.restype = ctypes.c_float

def matmul_2d_benchmark(n_rows, n_cols):
    a = np.random.uniform(1, 100, (n_rows, n_cols))
    b = np.random.uniform(1, 100, (n_rows, n_cols))
    a = np.array(a, dtype=np.float32, order='C')
    b = np.array(b, dtype=np.float32, order='C')
    M, K = a.shape
    K_, N = b.shape
    assert K == K_
    c = np.empty((M, N), dtype=np.float32, order='C')

    flops = libmatmul.matmul_benchmark_wrapper(a, b, c, M, N, K)    
    return flops


matmul_2d_benchmark(128, 128) 
matmul_2d_benchmark(256, 256) 
matmul_2d_benchmark(512, 512) 
matmul_2d_benchmark(1024, 1024) 
matmul_2d_benchmark(2048, 2048) 
matmul_2d_benchmark(4096, 4096) 