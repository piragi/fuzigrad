import ctypes
import numpy as np

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

    c = np.zeros((1000), dtype=np.float32, order='C')

    flops = libmatmul.mse(a, b, c, M, N)    
    # print(c)
    print(c.sum() / (M*N))
    np_mse = (np.square(a - b).mean())
    print(np_mse)
    
    return flops


#mse_benchmark(128, 128) 
mse_benchmark(256, 256) 
mse_benchmark(512, 512) 
#mse_benchmark(1024, 1024) 
#mse_benchmark(2048, 2048) 
#mse_benchmark(4096, 4096) 