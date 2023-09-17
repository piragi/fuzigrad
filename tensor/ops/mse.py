import ctypes
import numpy as np
import time
import math


MSE_BM = 64
MSE_BN = 64
BM = 256
NUMBER_OF_THREADS = BM / 4 
NUMBER_OF_WARPS = NUMBER_OF_THREADS / 32

# Load the shared library
libmatmul = ctypes.CDLL('/home/piragi/projects/fuzigrad/build/libkernel.so')

libmatmul.mse.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
]
libmatmul.mse.restype = ctypes.c_float

libmatmul.reduce_kernel.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
]
libmatmul.reduce_kernel.restype = ctypes.c_float

def mse(a, b):
    a = np.array(a.value, dtype=np.float32, order='C')
    b = np.array(b.value, dtype=np.float32, order='C')
    M, K = a.shape
    K_, N = b.shape
    assert K == K_
    block_dims = (math.ceil(M/MSE_BM) * math.ceil(N/MSE_BN))
    result = np.zeros((block_dims), dtype=np.float32, order='C')
    libmatmul.mse(a, b, result, M, N)
    return reduce(result) / (M * N) 

# TODO: np implementation through cpu is way faster like 3-10x 
def reduce(a):
    if (a.size < 128):
        a_ext = np.zeros((128))
        a_ext[:a.size] = a
        a = a_ext

    M = a.size
    a = np.array(a, dtype=np.float32, order='C')
    result = np.zeros((int) ((M / BM) * NUMBER_OF_WARPS), dtype=float) 
    result = np.array(result, dtype=np.float32, order='C')
    libmatmul.reduce_kernel(a, M, result)
    if (result.size > 1):
        result = reduce(result)

    return result

def get_next_power_of_two(x):
    if (x == 0):
        return 1
    if (x < 128):
        return 128
    if (x & (x-1) == 0):
        return x
    power = 1
    while power < x:
        power *= 2
    return power