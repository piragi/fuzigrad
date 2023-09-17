import ctypes
import numpy as np
import math

# Load the shared library
libmatmul = ctypes.CDLL('/home/piragi/projects/fuzigrad/build/libkernel.so')

# Define the argument types for the matmul_2d_wrapper function
libmatmul.matmul.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]

libmatmul.matmul.restype = None

def matmul_2d(a, b):
    M, K = a.value.shape
    K_, N = b.value.shape

    assert K == K_, "Inner dimensions must match"
    assert M >= 64, "M dimension must be bigger than 128"
    assert N >= 64, "N dimension must be bigger than 128"
    assert K >= 64, "K dimension must be bigger than 128"

    # Pad to next multiple of two
    M_ext = get_next_power_of_two(M)
    K_ext = get_next_power_of_two(K)
    N_ext = get_next_power_of_two(N) 

    a_ext = np.zeros((M_ext, K_ext))
    b_ext = np.zeros((K_ext, N_ext))
    a_ext[:M, :K] = a.value
    b_ext[:K, :N] = b.value

    # Ensure the matrices are of the correct data type and are contiguous in memory
    a = np.array(a_ext, dtype=np.float32, order='C')
    b = np.array(b_ext, dtype=np.float32, order='C')
    # Create an empty result array
    c = np.empty((M_ext, N_ext), dtype=np.float32, order='C')

    # Call the CUDA wrapper function
    libmatmul.matmul(a, b, c, M_ext, N_ext, K_ext)

    c = c[:M, :N]

    return c

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