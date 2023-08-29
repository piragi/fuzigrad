import ctypes
import numpy as np

# Load the shared library
libmatmul = ctypes.CDLL('./cuda/libmatmul.so')

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
    # Ensure the matrices are of the correct data type and are contiguous in memory
    a = np.array(a.value, dtype=np.float32, order='C')
    b = np.array(b.value, dtype=np.float32, order='C')
    M, K = a.shape
    K_, N = b.shape
    assert K == K_, "Inner dimensions must match"

    # Create an empty result array
    c = np.empty((M, N), dtype=np.float32, order='C')

    # Call the CUDA wrapper function
    libmatmul.matmul(a, b, c, M, N, K)

    return c