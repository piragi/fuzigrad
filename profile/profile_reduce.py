import ctypes
import numpy as np
import time
import sys
sys.path.append('/home/piragi/projects/fuzigrad')
import tensor.ops.mse as ops

def reduce_profile(M):
    np.random.seed(0)
    a = np.random.uniform(1, 100, M)
    result = ops.reduce(a)

    tolerance = 1e-5    
    abs_error = np.abs(a.sum()- result[0])
    rel_error = abs_error / np.abs(a.sum())
    assert abs_error <= tolerance or rel_error <= tolerance, f'Array of {M}; absolute error: {abs_error}, relative error: {rel_error}'

reduce_profile(127)
reduce_profile(128)
reduce_profile(256)
reduce_profile(1024)
reduce_profile(16384)