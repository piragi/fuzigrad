import numpy as np
import sys
sys.path.append('/home/piragi/projects/fuzigrad')
import tensor.ops.mse as ops
from tensor.tensor import Tensor    

def profile_mse(n_rows, n_cols):
    a = Tensor(np.random.uniform(1, 100, (n_rows, n_cols)))
    b = Tensor(np.random.uniform(1, 100, (n_rows, n_cols)))

    result = ops.mse(a, b)
    mse_cpu = (np.mean((a.value - b.value)**2))

    tolerance = 1e-5    
    abs_error = np.abs(mse_cpu - result)
    rel_error = abs_error / np.abs(mse_cpu)
    assert abs_error <= tolerance or rel_error <= tolerance, f'Matrix of {n_rows}x{n_cols} absolute error: {abs_error}, relative error: {rel_error}'


profile_mse(128, 128) 
profile_mse(256, 256) 
profile_mse(512, 512) 
profile_mse(1024, 1024) 
profile_mse(2048, 2048) 
profile_mse(4096, 4096) 
profile_mse(8192, 8192) 