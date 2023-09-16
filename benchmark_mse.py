import ctypes
import numpy as np
import math
import time
import tensor.ops.mse as ops

def mse_benchmark(n_rows, n_cols):
    np.random.seed(0)
    a = np.random.uniform(1, 100, (n_rows, n_cols))
    b = np.random.uniform(1, 100, (n_rows, n_cols))

    times = []
    for _ in range(1): 
        start = time.time()
        result = ops.mse(a, b)
        times.append(time.time()-start)
    print(f'time[ms] average of 1 in 100 iterations: {np.average(times) * 1e3}')
    mse_cpu = (np.square(a - b).sum())

    tolerance = 1e-5    
    abs_error = np.abs(mse_cpu - result)
    rel_error = abs_error / np.abs(mse_cpu)
    assert abs_error <= tolerance or rel_error <= tolerance, f'Matrix of {n_rows}x{n_cols} absolute error: {abs_error}, relative error: {rel_error}'


#mse_benchmark(128, 128) 
#mse_benchmark(256, 256) 
#mse_benchmark(512, 512) 
mse_benchmark(1024, 1024) 
mse_benchmark(2048, 2048) 
mse_benchmark(4096, 4096) 
mse_benchmark(8192, 8192) 