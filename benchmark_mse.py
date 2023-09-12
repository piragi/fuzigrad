import ctypes
import numpy as np
import math
import time

# Load the shared library
libmatmul = ctypes.CDLL('./build/libkernel_debug.so')
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

MSE_BM = 64
MSE_BN = 64

def mse_benchmark(n_rows, n_cols):
    np.random.seed(0)
    a = np.random.uniform(1, 100, (n_rows, n_cols))
    b = np.random.uniform(1, 100, (n_rows, n_cols))
    a = np.array(a, dtype=np.float32, order='C')
    b = np.array(b, dtype=np.float32, order='C')
    M, K = a.shape
    K_, N = b.shape
    assert K == K_

    results_row_offset_corrected_v2 = []

    debug = False
    iterations = 100
    if debug:
        iterations = 1
        for i in range(2):
            for j in range(2):
                warp_pos_x = i * 32
                warp_pos_y = j * 32
                for k in range(32):
                    thread_id = k + j*32 + i*64
                    thread_pos_y = k * 4
                    pos_x = warp_pos_x
                    pos_y = warp_pos_y + thread_pos_y
                    result1 = np.square(a[pos_x:pos_x+4, pos_y:pos_y+4] - b[pos_x:pos_x+4, pos_y:pos_y+4]).sum()
                    result2 = np.square(a[pos_x+16:pos_x+20, pos_y:pos_y+4] - b[pos_x+16:pos_x+20, pos_y:pos_y+4]).sum()
                    results_row_offset_corrected_v2.append({
                        'thread_id': thread_id,
                        'result': result1 + result2
                    })

        for element in results_row_offset_corrected_v2:
            print(f'thread_id: {element["thread_id"]}, result: {element["result"]}')

        print('-------')
        print(a[:4, 16:20])
        print(a[16:20, 16:20])



    #print(f'result: {result.sum()}')
    # TODO: synchronize constants across cuda and python
    block_dims = (math.ceil(M/MSE_BM) * math.ceil(N/MSE_BN))


    times = []
    for _ in range(1): 
        mse_gpu = np.zeros((block_dims), dtype=np.float32, order='C')
        start = time.time()
        _ = libmatmul.mse(a, b, mse_gpu, M, N)
        times.append(time.time()-start)
    print(f'time[ms] average of 1 in 100 iterations: {np.average(times) * 1e3}')
    mse_cpu = (np.square(a - b).mean())
    #print(mse_gpu)
    mse_gpu = (mse_gpu.sum() / (M*N))
    

    tolerance = 1e-5    
    abs_error = np.abs(mse_cpu - mse_gpu)
    rel_error = abs_error / np.abs(mse_cpu)
    assert abs_error <= tolerance or rel_error <= tolerance, f'Matrix of {M}x{N} absolute error: {abs_error}, relative error: {rel_error}'


    return _


mse_benchmark(128, 128) 
mse_benchmark(256, 256) 
mse_benchmark(512, 512) 
mse_benchmark(1024, 1024) 
mse_benchmark(2048, 2048) 
mse_benchmark(4096, 4096) 
mse_benchmark(8192, 8192) 