import numpy as np
import sys
sys.path.append('/home/piragi/projects/fuzigrad')
import tensor.ops.matmul as ops
from tensor.tensor import Tensor

def profile_matmul(n_rows, n_cols):
    a = Tensor(np.random.uniform(1, 100, (n_rows, n_cols)))
    b = Tensor(np.random.uniform(1, 100, (n_rows, n_cols)))

    flops = ops.matmul_2d(a, b)    
    return flops


profile_matmul(128, 128) 
profile_matmul(256, 256) 
profile_matmul(512, 512) 
profile_matmul(1024, 1024) 
profile_matmul(2048, 2048) 