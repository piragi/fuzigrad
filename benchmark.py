import numpy as np
import torch
import torch.nn.functional as F
import time
import tensor.ops.matmul as matmul
import tensor.ops.mse as mse
from tensor.tensor import Tensor

repeat = 50

def time_operation(op_func, op_name, verification_func, n_rows, n_cols):
    a, b = prepare_data(n_rows, n_cols)
    times = []

    for _ in range(repeat):
        start = time.time()
        c = op_func(a, b)
        times.append(time.time() - start)
    
    print(f'({n_rows} x {n_cols}) time[ms] average of 1 {op_name} operation in {repeat} iterations: {np.average(times) * 1e3}')

    c_torch = verification_func(a, b)
    c_torch_np = c_torch.numpy().astype(c.dtype)
    assert np.allclose(c, c_torch_np)

def prepare_data(n_rows, n_cols):
    rand1 = np.random.uniform(0, 100, (n_rows, n_cols))
    rand2 = np.random.uniform(0, 100, (n_rows, n_cols))
    a = Tensor(rand1)
    b = Tensor(rand2)
    return a, b

matrix_sizes = [128, 256, 512, 1024, 2048]

# Timing matrix multiplication
for size in matrix_sizes:
    time_operation(matmul.matmul_2d, 'matmul', lambda a, b: torch.tensor(a.value) @ torch.tensor(b.value), size, size)

# Timing mean squared error
for size in matrix_sizes:
    time_operation(mse.mse, 'mean squared error', lambda a, b: F.mse_loss(torch.tensor(a.value), torch.tensor(b.value)), size, size)
