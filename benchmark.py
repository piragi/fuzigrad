import numpy as np
import torch
import torch.nn.functional as F
import time
import matmul as op

from tensor import Tensor

np.random.seed(1337)

def mse_time_comparison(n_rows, n_cols):
    rand1 = np.random.uniform(0, 100, (n_rows, n_cols))
    rand2 = np.random.uniform(0, 100, (n_rows, n_cols))

    # FuziGrad MSE calculation
    a = Tensor(rand1)
    b = Tensor(rand2) # TODO: takes about a second to generate both?!
    start = time.time()
    c = a.mse(b)
    end = time.time()
    fuzi_time = end - start

    # PyTorch MSE calculation
    a_torch = torch.tensor(rand1, requires_grad=False)
    b_torch = torch.tensor(rand2, requires_grad=False)
    start = time.time()
    c_torch = F.mse_loss(a_torch, b_torch)
    end = time.time()
    torch_time = end - start

    # CPU numpy MSE calculation
    start = time.time()
    c_cpu = np.mean((rand1 - rand2) ** 2)
    end = time.time()
    cpu_time = end - start

    print(f'FuziGrad MSE Time: {fuzi_time} seconds')
    print(f'PyTorch MSE Time: {torch_time} seconds')
    print(f'CPU numpy MSE Time: {cpu_time} seconds')

def matmul_time(n_rows, n_cols):
    rand1 = np.random.uniform(0, 100, (n_rows, n_cols))
    rand2 = np.random.uniform(0, 100, (n_rows, n_cols))
    a = Tensor(rand1)
    b = Tensor(rand2)

    start = time.time()
    c_normal = op.matmul_normal(a, b)
    c = op.matmul(a, b)

    c_torch = torch.tensor(rand1) @ torch.tensor(rand2)
    assert np.allclose(c, c_torch)
    

#mse_time_comparison(20000, 20000)  # specify number of rows and columns for the test tensors
matmul_time(1024, 1024)