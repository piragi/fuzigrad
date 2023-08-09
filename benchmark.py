import numpy as np
import torch
import torch.nn.functional as F
import time
import sys
import matmul as op
import nvtx

from tensor import Tensor

#np.random.seed(1337)

@nvtx.annotate(color="blue")
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
    rand1 = np.random.uniform(0, 10, (n_rows, n_cols))
    rand2 = np.random.uniform(0, 10, (n_rows, n_cols))
    a = Tensor(rand1)
    b = Tensor(rand2)

    #c_normal = op.matmul_normal(a, b)
    #c = op.matmul(a, b)
    
    #c = op.matmul_1d_blocktiling(a,b)
    #c = op.matmul_2d_blocktiling(a,b)
    c = op.matmul_2d_blocktiling_cuda(a,b)

    c_torch = torch.tensor(rand1) @ torch.tensor(rand2)

    # Convert c_torch to numpy array and same data type as c
    c_torch_np = c_torch.numpy().astype(c.dtype)

    # Add prints and checks
    print(c)
    print(c_torch_np)
    print("0.0 in c: ", np.any(c == 0.0))
    print("0.0 in c_torch_np: ", np.any(c_torch_np == 0.0))
    # Print if there are any NaN or infinite values in c or c_torch_np
    print("NaN or inf in c: ", np.isnan(c).any() or np.isinf(c).any())
    print("NaN or inf in c_torch_np: ", np.isnan(c_torch_np).any() or np.isinf(c_torch_np).any())

    # Compute the absolute difference between the two matrices
    diff = np.abs(c - c_torch_np)

    # Find the indices where the difference exceeds a certain tolerance
    tolerance = 1e-2
    divergent_indices = np.where(diff > tolerance)

    # Number of divergences
    num_divergences = len(divergent_indices[0])

    if num_divergences > 0:
        # If there are divergent values, compute some stats and print a summary
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        print(f'Total divergent elements: {num_divergences}')
        print(f'Max divergence: {max_diff}')
        print(f'Mean divergence: {mean_diff}')
    else:
        print('No divergent elements found.')
    # Compare c and c_torch_np instead of c_torch
    assert np.allclose(c, c_torch_np)

#mse_time_comparison(20000, 20000)  # specify number of rows and columns for the test tensors
matmul_time(256, 256) 
