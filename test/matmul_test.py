import sys
sys.path.append('/home/piragi/projects/fuzigrad')

import pytest
import torch
from tensor.tensor import Tensor
import numpy as np


def test_matmul_64():
    rand = np.random.random((64, 64))
    matmul(rand)

def test_matmul_128():
    rand = np.random.random((128, 128))
    matmul(rand)

def test_matmul_224():
    rand = np.random.random((224, 224))
    matmul(rand)

def test_matmul_4096():
    rand = np.random.random((4096, 4096))
    matmul(rand)

def matmul(rand):
    a = Tensor(rand)
    b = Tensor(rand)
    a_torch = torch.tensor(rand)
    b_torch = torch.tensor(rand)
    c = a @ b
    c_torch = a_torch @ b_torch
    print(c.value)
    assert np.allclose(c.value, c_torch.numpy(), 0.01), "Matmul doesn't match with PyTorch."