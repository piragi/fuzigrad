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

def test_matmul_128_256():
    rand1 = np.random.uniform(0, 100, (128, 256))
    rand2 = np.random.uniform(0, 100, (256, 128)) 
    matmul(rand1, rand2)

def test_matmul_128_128_128_256():
    rand1 = np.random.uniform(0, 100, (128, 128))
    rand2 = np.random.uniform(0, 100, (128, 256)) 
    matmul(rand1, rand2)

def matmul(rand, rand2=None):
    if (rand2 is None):
        rand2 = rand

    a = Tensor(rand)
    b = Tensor(rand2)
    a_torch = torch.tensor(rand)
    b_torch = torch.tensor(rand2)
    c = a @ b
    c_torch = a_torch @ b_torch
    assert np.allclose(c.value, c_torch.numpy(), 0.01), "Matmul doesn't match with PyTorch."