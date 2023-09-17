import sys
sys.path.append('/home/piragi/projects/fuzigrad')

import pytest
from math import sqrt
import torch
from linear import Linear
from tensor.sgd import SGD
from tensor.value import Value
from tensor.tensor import Tensor
import numpy as np
import torch.nn.functional as F
import tensor.ops.mlops as mlops

def test_backward_gradients():
    b_torch = torch.Tensor([5.0]).double()
    b_torch.requires_grad = True
    c_torch = (b_torch/2)
    c_torch.backward()
    a = Value(2.0)
    b = Value(5.0)
    c = (b / a)
    c.backward()
    assert b.grad == b_torch.grad, "Gradients don't match with PyTorch."

same_tensors = {
    "a": ([[1,2,3], [1,2,3]]),
    "b": ([[1,2,3], [1,2,3]])
    }

def test_tensor_add():
    a = Tensor(same_tensors["a"])
    b = Tensor(same_tensors["b"])
    a_torch = torch.tensor(same_tensors["a"])
    b_torch = torch.tensor(same_tensors["b"])
    c = a + b
    c_torch = a_torch + b_torch
    assert np.allclose(c.value, c_torch.numpy()), "Tensor additions doesn't match with PyTorch."

def test_tensor_mul():
    a = Tensor(same_tensors["a"])
    a_torch = torch.tensor(same_tensors["a"])
    c = a * Tensor([1])
    c_torch = a_torch * 1
    assert np.allclose(c.value, c_torch.numpy()), "Tensor multiplication doesn't match with PyTorch."

def test_tensor_neg():
    a = Tensor(same_tensors["a"])
    a_torch = torch.tensor(same_tensors["a"])
    c = -a
    c_torch = -a_torch
    assert np.allclose(c.value, c_torch.numpy()), "Tensor negatives doesn't match with PyTorch."

def test_tensor_sub():
    a = Tensor(same_tensors["a"])
    b = Tensor(same_tensors["b"])
    a_torch = torch.tensor(same_tensors["a"])
    b_torch = torch.tensor(same_tensors["b"])
    c = a - b
    c_torch = a_torch - b_torch
    assert np.allclose(c.value, c_torch.numpy()), "Tensor subtraction doesn't match with PyTorch."

def test_relu():
    a = Tensor([[1, -1], [-1, 1]])
    a = a.relu()
    assert np.allclose(a.value, np.array([[1,0], [0, 1]])), "ReLU doesn't match with PyTorch."

def test_tensor_transpose():
    a = Tensor([[1,2,3], [1,2,3]])
    a = a.T
    a_torch = torch.tensor([[1,2,3], [1,2,3]])
    a_torch = a_torch.T
    assert np.allclose(a.value, a_torch.numpy()),  "Tensor transpose doesn't match with PyTorch."

# a = Tensor([[1.,2.,3.], [1.,2.,3.]], requires_grad=True)
# b = Tensor([[1.,2.], [1.,2.], [1.,2.]], requires_grad=True)
# c = a @ b
# a_torch = torch.tensor([[1.,2.,3.], [1.,2.,3.]], requires_grad=True)
# b_torch = torch.tensor([[1.,2.], [1.,2.], [1.,2.]], requires_grad=True)


# rand1 = np.random.uniform(0, 100, (785, 783))
# rand2 = np.random.uniform(0, 100, (785, 783))
# a_torch = torch.tensor(rand1)
# b_torch = torch.tensor(rand2)
# c_torch = F.mse_loss(a_torch, b_torch)
# a = Tensor(rand1)
# b = Tensor(rand2)
# c = a.mse(b)
# c_cpu = np.mean((rand1 - rand2)**2)
# assert np.allclose(c_torch, c.value)
# print(f'diff torch-fuzi {c_torch.numpy() - c.value}, fuzigrad:{c.value}, torch:{c_torch}, cpu:{c_cpu}')

def test_matmul():
    x_np = np.random.uniform(0, 100, (128, 256))
    w1_np = np.random.uniform(0, 100, (256, 128))
    b1_np = np.random.uniform(0, 100, (128, 128))
    w2_np = np.random.uniform(0, 100, (128, 256))

    x = Tensor(x_np, requires_grad=True)
    w1 = Tensor(w1_np, requires_grad=True)
    x_torch = torch.tensor(x_np, requires_grad=True)
    w1_torch = torch.tensor(w1_np, requires_grad=True)
    l1 = x @ w1
    l1_torch = x_torch @ w1_torch
    assert np.allclose(l1.value, l1_torch.detach().numpy(), 0.01)

    b1 = Tensor(b1_np, requires_grad=True)
    b1_torch = torch.tensor(b1_np, requires_grad=True)
    l2 = l1 + b1
    l2_torch = l1_torch + b1_torch
    assert np.allclose(l2.value, l2_torch.detach().numpy(), 0.01)

def mse_backward_matmul(): 
    x_np = np.random.uniform(0, 100, (100, 100))
    target_np = np.random.uniform(0, 100, (100, 100))
    w1_np = np.random.uniform(0, 100, (100, 200))
    b1_np = np.random.uniform(0, 100, (100, 200))
    w2_np = np.random.uniform(0, 100, (200, 100))
    b2_np = np.random.uniform(0, 100, (100, 100))

    x = Tensor(x_np, requires_grad=True)
    target = Tensor(target_np, requires_grad=True)
    w1 = Tensor(w1_np, requires_grad=True)
    b1 = Tensor(b1_np, requires_grad=True)
    w2 = Tensor(w2_np, requires_grad=True)
    b2 = Tensor(b2_np, requires_grad=True)

    x_torch = torch.tensor(x_np, requires_grad=True)
    target_torch = torch.tensor(target_np, requires_grad=True)
    w1_torch = torch.tensor(w1_np, requires_grad=True)
    b1_torch = torch.tensor(b1_np, requires_grad=True)
    w2_torch = torch.tensor(w2_np, requires_grad=True)
    b2_torch = torch.tensor(b2_np, requires_grad=True)

    l1 = x @ w1
    ir_torch = x_torch @ w1_torch
    assert np.allclose(l1.value, ir_torch.detach().numpy(), 0.01)
    l2 = l1 + b1
    l3 = l2 @ w2
    l4 = l3 + b2
    loss = l4.mse(target)
    ir_torch += b1_torch
    ir_torch = ir_torch @ w2_torch
    ir_torch += b2_torch
    loss_torch = F.mse_loss(ir_torch, target_torch)
    loss.backward()
    loss_torch.backward()

    assert np.allclose(w1.grad.value, w1_torch.grad)
    assert np.allclose(b1.grad.value, b1_torch.grad)
    assert np.allclose(w2.grad.value, w2_torch.grad)
    assert np.allclose(b2.grad.value, b2_torch.grad)

# # -- XOR NN --

# data = [
#     (Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True), 
#      Tensor([[0.0], [1.0], [1.0], [0.0]])),
# ]

# l1 = Linear(2, 5, 4)
# l2 = Linear(5, 1, 4)
# lr = 0.01
# for i in range(5):
#     epoch_loss = []
#     for x, target in data:
#         print(x.shape)
#         x = l1.forward(x)
#         x = l2.forward(x)
#         loss = x.mse(target)
#         epoch_loss.append(loss.value)
#         loss.backward()
#         optim = SGD(loss, lr=lr)
#         optim.optimize()
#     epoch_loss = np.mean(epoch_loss)
#     print(f'epoch_loss = {epoch_loss}')
#     if epoch_loss < 0.20:
#         break

# x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True)
# x = l1.forward(x)
# x = l2.forward(x)
# print(f'logits = {x.value}')