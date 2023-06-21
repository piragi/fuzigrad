import torch
from value import Value
from tensor import Tensor
import numpy as np
import torch.nn.functional as F

b_torch = torch.Tensor([5.0]).double()
b_torch.requires_grad = True
c_torch = (b_torch/2)
c_torch.backward()
a = Value(2.0)
b = Value(5.0)
c = (b / a)
c.backward()
assert b.grad == b_torch.grad


a = Tensor([[1,2,3], [1,2,3]])
b = Tensor([[1,2], [1,2], [1,2]])
c = a @ b
a_torch = torch.tensor([[1,2,3], [1,2,3]])
b_torch = torch.tensor([[1,2], [1,2], [1,2]])
c_torch = a_torch @ b_torch
assert np.allclose(c.value, c_torch.numpy())


a = Tensor([[1,2,3], [1,2,3]])
b = Tensor([[1,2,3], [1,2,3]])
a_torch = torch.tensor([[1,2,3], [1,2,3]])
b_torch = torch.tensor([[1,2,3], [1,2,3]])
c = a + b
c_torch = a_torch + b_torch
assert np.allclose(c.value, c_torch.numpy())

c = a * Tensor([1])
c_torch = a_torch * 1
assert np.allclose(c.value, c_torch.numpy())

c = -a
c_torch = -a_torch
assert np.allclose(c.value, c_torch.numpy())

a = Tensor([[1,2,3], [1,2,3]])
b = Tensor([[1,2,3], [1,2,3]])
a_torch = torch.tensor([[1,2,3], [1,2,3]])
b_torch = torch.tensor([[1,2,3], [1,2,3]])
c = a - b
c_torch = a_torch - b_torch
assert np.allclose(c.value, c_torch.numpy())

rand = np.random.random((1000, 1000))
a = Tensor(rand)
b = Tensor(rand)
a_torch = torch.tensor(rand)
b_torch = torch.tensor(rand)
c = a @ b
c_torch = a_torch @ b_torch
assert np.allclose(c.value, c_torch.numpy())

a = Tensor([[1, -1], [-1, 1]])
a = a.relu()
assert np.allclose(a.value, np.array([[1,0], [0, 1]]))

a = Tensor([[1,2,3], [1,2,3]])
a = a.T
a_torch = torch.tensor([[1,2,3], [1,2,3]])
a_torch = a_torch.T
assert np.allclose(a.value, a_torch.numpy())

a = Tensor([[1.,2.,3.], [1.,2.,3.]], requires_grad=True)
b = Tensor([[1.,2.], [1.,2.], [1.,2.]], requires_grad=True)
c = a @ b
a_torch = torch.tensor([[1.,2.,3.], [1.,2.,3.]], requires_grad=True)
b_torch = torch.tensor([[1.,2.], [1.,2.], [1.,2.]], requires_grad=True)

c_torch = a_torch @ b_torch
mean_c = c_torch.mean()  # mean of c_torch to get a scalar
mean_c.backward()
print(c_torch)
print(mean_c)
print(a_torch.grad)
print(f'a_torch.grad = {a_torch.grad}')
print(f'b_torch.grad = {b_torch.grad}')

c.backward()
print(f'a.grad = {a.grad.value}')
print(f'b.grad = {b.grad.value}')

a_torch = torch.tensor([[1,2,3], [1,2,3]])
b_torch = torch.tensor([1])
c_torch = a_torch * b_torch
print(c_torch)

rand1 = np.array([[1,2,3], [1,2,3]])
rand2 = np.array([[0.9, 2.1, 3.8], [0.2, 2.5, 4]])
a_torch = torch.tensor(rand1)
b_torch = torch.tensor(rand2)
c_torch = F.mse_loss(a_torch, b_torch)
a = Tensor(rand1)
b = Tensor(rand2)
c = a.mse(b)
print(f'diff torch-fuzi {c_torch - c.value}, c_torch:{c_torch}, c:{c.value}')
