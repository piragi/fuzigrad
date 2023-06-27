from math import sqrt
import torch
from linear import Linear
from sgd import SGD
from value import Value
from tensor import Tensor
import numpy as np
import torch.nn.functional as F
import mlops

#np.random.seed(1337)

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


rand1 = np.random.uniform(0, 100, (785, 783))
rand2 = np.random.uniform(0, 100, (785, 783))
a_torch = torch.tensor(rand1)
b_torch = torch.tensor(rand2)
c_torch = F.mse_loss(a_torch, b_torch)
a = Tensor(rand1)
b = Tensor(rand2)
c = a.mse(b)
c_cpu = np.mean((rand1 - rand2)**2)
assert np.allclose(c_torch, c.value)
print(f'diff torch-fuzi {c_torch.numpy() - c.value}, fuzigrad:{c.value}, torch:{c_torch}, cpu:{c_cpu}')

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
l2 = l1 + b1
l3 = l2 @ w2
l4 = l3 + b2
loss = l4.mse(target)
ir_torch = x_torch @ w1_torch
ir_torch += b1_torch
ir_torch = F.relu(ir_torch)
ir_torch = ir_torch @ w2_torch
ir_torch += b2_torch
ir_torch = F.relu(ir_torch)
loss_torch = F.mse_loss(ir_torch, target_torch)
loss.backward()
loss_torch.backward()

assert np.allclose(w1.grad.value, w1_torch.grad)
assert np.allclose(b1.grad.value, b1_torch.grad)
assert np.allclose(w2.grad.value, w2_torch.grad)
assert np.allclose(b2.grad.value, b2_torch.grad)

data = [
    (Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True), 
     Tensor([[0.0], [1.0], [1.0], [0.0]])),
]

l1 = Linear(2, 5, 4)
l2 = Linear(5, 1, 4)
lr = 0.01
for i in range(10):
    epoch_loss = []
    for x, target in data:
        print(x.shape)
        x = l1.forward(x)
        x = l2.forward(x)
        loss = x.mse(target)
        epoch_loss.append(loss.value)
        loss.backward()
        optim = SGD(loss, lr=lr)
        optim.optimize()
    epoch_loss = np.mean(epoch_loss)
    print(f'epoch_loss = {epoch_loss}')
    if epoch_loss < 0.20:
        break

x = Tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], requires_grad=True)
x = l1.forward(x)
x = l2.forward(x)
