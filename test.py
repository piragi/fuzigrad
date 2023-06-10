import torch

b = torch.Tensor([5.0]).double()
b.requires_grad = True

c = (b/2)
c.backward()

print(b.grad)