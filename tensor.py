import numpy as np
from mlops import matmul, add, mul, relu, transpose, mse

# TODO: do not initialize Mlops for every Tensor created. Once should be enough
class Tensor():
   def __init__(self, value, op=None, children=None, requires_grad=False):
       self.value = np.array(value)
       if requires_grad:
         self.grad = Tensor(np.zeros_like(self.value).astype(float))
       self._op = op
       self._prev = children
       self._backward = lambda: None
   
   def __matmul__(self, other):
      out = Tensor(matmul(self, other), op='@', children=(self, other))

      def backward():
         self.grad += out.grad @ other.T
         other.grad += self.T @ out.grad
      out._backward = backward
      return out
   
   def __rmatmul__(self, other):
      return Tensor(matmul(other, self))

   def __add__(self, other):
      out = Tensor(add(self, other), op='+', children=(self, other))

      def backward():
         self.grad += out.value
         other.grad += out.value
      out._backward = backward
      return out

   def __sub__(self, other): 
      return Tensor(add(self, -other))
   
   def __mul__(self, other):
      out = Tensor(mul(self, other), op='*', children=(self, other))

      def backward():
         self.grad += Tensor(out.grad) * Tensor(other.value) 
         other.grad += out.grad * self.value
      out._backward = backward
      return out

   def __neg__(self):
      return Tensor(mul(self, Tensor([-1])))

   def relu(self):
      return Tensor(relu(self))

   def mse(self, other):
      return Tensor(mse(self, other))
   
   def backward(self):
      self.grad = Tensor(np.ones_like(self.value)) * Tensor([1/(self.shape[0]*self.shape[1])])

      topo = []
      visited = set()

      def build_topo(v):
         if v not in visited:
            visited.add(v)
            if v._prev:
               for prev in v._prev:
                  build_topo(prev)
            topo.append(v)
      build_topo(self)

      for v in reversed(topo):
         v._backward()

   @property
   def shape(self):
      return self.value.shape

   @property
   def T(self):
      return Tensor(transpose(self))



   