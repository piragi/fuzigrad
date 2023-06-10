class Value():
    def __init__(self, value, op=None, children=None):
        self.value = value
        self.grad = 0
        self._op = op
        self._prev = children
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value + other.value, op='+', children=(self, other))
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
            
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.value * other.value, op='*', children=(self, other))

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.value**other, children=(self,))

        def _backward():
            self.grad += out.grad * (other * self.value**(other - 1))
        out._backward = _backward
        return out
            

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other**(-1)

    def backward(self):
        topo = []
        visited = set()

        self.grad = 1.0

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                if node._prev:
                    for child in node._prev:
                        build_topo(child)
                topo.append(node)

        build_topo(self)
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.value})"

a = Value(2.0)
b = Value(5.0)
c = (b / a)
print(c._prev)
c.backward()
print(a.grad)
print(b.grad)
