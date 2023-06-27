import numpy as np

from tensor import Tensor

class Linear():
    def __init__(self, ninput_features, noutput_features, batch_size):
        n = ninput_features + noutput_features  # number of inputs + number of outputs.
        self.weight = Tensor((np.random.rand(ninput_features, noutput_features) - 0.5) * np.sqrt(12 / n), requires_grad=True)
        #self.weight = Tensor(np.random.rand(ninput_features, noutput_features), requires_grad=True)
        self.bias = Tensor(np.random.rand(batch_size, noutput_features), requires_grad=True)

    def forward(self, x):
        x = x @ self.weight
        x = x + self.bias
        #print(x.value)
        x = x.relu()
        #print(x.value)
        return x