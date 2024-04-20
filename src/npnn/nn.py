"""Neural Network components implementation"""

from .base import Module, np
from .functional import Inner, Add
from .autograd import Tensor

rand = np.random.random


class Linear(Module):

    def __init__(self, in_size, out_size, bias=True) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.A = Tensor(rand((1, out_size, in_size)), requires_grad=True)
        if self.bias:
            self.b = Tensor(rand((1, out_size, 1)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        inner = Inner()
        add = Add()
        return add(inner(self.A, x), self.b)

    def parameters(self) -> list:
        return [self.A, self.b]


class Sequential(Module):

    def __init__(self, *layers) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def parameters(self) -> list:
        res = []
        for layer in self.layers:
            if isinstance(layer, Module):
                res.extend(layer.parameters())
        return res
