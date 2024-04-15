from numpy.random import random as rand

from base import Module
from functional import Inner, Add
from autograd import Tensor


class Linear(Module):

    def __init__(self, in_size, out_size, bias=True) -> None:
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.A = Tensor(rand((out_size, in_size)), requires_grad=True)
        if self.bias:
            self.b = Tensor(rand((out_size,)), requires_grad=True)

    def forward(self, x: Tensor):
        inner = Inner()
        add = Add()
        return add(inner(self.A, x), self.b)

    def parameters(self) -> tuple:
        return (self.A, self.b)
