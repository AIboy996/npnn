"""Neural Network Implementation"""

import numpy as np

from autograd import Tensor
from base import Module


class Sum(Module):
    """y = sum(X)"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data.sum().reshape((1,)))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, x: tuple, idx: int, i=None, j=None):
        return np.ones_like(x[idx])


class Add(Module):

    def forward(self, x, y) -> Tensor:
        res = Tensor(x.data + y.data)
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, x: tuple, idx=0, i=None, j=None):
        x = x[idx]
        assert x.ndim <= 2
        if x.ndim == 2:
            res = np.zeros_like(x)
            res[i, j] = 1
            return res
        else:
            return np.eye(*x.shape)


class RightMultiply(Module):
    """y = Xc"""

    def __init__(self, multiplier) -> None:
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data @ self.multiplier)
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, x, idx=0, i=None, j=None):
        res = np.zeros((x[idx].shape[0],))
        res[j] = self.multiplier[j]
        return res


class LeftMultiply(Module):
    """y = cX"""

    def __init__(self, multiplier) -> None:
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(self.multiplier @ x.data)
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, x, idx=0, i=None, j=None):
        res = np.zeros((x[idx].shape[0],))
        res[i] = self.multiplier[i]
        return res