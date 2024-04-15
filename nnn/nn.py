"""Neural Network Implementation"""

import numpy as np

from autograd import Tensor
from base import Module, NeuralNetwork

## basic functions
class Add(Module):
    """z = x + y"""

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(x.data + y.data)
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0, i=None, j=None) -> np.ndarray:
        """
        1. `d(X+Y) / dX = \mathbb{I} \otimes \mathbb{I}` if X is matrix
        2. `d(x+y) / dx = \mathbb{I}` if x is vector
        """
        x = back_childs[idx]
        assert x.ndim <= 2
        if x.ndim == 2:
            res = np.zeros_like(x)
            res[i, j] = 1
            return res
        else:
            return np.eye(*x.shape)


class Inner(Module):
    """
    z = x @ y where z is vector or scalar
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(x.data @ y.data)
        assert res.ndim == 1, "result should be vector or scalar"
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0, i=None, j=None) -> np.ndarray:
        """
        if x is matrix
        d x@y / dx[i,j] = [0,...,y[j],... ,0], where y[j] is the ith component
        or in tensor product: `d x@y / dX = c' \otimes \mathbb{I}`

        elif x is vector
        d x@y / dx = y.T
        """
        x, y = back_childs
        if idx == 0:
            if x.data.ndim == 1:
                return y.data.T
            else:
                res = np.zeros((x.shape[0],))
                res[i] = y.data[j]
                return res
        else:
            if y.data.ndim == 1:
                return x.data
            else:
                res = np.zeros((x.shape[0],))
                res[j] = x.data[i]
                return res


## loss functions
class Sum(Module):
    """y = sum(X)"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data.sum().reshape((1,)))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx: int, i=None, j=None) -> np.ndarray:
        """Easy"""
        return np.ones_like(back_childs[idx])


class Norm(Module):
    """
    y = sqrt(x.T @ x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        m = np.abs(x.data).max()
        a = x.data / m
        # in my test, arr@arr is faster than np.inner(a,a) and np.dot(a,a)
        # and np.sqrt(a@a) is faster than np.linalg.norm(a, 2)
        res = Tensor(m * np.sqrt(a @ a).reshape((1,)))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0, i=None, j=None) -> np.ndarray:
        """
        `d norm2(x) / dx = x / norm2(x)`
        """
        x = back_childs[idx]
        y = self.forward(x)
        return x.data / y.data


class NLL(Module):
    """
    Negative Log Likelihod
    l = -sum(x @ y)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(-(x.data @ y.data).sum().reshape((1,)))
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx: int, i=None, j=None) -> np.ndarray:
        """
        dl / dx = -y, dl / dy = -x
        """
        return -back_childs[1 - idx]


## activation functions
class Log(Module):
    """
    y = x.log()
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        assert (x.data > 0).all()
        res = Tensor(np.log(x))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx: int, i=None, j=None) -> np.ndarray:
        """
        dy / dx = diag(1/x)
        """
        return np.diag(1 / back_childs[idx].data)


class Softamx(Module):
    """
    y = softmax(x) = x.exp() / x.exp().sum(), this is an approximation for `argamx`
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # avoid overflow
        e = np.exp(x.data - x.data.max())
        res = Tensor(e / e.sum())
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0, i=None, j=None) -> np.ndarray:
        x = back_childs[idx]
        n = x.shape[0]
        y = self.forward(x).data
        grad = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                grad[i, j] = y[i] * (1 - y[i]) if i == j else -y[i] * y[j]
                grad[j, i] = grad[i, j]
        return grad


class ReLU(Module):
    """
    y = relu(x) = max(0,x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(np.maximum(x.data, 0))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0, i=None, j=None) -> np.ndarray:
        x = back_childs[idx].data
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        return np.diag(grad)
