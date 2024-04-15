"""Neural Network Implementation"""

import numpy as np

from autograd import Tensor
from base import Module


## basic functions
class Add(Module):
    """z = x + y"""

    def forward(self, x, y) -> Tensor:
        res = Tensor(x.data + y.data)
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, x: tuple, idx=0, i=None, j=None) -> np.ndarray:
        """
        1. `d(X+Y) / dX = \mathbb{I} \otimes \mathbb{I}` if X is matrix
        2. `d(x+y) / dx = \mathbb{I}` if x is vector
        """
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

    def gradient(self, x, idx=0, i=None, j=None) -> np.ndarray:
        """
        dXc / dX[i,j] = [0,...,c[k],... ,0], where c[k] is the ith component
        or in tensor product: `dXc / dX = c' \otimes \mathbb{I}`
        """
        res = np.zeros((x[idx].shape[0],))
        res[i] = self.multiplier[j]
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

    def gradient(self, x: tuple, idx: int, i=None, j=None) -> np.ndarray:
        """Easy"""
        return np.ones_like(x[idx])


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

    def gradient(self, x, idx=0, i=None, j=None) -> np.ndarray:
        """
        `d norm2(x) / dx = x / norm2(x)`
        """
        x = x[idx]
        y = self.forward(x).data
        return x.data / y


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

    def gradient(self, x: tuple, idx: int, i=None, j=None) -> np.ndarray:
        """
        dl / dx = -y, dl / dy = -x
        """
        return -x[1 - idx]


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

    def gradient(self, x: tuple, idx: int, i=None, j=None) -> np.ndarray:
        """
        dy / dx = diag(1/x)
        """
        return np.diag(1 / x[idx].data)


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

    def gradient(self, x, idx=0, i=None, j=None) -> np.ndarray:
        x = x[idx]
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

    def gradient(self, x, idx=0, i=None, j=None) -> np.ndarray:
        x = x[idx].data
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        return np.diag(grad)
