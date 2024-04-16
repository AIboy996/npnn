"""Tensor operation implementation"""

from functools import wraps

import numpy as np

from .autograd import Tensor
from .base import Operation

__all__ = [
    "Add",  # basic oprations
    "Inner",
    "Flatten",
    "Sum",  # loss functions
    "Norm",
    "NLL",
    "Log",  # activation functions
    "Softmax",
    "LogSoftmax",
    "ReLU",
]


def one_hot(x, NUM_CLASS):
    y = np.zeros((x.size, NUM_CLASS))
    y[np.arange(x.size), x] = 1
    return y


def singleton(original_cls):
    """
    Decorator for basic operations, make them all **singleton**.
    Otherwise eachtime we do operations, we will eastablish a new object.
    code refer to https://igeorgiev.eu/python/design-patterns/python-singleton-pattern-decorator/
    """
    original_new_method = original_cls.__new__
    instance = None

    @wraps(original_cls.__new__)
    def __new__(cls, *args, **kwargs):
        nonlocal instance
        if instance is None:
            instance = original_new_method(cls, *args, **kwargs)
        return instance

    original_cls.__new__ = __new__
    return original_cls


@singleton
class Add(Operation):
    """z = x + y"""

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(x.data + y.data)
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        1. `d(X+Y) / dX = \mathbb{I} \otimes \mathbb{I}` if X is matrix
        2. `d(x+y) / dx = \mathbb{I}` if x is vector
        """
        x = back_childs[idx]
        assert x.ndim <= 2
        if x.ndim == 1:
            return np.eye(*x.shape)
        else:
            raise NotImplementedError("No need.")


@singleton
class Inner(Operation):
    """
    z = x @ y where z is vector or scalar
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(x.data @ y.data)
        assert res.ndim == 1, "result should be vector or scalar"
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        if x is matrix
        d x@y / dx[i,j] = [0,...,y[j],... ,0], where y[j] is the ith component
        or in tensor product: `d x@y / dX = c' \otimes \mathbb{I}`

        elif x is vector
        d x@y / dx = y.T
        """
        x, y = back_childs
        x, y = x.data, y.data
        if idx == 0:
            if x.ndim == 1:
                return y.T
            else:
                m, n = x.shape
                res = np.zeros((m, m, n))
                res[np.arange(m), np.arange(m), :] = y
                return res
        else:
            if y.ndim == 1:
                return x
            else:
                m, n = y.shape
                res = np.zeros((n, m, n))
                res[np.arange(n), :, np.arange(n)] = x
                return res


@singleton
class Flatten(Operation):
    """y = x.flatten()"""

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data.flatten())
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx]
        if x.ndim == 1:
            return np.eye(*x.shape)
        else:
            m, n = x.shape
            res = np.zeros((m * n, m, n))
            for i in range(m * n):
                res[i, i // m, i % n] = 1
            return res


@singleton
class Sum(Operation):
    """y = sum(X)"""

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data.sum().reshape((1,)))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """Easy"""
        return np.ones_like(back_childs[idx])


@singleton
class Norm(Operation):
    """
    y = sqrt(x.T @ x)
    """

    def forward(self, x: Tensor) -> Tensor:
        m = np.abs(x.data).max()
        a = x.data / m
        # in my test, arr@arr is faster than np.inner(a,a) and np.dot(a,a)
        # and np.sqrt(a@a) is faster than np.linalg.norm(a, 2)
        res = Tensor(m * np.sqrt(a @ a).reshape((1,)))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        `d norm2(x) / dx = x / norm2(x)`
        """
        x = back_childs[idx]
        y = self.forward(x)
        return x.data / y.data


@singleton
class NLL(Operation):
    """
    Negative Log Likelihod
    l = -sum(x @ y)
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(-(x.data @ y.data).sum().reshape((1,)))
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        dl / dx = -y, dl / dy = -x
        """
        return -back_childs[1 - idx]


@singleton
class Log(Operation):
    """
    y = x.log()
    """

    def forward(self, x: Tensor) -> Tensor:
        assert (x.data > 0).all()
        res = Tensor(np.log(x))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        dy / dx = diag(1/x)
        """
        return np.diag(1 / back_childs[idx].data)


@singleton
class Softmax(Operation):
    """
    y = softmax(x) = x.exp() / x.exp().sum(), this is an approximation for `argamx`
    """

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def forward(self, x: Tensor) -> Tensor:
        # avoid overflow
        res = Tensor(Softmax.softmax(x.data))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx]
        y = self.softmax(x.data)
        vec = y.reshape((-1, 1))
        # broadcast
        return np.diag(y) - vec @ vec.T
        # n = x.shape[0]
        # grad = np.zeros((n, n))
        # for i in range(n):
        #     for j in range(i, n):
        #         grad[i, j] = y[i] * (1 - y[i]) if i == j else -y[i] * y[j]
        #         grad[j, i] = grad[i, j]
        # return grad


@singleton
class LogSoftmax(Operation):
    """
    y = log(softmax(x)), more numerical stable version.
    """

    def forward(self, x: Tensor) -> Tensor:
        a = x.data
        m = np.abs(a).max()
        e = np.exp(a - m)
        # avoid overflow
        res = Tensor(a - m - np.log(e.sum()))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx]
        n = x.shape[0]
        y = Softmax.softmax(x.data).reshape((1, n))
        # broadcast
        return np.eye(n) - y
        # grad = np.zeros((n,n))
        # for i in range(n):
        #     for j in range(n):
        #         grad[i, j] = 1 - y[j] if i == j else -y[j]
        # return grad


@singleton
class ReLU(Operation):
    """
    y = relu(x) = max(0,x)
    """

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(np.maximum(x.data, 0))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx].data
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        return np.diag(grad)
