"""Tensor operation implementation"""

from functools import wraps

from .autograd import Tensor
from .base import Operation, np

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
    """x.shape = (n, )"""
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
    """
    z = x + y

    x.shape = (batch, m, n)
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(x.data + y.data)
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx]
        b, m, n = x.shape
        if n == 1:
            return np.tile(np.eye(m), (b, 1, 1))
        else:
            raise NotImplementedError("No need.")


@singleton
class Inner(Operation):
    """
    z = x @ y

    x.shape = (batch, m, n)
    y.shape = (batch, n, k)
    z.shape = (batch, m, k)
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(np.einsum("bmn,bnk->bmk", x.data, y.data))
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x, y = back_childs
        x, y = x.data, y.data
        b1, m, n = x.shape
        b2, n, k = y.shape
        b = b1 if b2 == 1 else b2
        if idx == 0:
            # if x is vector
            if m == 1:
                return y.transpose((0, 2, 1))
            # if x is matrix, then y must be vector
            else:
                assert k == 1
                res = np.zeros((b, m, m, n))
                res[:, np.arange(m), np.arange(m), :] = np.tile(
                    y.transpose((0, 2, 1)), (1, m, 1)
                )
                return res
        else:
            if k == 1:
                return x
            else:
                assert m == 1
                res = np.zeros((b, k, n, k))
                res[:, np.arange(k), :, np.arange(k)] = np.tile(y, (1, k, 1)).transpose(
                    1, 0, 2
                )
                return res


@singleton
class Flatten(Operation):
    """y = x.flatten()"""

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data.reshape(x.data.shape[0], -1, 1))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx]
        b, m, n = x.shape
        if n == 1:
            return np.tile(np.eye(m), (b, 1, 1))
        else:
            res = np.zeros((b, m * n, m, n))
            for i in range(m * n):
                res[:, i, i // m, i % n] = 1
            return res


@singleton
class Sum(Operation):
    """y = sum(X)"""

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(x.data.sum(axis=(1, 2)).reshape((-1, 1, 1)))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """Easy"""
        return np.ones_like(back_childs[idx]).transpose((0, 2, 1))


@singleton
class Norm(Operation):
    """
    y = sqrt(x.T @ x)

    x.shape = (b,n,1)
    """

    def forward(self, x: Tensor) -> Tensor:
        # avoid overflow
        m = np.abs(x.data).max(axis=(1, 2), keepdims=True)  # shape=(b,1,1)
        a = x.data / m
        normsq = (a * a).sum(axis=(1, 2)).reshape(-1, 1, 1)
        res = Tensor(m * np.sqrt(normsq))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        `d norm2(x) / dx = x / norm2(x)`
        """
        x = back_childs[idx]
        y = self.forward(x)
        return (x.data / y.data).transpose((0, 2, 1))


@singleton
class NLL(Operation):
    """
    Negative Log Likelihod
    l = -sum(x @ y)

    x,y shape = (batch, n, 1)
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        res = Tensor(-(x.data * y.data).sum(axis=(1, 2)).reshape(-1, 1, 1))
        res.back_f = self
        res.back_childs = (x, y)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        dl / dx = -y, dl / dy = -x
        """
        return -back_childs[1 - idx].data.transpose((0, 2, 1))


@singleton
class Log(Operation):
    """
    y = x.log()

    x.shape = (batch, n, 1)
    """

    def forward(self, x: Tensor) -> Tensor:
        assert (x.data > 0).all()
        res = Tensor(np.log(x.data))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        """
        dy / dx = diag(1/x)
        """
        x = back_childs[idx].data
        b, n = x.shape[:2]
        # broadcast
        return np.tile(np.eye(n), (b, 1, 1)) * (1 / x)


@singleton
class Softmax(Operation):
    """
    y = softmax(x) = x.exp() / x.exp().sum(), this is an approximation for `argamx`

    x.shape = (batch, n, 1)
    """

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=(1, 2), keepdims=True))
        return e / e.sum(axis=(1, 2), keepdims=True)

    def forward(self, x: Tensor) -> Tensor:
        # avoid overflow
        res = Tensor(self.softmax(x.data))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx]
        b, n = x.shape[:2]
        y = self.softmax(x.data)
        # broadcast
        return np.tile(np.eye(n), (b, 1, 1)) * y - np.einsum(
            "bjk,bkl->bjl", y, y.transpose((0, 2, 1))
        )


@singleton
class LogSoftmax(Operation):
    """
    y = log(softmax(x)), more numerical stable version.

    x.shape = (batch, n, 1)
    """

    def forward(self, x: Tensor) -> Tensor:
        m = np.abs(x.data).max(axis=(1, 2), keepdims=True)
        e = np.exp(x.data - m)
        # avoid overflow
        log_softmax = x.data - m - np.log(e.sum(axis=(1, 2), keepdims=True))
        res = Tensor(log_softmax)
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx]
        b, n = x.shape[:2]
        y = Softmax.softmax(x.data).transpose((0, 2, 1))
        # broadcast
        return np.tile(np.eye(n), (b, 1, 1)) - y


@singleton
class ReLU(Operation):
    """
    y = relu(x) = max(0,x)

    x.shape = (batch, n, 1)
    """

    def forward(self, x: Tensor) -> Tensor:
        res = Tensor(np.maximum(x.data, 0))
        res.back_f = self
        res.back_childs = (x,)
        return res

    def gradient(self, back_childs: tuple, idx=0) -> np.ndarray:
        x = back_childs[idx].data
        b, n = x.shape[:2]
        grad = np.zeros_like(x)
        grad[x > 0] = 1
        # broadcast
        return np.tile(np.eye(n), (b, 1, 1)) * grad
