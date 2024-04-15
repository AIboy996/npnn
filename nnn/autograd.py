"""numpy neural network"""

import numpy as np

from base import Operation


class Tensor:

    def __init__(self, arr: np.array, requires_grad: bool = False):
        """
        Note that:
        we always have `self.back_f(*self.back_childs) == self`
        unless `self.back_f is None`
        """
        self.data: np.array = arr
        self.requires_grad: bool = requires_grad
        self.back_f: Operation | None = (
            None  # this indicates root node of the computional graph
        )
        self.back_childs: tuple[Tensor] = (self,)  # child nodes
        self.grad: np.array | None = None  # calcute grad if requires_grad = True

    def backward(self):
        """
        Backward Propagation algorithm.

        We only consider:
        1. input data is Vectors, shape = (m,)
        2. output metric is Scalars, shape = (1,)
        3. parameters are Matrices or Vectors, shape = (n,k) or (l,)

        For example: l = MSE(y, u), y = sigmoid(Bh+c), h = sigmoid(Ax+b)
        where
        x.shape = (m,)  A.shape = (k,m)  b.shape = (k,)
        h.shape = (k,)  B.shape = (n,k)  c.shape = (n,)
        y.shape = (n,)  l.shape = (1,)

        In this case, for i in range(k) for j in range(m) we have gradient:

        (dl / dA) [i,j] = (dl / dy) @ (dy / dh) @ (dh / dA[i,j])

        where each component is 2-d or 1-d array:
        (dl / dy).shape = (n,)
        (dy / dh).shape = (n,k)
        (dh / dA[i,j]).shape = (k,)
        """
        assert self.data.shape == (1,), "Only scalars can do backward"
        for idx, child in enumerate(self.back_childs):
            if child.back_f is None:  # root node
                if child.requires_grad:
                    assert child.ndim <= 2, "Parameters should have ndim <= 2"
                    # multivariable function, detivate of idx-th variable
                    child.grad = self.back_f.gradient(self.back_childs, idx)
                else:
                    pass
            else:
                child._bp(self.back_f.gradient(self.back_childs, idx))

    def _bp(self, grad: np.ndarray):
        """private gradient back propagation calculation"""
        assert grad.ndim <= 2, grad.shape
        assert self.ndim == 1, f"Only vectors can do _bp, {self.ndim = }"
        for idx, child in enumerate(self.back_childs):
            if child.back_f is None:
                if child.requires_grad:
                    assert child.ndim <= 2, "parameters should have ndim <= 2"
                    # parameter is vector
                    if child.ndim == 1:
                        child.grad = grad @ self.back_f.gradient(self.back_childs, idx)
                    # parameter is matrix
                    elif child.ndim == 2:
                        child.grad = np.zeros_like(child)
                        for i in range(child.shape[0]):
                            for j in range(child.shape[1]):
                                # back_f is a function: f(X1,X2,X3,...) map X into R^m
                                # we calcute detivates w.r.t. X[idx][i,j], which is a vector in R^m
                                child.grad[i, j] = grad @ self.back_f.gradient(
                                    self.back_childs, idx, i, j
                                )
                else:
                    pass
            else:
                # just like
                # (dl / dA) [i,j] = (dl / dy) @ (dy / dh) @ (dh / dA[i,j])
                # we **recursively** propagate gradient
                child._bp(grad @ self.back_f.gradient(self.back_childs, idx))

    def __getattr__(self, name: str):
        """
        When attr not defined, try to find in `self.data`, for example `self.shape` or `self.ndim`
        """
        return self.data.__getattribute__(name)

    def __repr__(self) -> str:
        return f"tensor({self.data}, requires_grad={self.requires_grad}, back_f={self.back_f})"

    def __neg__(self):
        """This will set requires_grad to False"""
        t = Tensor(-self.data)
        t.backf = self.back_f
        t.back_childs = self.back_childs
        return t


if __name__ == "__main__":
    from nnn.functional import *

    x = Tensor(np.random.random((3,)), requires_grad=True)
    print(x, -x, sep="\n")
