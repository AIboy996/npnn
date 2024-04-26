"""
Autograd algorithm implementation.
Calcute Tensor's grad by tracking `back_childs` and `back_f`(in fact these two generate a computional tree).
"""

from .base import Operation, np


class Tensor:
    """Data Container designed for gradient descent based Machine Learning."""

    def __init__(self, arr: np.array, requires_grad: bool = False):
        """
        Note that:
        1. we always have `self.back_f(*self.back_childs) == self`, unless `self.back_f is None`.
        2. we always have `self.data.shape = (batch, m, n)`, if self is parameter(require grad),
           batch==1 for sure.
        3. for vectors, we create column vectors(n==1) by defualt.
        """
        assert arr.ndim == 3, "tensor should have data.ndim==3 which is (batch, m, n)"
        self.data: np.array = arr
        self.requires_grad: bool = requires_grad
        self.back_f: Operation | None = (
            None  # this indicates root node of the computional graph
        )
        self.back_childs: tuple[Tensor] = (self,)  # child nodes
        if requires_grad:
            assert arr.shape[0] == 1, "tensor requires_grad should have batch==1"
            # initialize grad if requires_grad = True
            self.grad = np.zeros_like(arr)
        else:
            self.grad = None
        self.back_counter = 0

    def backward(self):
        """
        Backward Propagation algorithm.

        We only consider:
        1. input data are Vectors, with shape = (b,m,1)
        2. output metrics are Scalars, with shape = (1,1,1)
        3. parameters are Matrices or Vectors, with shape = (1,n,k) or (1,l,1)

        For example: l = MSE(y, u), y = sigmoid(Bh+c), h = sigmoid(Ax+b)
        where
        x.shape = (b,m,1)  A.shape = (1,k,m)  b.shape = (1,k,1)
        h.shape = (b,k,1)  B.shape = (1,n,k)  c.shape = (1,n,1)
        y.shape = (b,n,1)  u.shape = (b,n,1)  l.shape = (b,1,1)

        In this case, for i in range(k) for j in range(m) we have gradient:

        (dl / dA) = (dl / dy) @ (dy / dh) @ (dh / dA)

        where

        (dl / dy).shape = (b,1,n)
        (dy / dh).shape = (b,n,k)
        (dh / dA).shape = (b,k,k,m)

        @ operation can be done with Einstein summation:

        such as `np.einsum("bnk,bklm->bnlm", x,y, optimize=True)`

        refer to [EINSUM IS ALL YOU NEED - EINSTEIN SUMMATION IN DEEP LEARNING](https://rockt.github.io/2018/04/30/einsum)

        """
        assert self.data.shape[1:] == (1, 1), "Only scalars can do backward"
        for idx, child in enumerate(self.back_childs):
            if child.back_f is None:  # root node
                if child.requires_grad:
                    # multivariable function, detivate of idx-th variable
                    child.grad += self.back_f.gradient(self.back_childs, idx).transpose(
                        (0, 2, 1)
                    )
                    # count total case num
                    child.back_counter += self.shape[0]
                else:
                    pass
            else:
                child._bp(self.back_f.gradient(self.back_childs, idx))

    def _bp(self, grad_bp: np.ndarray):
        """private gradient back propagation calculation"""
        assert grad_bp.ndim <= 4, grad_bp.shape
        for idx, child in enumerate(self.back_childs):
            if child.back_f is None:
                if child.requires_grad:
                    grad = self.back_f.gradient(self.back_childs, idx)
                    # in recursion ermination condition, do `.squeeze(axis=1)` to match the shape
                    # and for batch train, we give mean grad of the batch
                    if child.grad.shape[-1] != 1:
                        child.grad += (
                            self.tensor_inner(grad_bp, grad)
                            .sum(axis=0, keepdims=True)
                            .squeeze(axis=1)
                        )
                    else:
                        child.grad += (
                            self.tensor_inner(grad_bp, grad)
                            .sum(axis=0, keepdims=True)
                            .transpose((0, 2, 1))
                        )
                    # count total case num
                    child.back_counter += self.shape[0]
                else:
                    pass
            else:
                # we **recursively** propagate gradient
                grad = self.back_f.gradient(self.back_childs, idx)
                child._bp(self.tensor_inner(grad_bp, grad))

    @staticmethod
    def tensor_inner(grad_bp, grad):
        """
        batch tensor inner product.
        """
        assert grad_bp.ndim == 3
        if grad.ndim == 4:
            return np.einsum("bnk,bklm->bnlm", grad_bp, grad)
        elif grad.ndim == 3:
            return np.einsum("bnk,bkl->bnl", grad_bp, grad)
        else:
            raise ValueError("ndim error")

    def __getattr__(self, name: str):
        """
        When attr not defined, try to find in `self.data`, for example `self.shape` or `self.ndim`
        """
        # without this statement, `self.data` will raise a RecursionError in pickle.loads
        if name == "data":
            return None
        return self.data.__getattribute__(name)

    def __repr__(self) -> str:
        return f"tensor({self.data}, requires_grad={self.requires_grad}, back_f={self.back_f})"

    def __neg__(self):
        """This will set requires_grad to False"""
        t = Tensor(-self.data)
        t.backf = self.back_f
        t.back_childs = self.back_childs
        return t

    @property
    def T(self):
        """
        Get transpose.
        This will set requires_grad to False
        """
        t = Tensor(self.data.transpose((0, 2, 1)))
        t.backf = self.back_f
        t.back_childs = self.back_childs
        return t


if __name__ == "__main__":
    from .functional import Inner
    x = Tensor(np.random.random((1, 3, 1)), requires_grad=True)
    loss = Inner()(x.T, x)  # this case is not considered
    loss.backward()
    print(x.grad)  # we will get a wrong grad.
