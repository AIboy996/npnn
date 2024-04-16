"""Provide gradietn descent optimizer"""

from .base import Optimizer
from .autograd import Tensor


class SGD(Optimizer):

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        super().__init__()
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.step_now = 0

    def step(self):
        """refer to https://pytorch.org/docs/stable/generated/torch.optim.SGD.html"""
        for param in self.params:
            x = param.data
            g = param.grad
            if self.weight_decay != 0:
                g = g + self.weight_decay * x
            if self.momentum != 0:
                # param.b is the momentum cache
                if self.step_now > 0:
                    param.b = self.momentum * param.b + (1 - self.dampening) * g
                else:
                    param.b = g
                if self.nesterov:
                    g = g + self.momentum * param.b
                else:
                    g = param.b
            # update
            param.data = x - self.lr * g
        self.step_now += 1


def test_SGD_Regression():
    """
    y_hat = X @ b, find b to minimize norm(y - y_hat)
    """
    import numpy as np
    import scipy
    from .functional import Inner, Norm, Add

    np.random.seed(0)
    inner = Inner()
    add = Add()
    norm = Norm()
    rand = np.random.random
    SIZE = 5
    X = Tensor(rand((SIZE, SIZE)) * 2)
    beta = Tensor(rand(SIZE), requires_grad=True)
    y = Tensor(4 * (rand(SIZE) - 0.5))
    optmizer = SGD(
        params=[beta], momentum=0.1, dampening=0.5, lr=0.001, weight_decay=0.01
    )
    for epoch in range(100000):
        loss = norm(add(inner(X, beta), -y))
        loss.backward()
        optmizer.step()
    res = scipy.optimize.lsq_linear(X, y)
    print(res.x, res.cost, res.status)
    print(beta.data, loss.data[0])


if __name__ == "__main__":
    test_SGD_Regression()
