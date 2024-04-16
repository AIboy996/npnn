"""Provide gradietn descent optimizer"""

import numpy as np

from .base import Optimizer
from .autograd import Tensor


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    refer to https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    """

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
        for param in self.params:
            x = param.data
            g = param.grad / param.back_counter
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


class Adam(Optimizer):
    """
    Adam optimizer is faster than SGD in most time.
    refer to
        https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
    and
        https://arxiv.org/pdf/1412.6980.pdf
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.001,
        betas: list[float] = [0.9, 0.999],
        eps: float = 1e-8,
        weight_decay: float = 0,
    ) -> None:
        super().__init__()
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_now = 0

    def step(self):
        for param in self.params:
            beta1, beta2 = self.betas
            x = param.data
            g = param.grad / param.back_counter
            if self.weight_decay != 0:
                g = g + self.weight_decay * x
            if self.step_now > 0:
                # Update biased first moment estimate)
                param.m = beta1 * param.m + (1 - beta1) * g
                # Update biased second raw moment estimate
                param.v = beta2 * param.v + (1 - beta2) * g**2
            else:
                param.m = np.zeros_like(g)
                param.v = np.zeros_like(g)
            # Correct bias
            m = param.m / (1 - pow(beta1, self.step_now + 1))
            v = param.v / (1 - pow(beta2, self.step_now + 1))
            # update
            param.data = x - self.lr * m / (np.sqrt(v) + self.eps)
        self.step_now += 1


def test_Regression(sgd=False, iterations=10_000):
    """
    Test on Linear Regression problem, compared with scipy.optimize.lsq_linear
    
    find b to minimize `norm(y - y_hat)` where `y_hat = X @ b`
    
    """
    import numpy as np
    import scipy
    from .functional import Inner, Norm, Add

    np.random.seed(0)
    inner = Inner()
    add = Add()
    norm = Norm()
    rand = np.random.random
    SIZE = 10
    X = Tensor(rand((SIZE, SIZE)) * 2)
    beta = Tensor(rand(SIZE), requires_grad=True)
    y = Tensor(4 * (rand(SIZE) - 0.5))
    if sgd:
        optmizer = SGD(
            params=[beta], momentum=0.1, dampening=0.5, lr=0.001, weight_decay=0.01
        )
    else:
        optmizer = Adam(params=[beta], lr=0.001, weight_decay=0.01)
    for _ in range(iterations):
        loss = norm(add(inner(X, beta), -y))
        loss.backward()
        optmizer.step()
    res = scipy.optimize.lsq_linear(X, y)
    print(res.x, res.cost, res.status)
    print(beta.data, loss.data[0])


if __name__ == "__main__":
    print("SGD optimization")
    test_Regression(True, 10000)
    print("Adam optimization")
    test_Regression(False, 10000)
