"""
In this file we test nnn api, compared with torch api ^&^.
"""

import unittest
from numpy.random import random

from autograd import Tensor
from nn import *

import torch  # only for test, ensure computation of nnn is right


TEST_SIZE = 5
rand = lambda size: random(size=size) - 0.5


class TestAutograd(unittest.TestCase):

    def test_Linear(self):
        """
        y = sum(A @ x + b)
        calcute dy/dA and dy/db

        we use `sum` to test since its derivative is super simple.
        """

        # nnn api
        A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        b = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        mul = RightMultiply(rand((TEST_SIZE,)))
        sum = Sum()
        add = Add()
        y = sum(add(mul(A), b))
        y.backward()
        A_grad_nnn = A.grad
        b_grad_nnn = b.grad

        # torch api
        A = torch.from_numpy(A.data)
        A.requires_grad = True
        b = torch.from_numpy(b.data)
        b.requires_grad = True
        x = torch.from_numpy(mul.multiplier)
        y = (A @ x + b).sum()
        y.backward()
        A_grad_torch = A.grad.numpy()
        b_grad_torch = b.grad.numpy()

        # check float almost equal
        self.assertTrue(np.isclose(A_grad_nnn, A_grad_torch).all())
        self.assertTrue(np.isclose(b_grad_nnn, b_grad_torch).all())

    def test_double_linear(self):
        """
        y = sum(B @ (A @ x + b) + c)
        calcute dy/dA and dy/db

        we use `sum` to test since its derivative is super simple.
        """

        # nnn api
        A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        b = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        mul = RightMultiply(rand((TEST_SIZE,)))
        sum = Sum()
        add = Add()
        y = sum(add(mul(A), b))
        y.backward()
        A_grad_nnn = A.grad
        b_grad_nnn = b.grad

        # torch api
        A = torch.from_numpy(A.data)
        A.requires_grad = True
        b = torch.from_numpy(b.data)
        b.requires_grad = True
        x = torch.from_numpy(mul.multiplier)
        y = (A @ x + b).sum()
        y.backward()
        A_grad_torch = A.grad.numpy()
        b_grad_torch = b.grad.numpy()

        # check float almost equal
        self.assertTrue(np.isclose(A_grad_nnn, A_grad_torch).all())
        self.assertTrue(np.isclose(b_grad_nnn, b_grad_torch).all())

    def test_Activation(self):
        """
        y = sum(activation(h)), h = A @ x + b
        calcute dy/dA and dy/db
        """
        for act_nnn, act_torch in [
            (Softamx(), torch.nn.Softmax(dim=0)),
            (ReLU(), torch.relu),
        ]:
            # nnn api
            A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
            b = Tensor(rand((TEST_SIZE,)), requires_grad=True)
            mul = RightMultiply(rand((TEST_SIZE,)))
            add = Add()
            sum = Sum()
            h = add(mul(A), b)
            y = sum(act_nnn(h))
            y.backward()
            A_grad_nnn = A.grad
            b_grad_nnn = b.grad

            # torch api
            A = torch.from_numpy(A.data)
            A.requires_grad = True
            b = torch.from_numpy(b.data)
            b.requires_grad = True
            x = torch.from_numpy(mul.multiplier)
            y = act_torch(A @ x + b).sum()
            y.backward()
            A_grad_torch = A.grad.numpy()
            b_grad_torch = b.grad.numpy()
            # check float almost equal
            self.assertTrue(np.isclose(A_grad_nnn, A_grad_torch).all())
            self.assertTrue(np.isclose(b_grad_nnn, b_grad_torch).all())

    def test_Loss(self):
        """
        y = loss(A @ x + b - z)

        calcute dy/dA and dy/db
        """
        # nnn api
        A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        b = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        z = -Tensor(rand((TEST_SIZE,)))  # add netgative z
        mul = RightMultiply(rand((TEST_SIZE,)))
        norm = Norm()
        add = Add()
        y = norm(
            add(
                add(
                    mul(A),
                    b,
                ),
                z,
            )
        )
        y.backward()
        A_grad_nnn = A.grad
        b_grad_nnn = b.grad

        # torch api
        A = torch.from_numpy(A.data)
        A.requires_grad = True
        b = torch.from_numpy(b.data)
        b.requires_grad = True
        z = torch.from_numpy(z.data)
        x = torch.from_numpy(mul.multiplier)
        y = torch.norm(A @ x + b + z, p=2)
        y.backward()
        A_grad_torch = A.grad.numpy()
        b_grad_torch = b.grad.numpy()

        # check float almost equal
        self.assertTrue(np.isclose(A_grad_nnn, A_grad_torch).all())
        self.assertTrue(np.isclose(b_grad_nnn, b_grad_torch).all())


if __name__ == "__main__":
    unittest.main()
