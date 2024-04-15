"""Test nnn api, compared with torch"""

import unittest
from numpy.random import random as rand

from autograd import Tensor
from nn import *

import torch  # only for test, ensure computation of nnn is right


class TestAutograd(unittest.TestCase):

    def test_RightMultiply(self):
        """y = sum(X @ c)"""
        # nnn api
        X = Tensor(rand((3, 3)), requires_grad=True)
        mul = RightMultiply(rand((3,)))
        sum = Sum()
        y = sum(mul(X))
        y.backward()
        grad_nnn = X.grad
        # torch api
        X = torch.from_numpy(X.data)
        X.requires_grad = True
        c = torch.from_numpy(mul.multiplier)
        y = (X @ c).sum()
        y.backward()
        grad_torch = X.grad.numpy()

        self.assertTrue((grad_nnn == grad_torch).all())

    def test_LeftMultiply(self):
        """y = sum(c @ X)"""
        # nnn api
        X = Tensor(rand((3, 3)), requires_grad=True)
        mul = LeftMultiply(rand((3,)))
        sum = Sum()
        y = sum(mul(X))
        y.backward()
        grad_nnn = X.grad
        # torch api
        X = torch.from_numpy(X.data)
        X.requires_grad = True
        c = torch.from_numpy(mul.multiplier)
        y = (c @ X).sum()
        y.backward()
        grad_torch = X.grad.numpy()

        self.assertTrue((grad_nnn == grad_torch).all())


if __name__ == "__main__":
    unittest.main()
