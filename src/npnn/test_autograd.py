"""
In this file we test npnn api, compared with torch api ^&^.
"""

import unittest

from .base import np
from .autograd import Tensor
from .functional import *

import torch  # only for test, ensure computation of npnn is right


TEST_SIZE = 5
rand = lambda size: np.random.random(size=size) - 0.5


class TestAutograd(unittest.TestCase):

    def test_Flatten(self):
        """
        y = sum(x.flatten())
        """

        # npnn api
        x = Tensor(rand((3, 3)), requires_grad=True)
        y = Sum()(Flatten()(x))
        y.backward()
        grad_nnn = x.grad

        # torch api
        if np.__name__ == "cupy":
            x = torch.from_numpy(x.data.get())
        elif np.__name__ == "numpy":
            x = torch.from_numpy(x.data)
        x.requires_grad = True
        y = torch.sum(x.flatten())
        y.backward()
        grad_torh = x.grad.numpy()

        # check float almost equal
        self.assertTrue(np.allclose(grad_nnn, grad_torh))

    def test_Linear(self):
        """
        Test linear function:

        y = sum((A @ x) @ B + c)

        by calcuting dy/dA and dy/db.
        we use `sum` to test since its derivative is super simple.
        """

        # npnn api
        A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        B = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        c = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        x = Tensor(rand((TEST_SIZE,)))
        inner = Inner()
        sum = Sum()
        add = Add()
        y = sum(
            add(
                inner(inner(A, x), B),  # test left and right
                c,
            )
        )
        y.backward()
        A_grad_nnn = A.grad
        B_grad_nnn = B.grad
        c_grad_nnn = c.grad

        # torch api
        if np.__name__ == "cupy":
            A = torch.from_numpy(A.data.get())
            B = torch.from_numpy(B.data.get())
            c = torch.from_numpy(c.data.get())
            x = torch.from_numpy(x.data.get())
        elif np.__name__ == "numpy":
            A = torch.from_numpy(A.data)
            B = torch.from_numpy(B.data)
            c = torch.from_numpy(c.data)
            x = torch.from_numpy(x.data)
        A.requires_grad = True
        B.requires_grad = True
        c.requires_grad = True
        y = ((A @ x) @ B + c).sum()
        y.backward()
        A_grad_torch = A.grad.numpy()
        B_grad_torch = B.grad.numpy()
        c_grad_torch = c.grad.numpy()

        # check float almost equal
        self.assertTrue(np.allclose(A_grad_nnn, A_grad_torch))
        self.assertTrue(np.allclose(B_grad_nnn, B_grad_torch))
        self.assertTrue(np.allclose(c_grad_nnn, c_grad_torch))

    def test_Activation(self):
        """
        Test linear layer + acitvation function:

        y = sum(activation(h)), h = A @ x + b

        by calcuting dy/dA and dy/db.
        """
        for act_nnn, act_torch in [
            (Softmax(), torch.nn.Softmax(dim=0)),
            (LogSoftmax(), torch.nn.LogSoftmax(dim=0)),
            (ReLU(), torch.relu),
        ]:
            # npnn api
            A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
            b = Tensor(rand((TEST_SIZE,)), requires_grad=True)
            x = Tensor(rand((TEST_SIZE,)))
            inner = Inner()
            add = Add()
            sum = Sum()
            h = add(inner(A, x), b)
            y = sum(act_nnn(h))
            y.backward()
            A_grad_nnn = A.grad
            b_grad_nnn = b.grad

            # torch api
            if np.__name__ == "cupy":
                A = torch.from_numpy(A.data.get())
                b = torch.from_numpy(b.data.get())
                x = torch.from_numpy(x.data.get())
            elif np.__name__ == "numpy":
                A = torch.from_numpy(A.data)
                b = torch.from_numpy(b.data)
                x = torch.from_numpy(x.data)
            A.requires_grad = True
            b.requires_grad = True
            y = act_torch(A @ x + b).sum()
            y.backward()
            A_grad_torch = A.grad.numpy()
            b_grad_torch = b.grad.numpy()
            # check float almost equal
            self.assertTrue(np.allclose(A_grad_nnn, A_grad_torch))
            self.assertTrue(np.allclose(b_grad_nnn, b_grad_torch))

    def test_Loss(self):
        """
        Test linear layer + loss function:

        y = loss(A @ x + b - z)

        by calcuting dy/dA and dy/db.
        """
        # npnn api
        A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        b = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        z = -Tensor(rand((TEST_SIZE,)))  # add netgative z
        x = Tensor(rand((TEST_SIZE,)))
        inner = Inner()
        norm = Norm()
        add = Add()
        y = norm(
            add(
                add(
                    inner(A, x),
                    b,
                ),
                z,
            )
        )
        y.backward()
        A_grad_nnn = A.grad
        b_grad_nnn = b.grad

        # torch api
        if np.__name__ == "cupy":
            A = torch.from_numpy(A.data.get())
            b = torch.from_numpy(b.data.get())
            z = torch.from_numpy(z.data.get())
            x = torch.from_numpy(x.data.get())
        elif np.__name__ == "numpy":
            A = torch.from_numpy(A.data)
            b = torch.from_numpy(b.data)
            z = torch.from_numpy(z.data)
            x = torch.from_numpy(x.data)
        A.requires_grad = True
        b.requires_grad = True
        y = torch.norm(A @ x + b + z, p=2)
        y.backward()
        A_grad_torch = A.grad.numpy()
        b_grad_torch = b.grad.numpy()

        # check float almost equal
        self.assertTrue(np.allclose(A_grad_nnn, A_grad_torch))
        self.assertTrue(np.allclose(b_grad_nnn, b_grad_torch))

    def test_NN(self):
        """
        Test a Neural Network with 3-hidden layers:

        h1 = relu(A @ x + a)
        h2 = relu(B @ h1 + b)
        y_hat = relu(C @ h2 + c)

        choose L2 norm as loss function:

        l = norm(y_hat + (-y))

        """

        # npnn api
        A = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        B = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        C = Tensor(rand((TEST_SIZE, TEST_SIZE)), requires_grad=True)
        a = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        b = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        c = Tensor(rand((TEST_SIZE,)), requires_grad=True)
        x = Tensor(rand((TEST_SIZE,)))
        y = Tensor(rand((TEST_SIZE,)))
        inner = Inner()
        norm = Norm()
        add = Add()
        relu = ReLU()
        h1 = relu(add(inner(A, x), a))
        h2 = relu(add(inner(B, h1), b))
        y_hat = relu(add(inner(C, h2), c))
        l = norm(add(y_hat, (-y)))
        l.backward()
        nnn_grad = [A.grad, B.grad, C.grad, a.grad, b.grad, c.grad]

        # torch api
        if np.__name__ == "cupy":
            A = torch.from_numpy(A.data.get())
            B = torch.from_numpy(B.data.get())
            C = torch.from_numpy(C.data.get())
            a = torch.from_numpy(a.data.get())
            b = torch.from_numpy(b.data.get())
            c = torch.from_numpy(c.data.get())
            x = torch.from_numpy(x.data.get())
            y = torch.from_numpy(y.data.get())
        elif np.__name__ == "numpy":
            A = torch.from_numpy(A.data)
            B = torch.from_numpy(B.data)
            C = torch.from_numpy(C.data)
            a = torch.from_numpy(a.data)
            b = torch.from_numpy(b.data)
            c = torch.from_numpy(c.data)
            x = torch.from_numpy(x.data)
            y = torch.from_numpy(y.data)
        A.requires_grad = True
        B.requires_grad = True
        C.requires_grad = True
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        h1 = torch.relu(A @ x + a)
        h2 = torch.relu(B @ h1 + b)
        y_hat = torch.relu(C @ h2 + c)
        l = torch.norm(y_hat - y, p=2)
        l.backward()
        torch_grad = [A.grad, B.grad, C.grad, a.grad, b.grad, c.grad]

        # check float almost equal
        for g1, g2 in zip(nnn_grad, torch_grad):
            self.assertTrue(np.allclose(g1, g2))


if __name__ == "__main__":
    unittest.main()
