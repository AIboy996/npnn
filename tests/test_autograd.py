"""
In this file we test npnn api, compared with torch api ^&^.
"""

import unittest

from npnn.base import np
from npnn.autograd import Tensor
from npnn.functional import (
    Add,
    Inner,
    Flatten,
    Sum,
    Norm,
    NLL,
    Log,
    Softmax,
    LogSoftmax,
    ReLU,
)

import torch  # only for test, ensure computation of npnn is right

TEST_BATCH_SIZE = 8
TEST_SIZE = 5


def rand(size):
    return np.random.random(size=size) - 0.5


class TestAutograd(unittest.TestCase):

    def test_Norm(self):
        # npnn api
        A = Tensor(np.random.random((1, TEST_SIZE, TEST_SIZE)), requires_grad=True)
        x = Tensor(np.random.random((TEST_BATCH_SIZE, TEST_SIZE, 1)))
        loss = Norm()(Inner()(A, x))
        loss.backward()
        l_npnn = loss.data.sum()
        grad_npnn = A.grad

        # torch api
        if np.__name__ == "cupy":
            A = torch.from_numpy(A.data.get())
            x = torch.from_numpy(x.data.get())
        elif np.__name__ == "numpy":
            A = torch.from_numpy(A.data)
            x = torch.from_numpy(x.data)
        A.requires_grad = True
        # find norm in each batch, not global norm
        loss = torch.norm(A @ x, p=2, dim=(1, 2)).sum()
        loss.backward()
        l_torch = loss.detach().numpy()
        grad_torch = A.grad.numpy()

        self.assertTrue(np.allclose(l_npnn, l_torch))
        self.assertTrue(np.allclose(grad_npnn, grad_torch))

    def test_SumLog(self):
        """
        y = sum(log(A @ x))
        """

        # npnn api
        A = Tensor(np.random.random((1, TEST_SIZE, TEST_SIZE)), requires_grad=True)
        x = Tensor(np.random.random((TEST_BATCH_SIZE, TEST_SIZE, 1)))
        loss = Sum()(Log()(Inner()(A, x)))
        loss.backward()
        l_npnn = loss.data.sum()
        grad_npnn = A.grad

        # torch api
        if np.__name__ == "cupy":
            A = torch.from_numpy(A.data.get())
            x = torch.from_numpy(x.data.get())
        elif np.__name__ == "numpy":
            A = torch.from_numpy(A.data)
            x = torch.from_numpy(x.data)
        A.requires_grad = True
        loss = torch.sum((A @ x).log())
        loss.backward()
        l_torch = loss.detach().numpy()
        grad_torch = A.grad.numpy()

        self.assertTrue(np.allclose(l_npnn, l_torch))
        self.assertTrue(np.allclose(grad_npnn, grad_torch))

    def test_Flatten(self):
        """
        y = sum(x.flatten())
        """

        # npnn api
        x = Tensor(rand((1, TEST_SIZE, TEST_SIZE)), requires_grad=True)
        y = Sum()(Flatten()(x))
        y.backward()
        grad_npnn = x.grad

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
        self.assertTrue(np.allclose(grad_npnn, grad_torh))

    def test_Linear(self):
        """
        Test linear function:

        y = sum(A @ x + b)

        by calcuting dy/dA and dy/db.
        we use `sum` to test since its derivative is super simple.
        """

        # npnn api
        A = Tensor(rand((1, TEST_SIZE, TEST_SIZE)), requires_grad=True)
        b = Tensor(rand((1, TEST_SIZE, 1)), requires_grad=True)
        x = Tensor(rand((TEST_BATCH_SIZE, TEST_SIZE, 1)))
        inner = Inner()
        sum = Sum()
        add = Add()
        y = sum(add(inner(A, x), b))
        y.backward()
        A_grad_npnn = A.grad
        b_grad_npnn = b.grad

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
        y = (A @ x + b).sum()
        y.backward()
        A_grad_torch = A.grad.numpy()
        b_grad_torch = b.grad.numpy()

        # check float almost equal
        self.assertTrue(np.allclose(A_grad_npnn, A_grad_torch))
        self.assertTrue(np.allclose(b_grad_npnn, b_grad_torch))

    def test_Activation(self):
        """
        Test linear layer + acitvation function:

        y = norm2(activation(h)), h = A @ x + b

        by calcuting dy/dA and dy/db.
        """
        for act_npnn, act_torch in [
            (Softmax(), torch.nn.Softmax(dim=1)),
            (LogSoftmax(), torch.nn.LogSoftmax(dim=1)),
            (ReLU(), torch.relu),
        ]:
            # npnn api
            A = Tensor(rand((1, TEST_SIZE, TEST_SIZE)), requires_grad=True)
            b = Tensor(rand((1, TEST_SIZE, 1)), requires_grad=True)
            x = Tensor(rand((1, TEST_SIZE, 1)))
            inner = Inner()
            add = Add()
            norm = Norm()
            h = add(inner(A, x), b)
            y = norm(act_npnn(h))
            y.backward()
            A_grad_npnn = A.grad
            b_grad_npnn = b.grad
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
            y = torch.norm(act_torch(A @ x + b), p=2, dim=(1, 2)).sum()
            y.backward()
            A_grad_torch = A.grad.numpy()
            b_grad_torch = b.grad.numpy()
            # check float almost equal
            self.assertTrue(np.allclose(A_grad_npnn, A_grad_torch))
            self.assertTrue(np.allclose(b_grad_npnn, b_grad_torch))

    def test_NN(self):
        """
        Test a Neural Network with 3-hidden layers:

        h1 = relu(A @ x + a)
        h2 = relu(B @ h1 + b)
        h3 = C @ h2 + c
        y_hat = log_softmax(h3)

        choose NLL as loss function:

        l = NLLLoss(y_hat, y)

        Note that this is equivalent to:

        l = CrossEntropyLoss(h3, y)
        """

        # npnn api
        A = Tensor(rand((1, TEST_SIZE * 2, TEST_SIZE * 3)), requires_grad=True)
        B = Tensor(rand((1, TEST_SIZE, TEST_SIZE * 2)), requires_grad=True)
        C = Tensor(rand((1, TEST_SIZE, TEST_SIZE)), requires_grad=True)
        a = Tensor(rand((1, TEST_SIZE * 2, 1)), requires_grad=True)
        b = Tensor(rand((1, TEST_SIZE, 1)), requires_grad=True)
        c = Tensor(rand((1, TEST_SIZE, 1)), requires_grad=True)
        x = Tensor(rand((TEST_BATCH_SIZE, TEST_SIZE * 3, 1)))
        y = Tensor(rand((TEST_BATCH_SIZE, TEST_SIZE, 1)))  # target is probailbity map
        inner = Inner()
        nll = NLL()
        add = Add()
        relu = ReLU()
        log_softmax = LogSoftmax()
        h1 = relu(add(inner(A, x), a))
        h2 = relu(add(inner(B, h1), b))
        y_hat_npnn = log_softmax(add(inner(C, h2), c))
        loss = nll(y_hat_npnn, y)
        loss.backward()
        npnn_grad = [A.grad, B.grad, C.grad, a.grad, b.grad, c.grad]

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
            y = torch.from_numpy(y.data).long()
        A.requires_grad = True
        B.requires_grad = True
        C.requires_grad = True
        a.requires_grad = True
        b.requires_grad = True
        c.requires_grad = True
        h1 = torch.relu(A @ x + a)
        h2 = torch.relu(B @ h1 + b)
        h3 = C @ h2 + c
        loss = torch.nn.CrossEntropyLoss(reduction="sum")(h3, y)
        loss.backward()
        torch_grad = [A.grad, B.grad, C.grad, a.grad, b.grad, c.grad]

        # check float almost equal
        for g1, g2 in zip(npnn_grad, torch_grad):
            self.assertTrue(np.allclose(g1, g2))


if __name__ == "__main__":
    unittest.main()
