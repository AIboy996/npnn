"""model"""

from npnn import Tensor, np
import npnn.nn as nn
import npnn.functional as F


class FNN(nn.Module):
    def __init__(
        self, in_size, out_size, hidden_size: list[int] = [512, 256], activation=F.ReLU
    ):
        super().__init__()
        self.flatten = F.Flatten()
        hidden_layers = []
        for m, n in zip(
            (in_size, *hidden_size),
            (*hidden_size, out_size),
        ):
            hidden_layers.append(nn.Linear(m, n))
            hidden_layers.append(activation())

        self.linear_relu_stack = nn.Sequential(*hidden_layers, F.LogSoftmax())

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def parameters(self) -> list:
        return self.linear_relu_stack.parameters()


def test_NeuralNetwork():
    np.random.seed(0)
    model = FNN(28*28, 10)
    x = Tensor(np.random.random((1, 28, 28)))
    logits = model(x)
    loss = F.Norm()(logits)
    loss.backward()
    print(model)
    print(model.parameters()[-1].grad)
    print(loss)


if __name__ == "__main__":
    test_NeuralNetwork()
