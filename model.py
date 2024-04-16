"""model"""

import numpy as np

from nnn import Tensor
import nnn.nn as nn
import nnn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = F.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            F.ReLU(),
            nn.Linear(512, 512),
            F.ReLU(),
            nn.Linear(512, 10),
            F.LogSoftmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

    def parameters(self) -> list:
        return self.linear_relu_stack.parameters()

def test_NeuralNetwork():
    model = NeuralNetwork()
    x = Tensor(np.random.random((1, 28, 28)))
    logits = model(x)
    l = F.Norm()(logits)
    l.backward()
    print(logits, l)

if __name__ == '__main__':
    test_NeuralNetwork()