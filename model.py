"""model"""

import numpy as np

from nnn import nn
from nnn import functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            F.ReLU(),
            nn.Linear(512, 512),
            F.ReLU(),
            nn.Linear(512, 10),
            F.Softmax(),
            F.Log()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def test_NeuralNetwork():
    model = NeuralNetwork()
    x = np.rand(1, 28, 28)
    logits = model(x)