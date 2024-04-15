"""base template"""
import numpy as np

class Module:

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *x):
        """Forward call"""
        return self.forward(*x)

    def forward(self):
        raise NotImplementedError

    def gradient(self) -> np.ndarray:
        """Gradietn function"""
        raise NotImplementedError

class NeuralNetwork(Module):

    def __init__(self) -> None:
        super().__init__()

    def gradient(self):
        raise NotImplementedError("NN don't have gradient.")
    
    def parameters(self) -> tuple:
        """Return all parameters to be optimized."""
        raise NotImplementedError