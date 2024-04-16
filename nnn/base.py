"""Base classes"""
import numpy as np

class Operation:

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

class Module:

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, *x):
        """Forward call"""
        return self.forward(*x)
    
    def parameters(self) -> list:
        """Return all parameters to be optimized."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__}"

class Optimizer:

    def __init__(self) -> None:
        super().__init__()
    
    def step(self):
        """Take a step on gradient direction."""
        raise NotImplementedError

    def zero_grad(self):
        """Only should call this if you want to perform mini-batch gradient descent."""
        for param in self.params:
            param.grad = np.zeros_like(param.data)
            param.back_counter = 0