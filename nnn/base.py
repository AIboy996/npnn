"""base template"""


class Module:

    def __init__(self) -> None:
        pass

    def __call__(self, *x):
        """Forward call"""
        return self.forward(*x)

    def forward(self):
        raise NotImplementedError

    def gradient(self, x: tuple, idx: int, i=None, j=None):
        """Gradietn function"""
        raise NotImplementedError
