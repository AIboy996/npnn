"""Provide gradietn descent optimizer"""

from .base import Optimizer


class SGD(Optimizer):

    def __init__(self) -> None:
        super().__init__()
