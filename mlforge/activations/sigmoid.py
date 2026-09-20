import math

from mlforge.core.module import Module
from mlforge.math.matrix import Matrix
from mlforge.math.vector import Vector


class Sigmoid(Module):

    @staticmethod
    def _sigmoid_scalar(x):
        return 1 / (1 + math.exp(-x))

    def forward(self, x):

        if isinstance(x, (int, float)):
            return self._sigmoid_scalar(x)

        if isinstance(x, Vector):
            return Vector([
                self._sigmoid_scalar(value)
                for value in x.data
            ])

        if isinstance(x, Matrix):
            return Matrix([
                [
                    self._sigmoid_scalar(value)
                    for value in row
                ]
                for row in x.data
            ])

        raise TypeError(
            "Sigmoid.forward() expects a scalar, Vector, or Matrix."
        )