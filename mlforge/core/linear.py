from random import uniform

from mlforge.core.module import Module
from mlforge.core.parameter import Parameter
from mlforge.math.matrix import Matrix
from mlforge.math.vector import Vector


class Linear(Module):

    def __init__(self, in_features, out_features):

        if in_features <= 0:
            raise ValueError("in_features must be positive.")

        if out_features <= 0:
            raise ValueError("out_features must be positive.")

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(
            Matrix(
                [
                    [
                        uniform(-0.1, 0.1)
                        for _ in range(in_features)
                    ]
                    for _ in range(out_features)
                ]
            )
        )

        self.bias = Parameter(
            Vector(
                [
                    uniform(-0.1, 0.1)
                    for _ in range(out_features)
                ]
            )
        )

    def forward(self, x: Matrix):

        if not isinstance(x, Matrix):
            raise TypeError(
                "Linear.forward() expects a Matrix."
            )

        if x.cols != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} features but got {x.cols}."
            )

        output = x @ self.weight.value.T
        output = output + self.bias.value

        return output

    def parameters(self):

        return [
            self.weight,
            self.bias,
        ]