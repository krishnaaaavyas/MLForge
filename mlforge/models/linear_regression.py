from mlforge.core.linear import Linear
from mlforge.core.module import Module
from mlforge.math.matrix import Matrix


class LinearRegression(Module):

    def __init__(self, in_features=1):

        super().__init__()

        if in_features <= 0:
            raise ValueError(
                "in_features must be positive."
            )

        self.linear = Linear(
            in_features=in_features,
            out_features=1,
        )

    def forward(self, x: Matrix):

        return self.linear.forward(x)

    def parameters(self):

        return self.linear.parameters()

    def predict(self, x: Matrix):

        return self.forward(x)