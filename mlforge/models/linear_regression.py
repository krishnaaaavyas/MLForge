from mlforge.core.linear import Linear
from mlforge.core.module import Module


class LinearRegression(Module):

    def __init__(self, in_features=1):
        super().__init__()

        self.linear = Linear(
            in_features=in_features,
            out_features=1,
        )

    def forward(self, x):
        return self.linear.forward(x)

    def parameters(self):
        return self.linear.parameters()