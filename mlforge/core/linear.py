from .module import Module
from .parameter import Parameter


class Linear(Module):

    def __init__(self, weight=0.0, bias=0.0):

        self.weight = Parameter(weight)
        self.bias = Parameter(bias)

    def forward(self, x):

        return self.weight.value * x + self.bias.value

    def parameters(self):

        return [
            self.weight,
            self.bias,
        ]