import math

from mlforge.core.module import Module


class Sigmoid(Module):

    def forward(self, x):

        if isinstance(x, list):
            return [self.forward(v) for v in x]

        return 1 / (1 + math.exp(-x))