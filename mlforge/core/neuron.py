from mlforge.core.parameter import Parameter


class Neuron:

    def __init__(self, weight, bias):

        self.weight = Parameter(weight)
        self.bias = Parameter(bias)

    def forward(self, x):

        return self.weight.value * x + self.bias.value