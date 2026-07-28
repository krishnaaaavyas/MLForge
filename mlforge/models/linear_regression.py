from mlforge.core.linear import Neuron


class LinearRegression:

    def __init__(self):

        self.neuron = Neuron(
            weight=0,
            bias=0,
        )

        self.loss_history = []

    def forward(self, X):

        return [
            self.neuron.forward(x)
            for x in X
        ]