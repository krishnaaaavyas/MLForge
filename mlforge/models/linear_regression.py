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

    def backward(self, X, gradients):
    
        dw = 0.0
        db = 0.0

        n = len(X)

        for x, grad in zip(X, gradients):

            dw += grad * x
            db += grad

        self.linear.weight.grad = dw / n
        self.linear.bias.grad = db / n

    def parameters(self):
        return self.linear.parameters()