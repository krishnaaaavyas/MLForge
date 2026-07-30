from mlforge.core.linear import Linear


class LinearRegression:
    """
    Linear Regression model.

    Uses a single Linear layer internally.
    """

    def __init__(self, weight=0.0, bias=0.0):
        self.linear = Linear(weight, bias)

    def forward(self, x):
        return self.linear.forward(x)

    def predict(self, x):
        return self.forward(x)

    def backward(self, x, grad_output):
        """
        grad_output = dL/dy
        """

        self.linear.weight.grad += grad_output * x
        self.linear.bias.grad += grad_output

    def parameters(self):
        return self.linear.parameters()