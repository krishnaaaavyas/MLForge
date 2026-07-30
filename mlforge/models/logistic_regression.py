from mlforge.core.linear import Linear
from mlforge.activations.sigmoid import Sigmoid


class LogisticRegression:
    """
    Logistic Regression

    Linear
        ↓
    Sigmoid
    """

    def __init__(self, weight=0.0, bias=0.0):
        self.linear = Linear(weight, bias)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        z = self.linear.forward(x)
        return self.sigmoid.forward(z)

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x):
        return 1 if self.forward(x) >= 0.5 else 0

    def backward(self, x, grad_output):
        """
        grad_output = dL/dz
        """

        self.linear.weight.grad += grad_output * x
        self.linear.bias.grad += grad_output

    def parameters(self):
        return self.linear.parameters()