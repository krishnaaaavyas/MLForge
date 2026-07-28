import math


class Sigmoid:
    """
    Sigmoid activation function.
    """

    def forward(self, x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def backward(self, predictions, targets):

        gradients = []

        n = len(predictions)

        for p, y in zip(predictions, targets):

            p = max(self.epsilon, min(1 - self.epsilon, p))

            grad = (p - y) / (p * (1 - p))

            gradients.append(grad / n)

        return gradients