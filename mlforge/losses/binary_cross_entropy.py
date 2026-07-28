import math


class BinaryCrossEntropy:
    """
    Binary Cross Entropy Loss.
    """

    def __init__(self):
        self.epsilon = 1e-15

    def forward(self, predictions, targets):

        loss = 0

        for p, y in zip(predictions, targets):

            p = max(self.epsilon, min(1 - self.epsilon, p))

            loss += -(
                y * math.log(p)
                + (1 - y) * math.log(1 - p)
            )

        return loss / len(predictions)