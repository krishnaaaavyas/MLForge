from mlforge.core.linear import Linear
from mlforge.core.module import Module
from mlforge.math.matrix import Matrix
from mlforge.math.vector import Vector
from mlforge.activations.sigmoid import Sigmoid


class LogisticRegression(Module):

    def __init__(self, in_features=1):

        super().__init__()

        if in_features <= 0:
            raise ValueError(
                "in_features must be positive."
            )

        self.linear = Linear(
            in_features=in_features,
            out_features=1,
        )

        self.sigmoid = Sigmoid()

    def forward(self, x: Matrix):

        z = self.linear.forward(x)

        return self.sigmoid.forward(z)

    def predict_proba(self, x: Matrix):

        return self.forward(x)

    def predict(self, x: Matrix, threshold=0.5):

        probabilities = self.predict_proba(x)

        predictions = []

        for row in probabilities.data:

            predictions.append(
                1 if row[0] >= threshold else 0
            )

        return predictions

    def parameters(self):

        return self.linear.parameters()