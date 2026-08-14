from mlforge.core.linear import Linear
from mlforge.core.module import Module
from mlforge.activations.sigmoid import Sigmoid


class LogisticRegression(Module):

    def __init__(self, in_features=1):
        super().__init__()

        self.linear = Linear(
            in_features=in_features,
            out_features=1,
        )

        self.sigmoid = Sigmoid()

    def forward(self, x):
        z = self.linear.forward(x)
        return self.sigmoid.forward(z)

    def predict_proba(self, x):
        return self.forward(x)

    def predict(self, x, threshold=0.5):
        probabilities = self.predict_proba(x)

        return [
            1 if probability >= threshold else 0
            for probability in probabilities
        ]

    def parameters(self):
        return self.linear.parameters()