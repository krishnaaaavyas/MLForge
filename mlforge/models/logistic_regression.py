from mlforge.core.linear import Neuron
from mlforge.activations import Sigmoid


class LogisticRegression:

    def __init__(self):

        self.neuron = Neuron(
            weight=0.0,
            bias=0.0,
        )

        self.activation = Sigmoid()

        self.loss_history = []

    def predict_proba(self, X):

        probabilities = []

        for x in X:

            z = self.neuron.forward(x)

            probabilities.append(
                self.activation.forward(z)
            )

        return probabilities

    def predict(self, X):

        probabilities = self.predict_proba(X)

        predictions = []

        for probability in probabilities:

            if probability >= 0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions