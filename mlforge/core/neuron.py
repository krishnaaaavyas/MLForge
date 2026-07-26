class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights      # List of weights
        self.bias = bias

    def predict(self, inputs):
        """
        Calculate:
        w1*x1 + w2*x2 + ... + wn*xn + bias
        """

        score = 0

        for weight, value in zip(self.weights, inputs):
            score += weight * value

        score += self.bias

        return score


if __name__ == "__main__":

    neuron = Neuron(
        weights=[2, 5],
        bias=-8
    )

    student = [6, 0.8]

    prediction = neuron.predict(student)

    print(prediction)