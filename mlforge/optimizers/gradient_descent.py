class GradientDescent:

    def __init__(self, learning_rate=0.01):

        self.learning_rate = learning_rate

    def step(self, parameters):

        for parameter in parameters:

            parameter.value -= (
                self.learning_rate
                * parameter.grad
            )

    def zero_grad(self, parameters):

        for parameter in parameters:

            parameter.grad = 0.0