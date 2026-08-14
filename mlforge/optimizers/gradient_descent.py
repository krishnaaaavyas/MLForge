class GradientDescent:

    def __init__(self, parameters, learning_rate=0.01):

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive."
            )

        self.parameters = list(parameters)
        self.learning_rate = learning_rate

    def step(self):

        for parameter in self.parameters:

            if parameter.grad is None:
                continue

            parameter.value = (
                parameter.value
                - self.learning_rate * parameter.grad
            )

    def zero_grad(self):

        for parameter in self.parameters:
            parameter.zero_grad()