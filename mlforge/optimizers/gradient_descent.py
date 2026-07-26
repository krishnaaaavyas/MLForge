class GradientDescent:

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate


    def step(self, parameter):

        parameter.value -= (
            self.learning_rate * parameter.grad
        )

    def step(self, weight, gradient):
        return weight - self.learning_rate * gradient

