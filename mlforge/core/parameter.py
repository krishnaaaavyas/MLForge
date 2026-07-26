class Parameter:
    """
    Represents a trainable parameter.
    """

    def __init__(self, value: float):

        self.value = value
        self.grad = 0.0

    def zero_grad(self):
        """
        Reset the stored gradient.
        """
        self.grad = 0.0

    def __repr__(self):
        return (
            f"Parameter(value={self.value}, grad={self.grad})"
        )