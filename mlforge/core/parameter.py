class Parameter:
    """
    A trainable parameter.

    The value can be:
        - float
        - Vector
        - Matrix

    Gradients always mirror the same structure.
    """

    def __init__(self, value):
        self.value = value
        self.grad = None

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        return (
            f"Parameter(value={self.value}, grad={self.grad})"
        )