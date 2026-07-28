class Module:
    """
    Base class for every trainable object.
    """

    def parameters(self):
        """
        Returns every trainable Parameter.
        """
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError