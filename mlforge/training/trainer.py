
class Trainer:

    def __init__(
        self,
        loss_fn,
        optimizer,
        epochs=100,
    ):

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs