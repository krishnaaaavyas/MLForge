class MeanSquaredError:

    def forward(self, predictions, targets):
        if len(predictions) != len(targets):
            raise ValueError(
                "Predictions and targets must have the same length."
            )

        if len(predictions) == 0:
            raise ValueError(
                "Predictions and targets cannot be empty."
            )

        squared_errors = [
            (prediction - target) ** 2
            for prediction, target in zip(predictions, targets)
        ]

        return sum(squared_errors) / len(squared_errors)

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)