from mlforge.losses.mse import calculate_mse


class Trainer:

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def fit(
        self,
        model,
        X,
        y,
        epochs=100,
    ):

        n = len(X)

        for epoch in range(epochs):

            predictions = model.predict(X)

            loss = calculate_mse(predictions, y)

            model.loss_history.append(loss)

            dw = 0.0
            db = 0.0

            for x, actual, pred in zip(X, y, predictions):

                error = pred - actual

                dw += error * x

                db += error

            dw = (2 / n) * dw
            db = (2 / n) * db

            model.neuron.weight.grad = dw
            model.neuron.bias.grad = db

            self.optimizer.step(model.neuron.weight)
            self.optimizer.step(model.neuron.bias)

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:3} "
                    f"Loss={loss:.4f} "
                    f"W={model.neuron.weight.value:.4f} "
                    f"B={model.neuron.bias.value:.4f}"
                )