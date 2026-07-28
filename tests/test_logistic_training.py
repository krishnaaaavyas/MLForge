from mlforge.models import LogisticRegression
from mlforge.losses import BinaryCrossEntropy
from mlforge.optimizers import GradientDescent


def test_logistic_training():

    X = [1,2,3,4,5,6,7,8]

    y = [0,0,0,0,1,1,1,1]

    model = LogisticRegression()

    loss_fn = BinaryCrossEntropy()

    optimizer = GradientDescent(0.5)

    initial = loss_fn.forward(
        model.forward(X),
        y
    )

    for _ in range(500):

        predictions = model.forward(X)

        gradients = loss_fn.backward(
            predictions,
            y
        )

        model.backward(
            X,
            gradients
        )

        optimizer.step(
            model.parameters()
        )

        optimizer.zero_grad(
            model.parameters()
        )

    final = loss_fn.forward(
        model.forward(X),
        y
    )

    assert final < initial