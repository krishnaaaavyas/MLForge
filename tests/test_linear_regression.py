from mlforge.math.matrix import Matrix
from mlforge.models.linear_regression import LinearRegression


def test_linear_regression_forward():

    model = LinearRegression(
        in_features=2
    )

    X = Matrix([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    predictions = model.forward(X)

    assert predictions.shape == (3, 1)