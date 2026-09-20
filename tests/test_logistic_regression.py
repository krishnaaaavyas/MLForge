from mlforge.math.matrix import Matrix
from mlforge.models.logistic_regression import LogisticRegression


def test_logistic_regression_forward():

    model = LogisticRegression(
        in_features=2
    )

    X = Matrix([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (3, 1)

    for row in probabilities.data:
        assert 0.0 <= row[0] <= 1.0


def test_logistic_regression_predict():

    model = LogisticRegression(
        in_features=2
    )

    X = Matrix([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])

    predictions = model.predict(X)

    assert len(predictions) == 3

    assert all(
        prediction in [0, 1]
        for prediction in predictions
    )
def test_logistic_regression_known_parameters():

    model = LogisticRegression(
        in_features=2
    )

    model.linear.weight.value.data = [
        [2.0, 3.0]
    ]

    model.linear.bias.value.data = [
        1.0
    ]

    X = Matrix([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (3, 1)

    assert abs(probabilities.data[0][0] - 0.999876605) < 1e-6
    assert abs(probabilities.data[1][0] - 0.999999994) < 1e-8
    assert abs(probabilities.data[2][0] - 1.0) < 1e-12