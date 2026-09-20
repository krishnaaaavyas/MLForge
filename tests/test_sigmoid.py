from mlforge.activations.sigmoid import Sigmoid
from mlforge.math.matrix import Matrix
from mlforge.math.vector import Vector


def test_sigmoid_scalar():

    sigmoid = Sigmoid()

    result = sigmoid.forward(0)

    assert result == 0.5


def test_sigmoid_vector():

    sigmoid = Sigmoid()

    result = sigmoid.forward(
        Vector([0, 1])
    )

    assert len(result) == 2

    assert abs(result[0] - 0.5) < 1e-9
    assert abs(result[1] - 0.7310585786) < 1e-9


def test_sigmoid_matrix():

    sigmoid = Sigmoid()

    result = sigmoid.forward(
        Matrix([
            [0, 1],
            [-1, 2],
        ])
    )

    assert result.shape == (2, 2)

    assert abs(result.data[0][0] - 0.5) < 1e-9
    assert abs(result.data[0][1] - 0.7310585786) < 1e-9
    assert abs(result.data[1][0] - 0.2689414214) < 1e-9
    assert abs(result.data[1][1] - 0.8807970779) < 1e-9