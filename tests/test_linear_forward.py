from mlforge.core.linear import Linear
from mlforge.math.matrix import Matrix


def test_linear_forward():

    layer = Linear(
        in_features=4,
        out_features=3,
    )

    x = Matrix(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]
    )

    output = layer.forward(x)

    assert output.shape == (2, 3)