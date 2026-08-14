from mlforge.core.parameter import Parameter
from mlforge.math.matrix import Matrix
from mlforge.optimizers.gradient_descent import GradientDescent


def test_gradient_descent_updates_matrix():

    weight = Parameter(
        Matrix([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
    )

    weight.grad = Matrix([
        [0.1, 0.2],
        [0.3, 0.4],
    ])

    optimizer = GradientDescent(
        [weight],
        learning_rate=0.1,
    )

    optimizer.step()

    assert weight.value.data == [
        [0.99, 1.98],
        [2.97, 3.96],
    ]