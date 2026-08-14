from mlforge.core.parameter import Parameter
from mlforge.math.vector import Vector
from mlforge.optimizers.gradient_descent import GradientDescent


def test_gradient_descent_updates_vector():

    weight = Parameter(
        Vector([1.0, 2.0, 3.0])
    )

    weight.grad = Vector([
        0.1,
        0.2,
        0.3,
    ])

    optimizer = GradientDescent(
        [weight],
        learning_rate=0.1,
    )

    optimizer.step()

    assert weight.value.data == [
        0.99,
        1.98,
        2.97,
    ]