from mlforge.core.parameter import Parameter
from mlforge.optimizers.gradient_descent import GradientDescent


def test_gradient_descent_step():

    weight = Parameter(10.0)
    weight.grad = 2.0

    optimizer = GradientDescent(
        [weight],
        learning_rate=0.1,
    )

    optimizer.step()

    assert weight.value == 9.8


def test_gradient_descent_zero_grad():

    weight = Parameter(10.0)
    weight.grad = 2.0

    optimizer = GradientDescent(
        [weight],
        learning_rate=0.1,
    )

    optimizer.zero_grad()

    assert weight.grad is None