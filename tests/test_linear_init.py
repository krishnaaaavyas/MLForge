from mlforge.core.linear import Linear


def test_linear_initialization():

    layer = Linear(
        in_features=4,
        out_features=3,
    )

    assert layer.weight.value.shape == (3, 4)
    assert layer.bias.value.shape == (3,)