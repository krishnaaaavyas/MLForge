from mlforge.core.linear import Linear


def test_module_parameters():

    layer = Linear(
        in_features=4,
        out_features=3
    )

    params = layer.parameters()

    assert len(params) == 2

    assert params[0] is layer.weight
    assert params[1] is layer.bias