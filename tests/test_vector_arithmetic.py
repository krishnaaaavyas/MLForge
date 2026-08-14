from mlforge.math.vector import Vector


def test_vector_scalar_multiplication():

    vector = Vector([1, 2, 3])

    result = vector * 2

    assert result.data == [2.0, 4.0, 6.0]


def test_scalar_vector_multiplication():

    vector = Vector([1, 2, 3])

    result = 2 * vector

    assert result.data == [2.0, 4.0, 6.0]


def test_vector_subtraction():

    a = Vector([5, 6, 7])
    b = Vector([1, 2, 3])

    result = a - b

    assert result.data == [4.0, 4.0, 4.0]