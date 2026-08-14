from mlforge.math.matrix import Matrix


def test_matrix_scalar_multiplication():

    matrix = Matrix([
        [1, 2],
        [3, 4],
    ])

    result = matrix * 2

    assert result.data == [
        [2.0, 4.0],
        [6.0, 8.0],
    ]


def test_scalar_matrix_multiplication():

    matrix = Matrix([
        [1, 2],
        [3, 4],
    ])

    result = 2 * matrix

    assert result.data == [
        [2.0, 4.0],
        [6.0, 8.0],
    ]


def test_matrix_subtraction():

    a = Matrix([
        [5, 6],
        [7, 8],
    ])

    b = Matrix([
        [1, 2],
        [3, 4],
    ])

    result = a - b

    assert result.data == [
        [4.0, 4.0],
        [4.0, 4.0],
    ]


def test_matrix_addition():

    a = Matrix([
        [1, 2],
        [3, 4],
    ])

    b = Matrix([
        [5, 6],
        [7, 8],
    ])

    result = a + b

    assert result.data == [
        [6.0, 8.0],
        [10.0, 12.0],
    ]