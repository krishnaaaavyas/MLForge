from mlforge.math.matrix import Matrix
from mlforge.math.vector import Vector

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

def test_matrix_transpose():

    matrix = Matrix([
        [1, 2, 3],
        [4, 5, 6],
    ])

    result = matrix.T

    assert result.shape == (3, 2)

    assert result.data == [
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
    ]

def test_matrix_vector_broadcast_addition():

    matrix = Matrix([
        [10, 20],
        [30, 40],
        [50, 60],
    ])

    vector = Vector([
        1,
        2,
    ])

    result = matrix + vector

    assert result.shape == (3, 2)

    assert result.data == [
        [11.0, 22.0],
        [31.0, 42.0],
        [51.0, 62.0],
    ]