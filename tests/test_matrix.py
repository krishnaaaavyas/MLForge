# tests/test_matrix.py
import pytest
from mlforge.math.vector import Vector
from mlforge.math.matrix import Matrix

def test_matrix_vector_matmul():
    # Weight matrix: 3 neurons, 4 features (3x4)
    W = Matrix([
        [2.0, 1.0, 0.5, 3.0],
        [0.0, 1.5, 2.0, 1.0],
        [1.0, 0.0, 4.0, 0.5]
    ])
    
    # Input vector x: 4 features
    x = Vector([1.0, 2.0, 3.0, 4.0])
    
    # Expected calculations:
    # y0 = 2*1 + 1*2 + 0.5*3 + 3*4 = 2 + 2 + 1.5 + 12 = 17.5
    # y1 = 0*1 + 1.5*2 + 2*3 + 1*4 = 0 + 3 + 6 + 4 = 13.0
    # y2 = 1*1 + 0*2 + 4*3 + 0.5*4 = 1 + 0 + 12 + 2 = 15.0
    
    res = W @ x
    assert isinstance(res, Vector)
    assert len(res) == 3
    assert res.data == [17.5, 13.0, 15.0]

def test_matrix_matrix_matmul():
    # A is (2 x 3)
    A = Matrix([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])
    
    # B is (3 x 2)
    B = Matrix([
        [7.0, 8.0],
        [9.0, 1.0],
        [2.0, 3.0]
    ])
    
    # Expected C = A @ B (2 x 2):
    # C[0][0] = 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
    # C[0][1] = 1*8 + 2*1 + 3*3 = 8 + 2 + 9 = 19
    # C[1][0] = 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
    # C[1][1] = 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    
    C = A @ B
    assert C.shape == (2, 2)
    assert C.data == [
        [31.0, 19.0],
        [85.0, 55.0]
    ]

def test_matrix_shape_mismatch():
    A = Matrix([[1, 2], [3, 4]]) # 2x2
    x = Vector([1, 2, 3])        # len 3
    
    with pytest.raises(ValueError):
        _ = A @ x