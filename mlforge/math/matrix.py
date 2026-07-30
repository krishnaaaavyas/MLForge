# mlforge/math/matrix.py
from typing import List, Tuple, Union
from mlforge.math.vector import Vector

class Matrix:
    def __init__(self, data: List[List[Union[int, float]]]):
        if not data or not isinstance(data, list) or not isinstance(data[0], list):
            raise ValueError("Matrix must be initialized with a 2D list of numbers.")
        
        self.rows = len(data)
        self.cols = len(data[0])
        
        # Ensure all rows have equal column length
        for row in data:
            if len(row) != self.cols:
                raise ValueError("All rows in a Matrix must have the same length.")
        
        self.data = [[float(val) for val in row] for row in data]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, index: int) -> List[float]:
        return self.data[index]

    def __repr__(self) -> str:
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix([\n  {rows_str}\n], shape={self.shape})"

    def get_row(self, row_idx: int) -> Vector:
        """Returns the specified row as a Vector object."""
        return Vector(self.data[row_idx])

    def get_col(self, col_idx: int) -> Vector:
        """Returns the specified column as a Vector object."""
        return Vector([self.data[r][col_idx] for r in range(self.rows)])

    def matmul(self, other: Union["Matrix", Vector]) -> Union["Matrix", Vector]:
        """
        Handles:
        1. Matrix x Matrix: (M x N) * (N x P) -> (M x P)
        2. Matrix x Vector: (M x N) * (N x 1) -> Vector of length M
        """
        if isinstance(other, Vector):
            if self.cols != len(other):
                raise ValueError(
                    f"Shape mismatch for Matrix-Vector multiplication: {self.shape} vs Vector of length {len(other)}"
                )
            # Dot product of each row with the vector
            result = [self.get_row(r).dot(other) for r in range(self.rows)]
            return Vector(result)

        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError(
                    f"Shape mismatch for Matrix-Matrix multiplication: {self.shape} vs {other.shape}"
                )
            
            result_data = []
            for r in range(self.rows):
                row_vec = self.get_row(r)
                row_result = []
                for c in range(other.cols):
                    col_vec = other.get_col(c)
                    row_result.append(row_vec.dot(col_vec))
                result_data.append(row_result)
            
            return Matrix(result_data)

        else:
            raise TypeError(f"Unsupported operand type for matmul: {type(other)}")

    def __matmul__(self, other: Union["Matrix", Vector]) -> Union["Matrix", Vector]:
        """Allows using the @ operator: C = A @ B"""
        return self.matmul(other)