from typing import List, Tuple, Union

from mlforge.math.vector import Vector


Number = Union[int, float]


class Matrix:

    def __init__(self, data: List[List[Number]]):

        if (
            not isinstance(data, list)
            or not data
            or not isinstance(data[0], list)
            or not data[0]
        ):
            raise ValueError(
                "Matrix must be initialized with a non-empty 2D list."
            )

        self.rows = len(data)
        self.cols = len(data[0])

        for row in data:
            if not isinstance(row, list):
                raise ValueError(
                    "Each Matrix row must be a list."
                )

            if len(row) != self.cols:
                raise ValueError(
                    "All rows in a Matrix must have the same length."
                )

        self.data = [
            [float(value) for value in row]
            for row in data
        ]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.rows, self.cols

    def __getitem__(self, index: int):
        return self.data[index]

    def __repr__(self):
        return f"Matrix(shape={self.shape})"

    def get_row(self, row_idx: int) -> Vector:

        return Vector(self.data[row_idx])

    def get_col(self, col_idx: int) -> Vector:

        return Vector([
            self.data[row][col_idx]
            for row in range(self.rows)
        ])

    def __add__(self, other: "Matrix"):

        if not isinstance(other, Matrix):
            raise TypeError(
                "Matrix can only be added to another Matrix."
            )

        if self.shape != other.shape:
            raise ValueError(
                f"Matrix shape mismatch for addition: "
                f"{self.shape} vs {other.shape}"
            )

        result = [
            [
                self.data[r][c] + other.data[r][c]
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

        return Matrix(result)

    def __sub__(self, other: "Matrix"):

        if not isinstance(other, Matrix):
            raise TypeError(
                "Matrix can only be subtracted from another Matrix."
            )

        if self.shape != other.shape:
            raise ValueError(
                f"Matrix shape mismatch for subtraction: "
                f"{self.shape} vs {other.shape}"
            )

        result = [
            [
                self.data[r][c] - other.data[r][c]
                for c in range(self.cols)
            ]
            for r in range(self.rows)
        ]

        return Matrix(result)

    def __mul__(self, scalar: Number):

        if not isinstance(scalar, (int, float)):
            raise TypeError(
                "Matrix can only be multiplied by a scalar."
            )

        result = [
            [
                value * scalar
                for value in row
            ]
            for row in self.data
        ]

        return Matrix(result)

    def __rmul__(self, scalar: Number):

        return self.__mul__(scalar)

    def matmul(self, other):

        if isinstance(other, Vector):

            if self.cols != len(other):
                raise ValueError(
                    f"Shape mismatch for Matrix-Vector multiplication: "
                    f"{self.shape} vs Vector of length {len(other)}"
                )

            result = [
                self.get_row(row).dot(other)
                for row in range(self.rows)
            ]

            return Vector(result)

        if isinstance(other, Matrix):

            if self.cols != other.rows:
                raise ValueError(
                    f"Shape mismatch for Matrix-Matrix multiplication: "
                    f"{self.shape} vs {other.shape}"
                )

            result = []

            for r in range(self.rows):

                row_result = []

                for c in range(other.cols):

                    value = self.get_row(r).dot(
                        other.get_col(c)
                    )

                    row_result.append(value)

                result.append(row_result)

            return Matrix(result)

        raise TypeError(
            f"Unsupported operand type for matmul: {type(other)}"
        )

    def __matmul__(self, other):

        return self.matmul(other)