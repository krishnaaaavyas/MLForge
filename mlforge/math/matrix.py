from typing import Iterator, List, Tuple, Union

from mlforge.math.vector import Vector

Number = Union[int, float]


class Matrix:
    """
    Two-dimensional matrix.
    """

    def __init__(self, data: List[List[Number]]):

        if (
            not isinstance(data, list)
            or len(data) == 0
            or not isinstance(data[0], list)
        ):
            raise ValueError("Matrix must be initialized with a non-empty 2D list.")

        self.rows = len(data)
        self.cols = len(data[0])

        for row in data:
            if len(row) != self.cols:
                raise ValueError(
                    "All rows must contain the same number of columns."
                )

        self.data = [
            [float(x) for x in row]
            for row in data
        ]

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.rows, self.cols)

    def __repr__(self):
        return f"Matrix(shape={self.shape})"

    def __iter__(self) -> Iterator[List[float]]:
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def copy(self):
        return Matrix(
            [row.copy() for row in self.data]
        )

    def get_row(self, index: int) -> Vector:
        return Vector(self.data[index])

    def get_col(self, index: int) -> Vector:
        return Vector(
            [row[index] for row in self.data]
        )

    @property
    def T(self):
        """
        Matrix transpose.
        """
        return Matrix(
            [
                [self.data[r][c] for r in range(self.rows)]
                for c in range(self.cols)
            ]
        )

    def __add__(self, other: "Matrix"):

        if self.shape != other.shape:
            raise ValueError("Matrix dimensions must match.")

        return Matrix(
            [
                [
                    self.data[r][c] + other.data[r][c]
                    for c in range(self.cols)
                ]
                for r in range(self.rows)
            ]
        )

    def matmul(self, other):

        if isinstance(other, Vector):

            if self.cols != len(other):
                raise ValueError(
                    f"Shape mismatch {self.shape} x {other.shape}"
                )

            return Vector(
                [
                    self.get_row(r).dot(other)
                    for r in range(self.rows)
                ]
            )

        elif isinstance(other, Matrix):

            if self.cols != other.rows:
                raise ValueError(
                    f"Shape mismatch {self.shape} x {other.shape}"
                )

            result = []

            for r in range(self.rows):

                row = []

                for c in range(other.cols):

                    row.append(
                        self.get_row(r).dot(
                            other.get_col(c)
                        )
                    )

                result.append(row)

            return Matrix(result)

        else:
            raise TypeError(
                "Unsupported operand for matrix multiplication."
            )

    def __matmul__(self, other):
        return self.matmul(other)

    def __add__(self, other):

        if isinstance(other, Matrix):

            if self.shape != other.shape:
                raise ValueError("Matrix dimensions must match.")

            return Matrix(
                [
                    [
                        self.data[r][c] + other.data[r][c]
                        for c in range(self.cols)
                    ]
                    for r in range(self.rows)
                ]
            )

        elif isinstance(other, Vector):

            if self.cols != len(other):
                raise ValueError(
                    "Vector length must equal number of columns."
                )

            # Broadcast the vector across every row

            return Matrix(
                [
                    [
                        self.data[r][c] + other[c]
                        for c in range(self.cols)
                    ]
                    for r in range(self.rows)
                ]
            )

        raise TypeError(
            "Matrix can only be added to Matrix or Vector."
        )