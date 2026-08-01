from typing import Iterator, List, Union

Number = Union[int, float]


class Vector:
    """
    A one-dimensional mathematical vector.

    Example:
        [1, 2, 3]
    """

    def __init__(self, data: List[Number]):
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Vector must be initialized with a non-empty list.")

        self.data = [float(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[float]:
        return iter(self.data)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __repr__(self) -> str:
        return f"Vector({self.data})"

    @property
    def shape(self):
        return (len(self.data),)

    def copy(self):
        return Vector(self.data.copy())

    def dot(self, other: "Vector") -> float:
        if len(self) != len(other):
            raise ValueError(
                f"Vector dimension mismatch: {len(self)} vs {len(other)}"
            )

        return sum(a * b for a, b in zip(self, other))

    def __add__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("Vectors must have equal length.")

        return Vector(
            [a + b for a, b in zip(self, other)]
        )

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("Vectors must have equal length.")

        return Vector(
            [a - b for a, b in zip(self, other)]
        )

    def __mul__(self, scalar: Number) -> "Vector":
        if not isinstance(scalar, (int, float)):
            raise TypeError("Vector can only be multiplied by a scalar.")

        return Vector(
            [x * scalar for x in self]
        )

    def __rmul__(self, scalar: Number):
        return self * scalar

    def __truediv__(self, scalar: Number):
        if scalar == 0:
            raise ZeroDivisionError("Division by zero.")

        return Vector(
            [x / scalar for x in self]
        )