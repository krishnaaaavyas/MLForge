from typing import List, Union


Number = Union[int, float]


class Vector:

    def __init__(self, data: List[Number]):

        if not isinstance(data, list) or not data:
            raise ValueError(
                "Vector must be initialized with a non-empty list."
            )

        self.data = [float(x) for x in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"Vector({self.data})"

    def dot(self, other: "Vector") -> float:

        if len(self) != len(other):
            raise ValueError(
                f"Vector dimension mismatch for dot product: "
                f"{len(self)} vs {len(other)}"
            )

        return sum(
            a * b
            for a, b in zip(self.data, other.data)
        )

    def __add__(self, other: "Vector"):

        if len(self) != len(other):
            raise ValueError(
                "Vectors must have the same length for addition."
            )

        return Vector([
            a + b
            for a, b in zip(self.data, other.data)
        ])

    def __sub__(self, other: "Vector"):

        if len(self) != len(other):
            raise ValueError(
                "Vectors must have the same length for subtraction."
            )

        return Vector([
            a - b
            for a, b in zip(self.data, other.data)
        ])

    def __mul__(self, scalar: Number):

        if not isinstance(scalar, (int, float)):
            raise TypeError(
                "Vector can only be multiplied by a scalar."
            )

        return Vector([
            value * scalar
            for value in self.data
        ])

    def __rmul__(self, scalar: Number):

        return self.__mul__(scalar)