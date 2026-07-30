# mlforge/math/vector.py
from typing import List, Union

class Vector:
    def __init__(self, data: List[Union[int, float]]):
        if not isinstance(data, list) or not data:
            raise ValueError("Vector must be initialized with a non-empty list of numbers.")
        self.data = [float(x) for x in data]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> float:
        return self.data[index]

    def __repr__(self) -> str:
        return f"Vector({self.data})"

    def dot(self, other: "Vector") -> float:
        """Computes the dot product between two vectors: w . x"""
        if len(self) != len(other):
            raise ValueError(
                f"Vector dimension mismatch for dot product: {len(self)} vs {len(other)}"
            )
        return sum(a * b for a, b in zip(self.data, other.data))

    def __add__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("Vectors must be of identical length for addition.")
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other: "Vector") -> "Vector":
        if len(self) != len(other):
            raise ValueError("Vectors must be of identical length for subtraction.")
        return Vector([a - b for a, b in zip(self.data, other.data)])