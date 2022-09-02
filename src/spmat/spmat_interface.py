from typing import List, Protocol, Tuple
import numpy as np


class SpecialMatrix(Protocol):
    """Protocol for different matrix implementations (ILMat, BDLMat, DLMat, etc.)

    Useful to have since we can define what methods and attributes are expected in each
    implementation.

    Protocols are the more flexible way of doing interfaces, indicate what's expected in each
    subclass without enforcing them at runtime. Abstract Base Classes (ABCs) are stricter,
    your code will fail if self.dot isn't implemented, for example
    """
    matrix: np.array

    def dot(self, x: np.array) -> np.array:
        """Dot product of diagonals"""
        raise NotImplementedError

    def inv_dot(self, x: np.array) -> np.array:
        """Inverse dot product"""
        raise NotImplementedError

    def log_determinant(self) -> float:
        """Log determinant"""
        raise NotImplementedError


