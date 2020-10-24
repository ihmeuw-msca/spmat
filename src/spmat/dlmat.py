"""
Sum of diagonal and low rank matrices
"""
from __future__ import annotations
from typing import Iterable
import numpy as np


class ILMat:
    def __init__(self,
                 lmat: np.ndarray,
                 altmat: ILMat = None):
        self.lmat = lmat
        self.altmat = ILMat(self.lmat.T, altmat=self) if altmat is None else altmat
        self.check_attr()

    @property
    def dsize(self) -> int:
        return self.lmat.shape[0]

    @property
    def lrank(self) -> int:
        return min(self.lmat.shape)

    @property
    def mat(self) -> np.ndarray:
        return np.identity(self.dsize) + self.lmat.dot(self.lmat.T)

    @property
    def invmat(self) -> np.ndarray:
        if self.is_lrank():
            result = np.identity(self.dsize) - self.lmat.dot(
                self.altmat.invmat.dot(self.lmat.T)
            )
        else:
            result = np.linalg.inv(self.mat)
        return result

    def check_attr(self):
        if self.lmat.ndim != 2:
            raise ValueError("`lmat` must be a matrix.")

    def is_lrank(self) -> bool:
        return self.lrank < self.dsize

    def dot(self, array: Iterable) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if array.ndim not in [1, 2]:
            raise ValueError("`array` must be a vector or matrix.")
        result = array + self.lmat.dot(self.lmat.T.dot(array))
        return result

    def invdot(self, array: Iterable) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            array = np.asarray(array)
        if self.is_lrank():
            result = array - self.lmat.dot(
                self.altmat.invdot(self.lmat.T.dot(array))
            )
        else:
            result = np.linalg.solve(self.mat, array)
        return result

    def logdet(self) -> float:
        if self.is_lrank():
            result = self.altmat.logdet()
        else:
            result = np.log(np.linalg.eigvals(self.mat)).sum()
        return result

    def __repr__(self) -> str:
        return f"ILMat(dsize={self.dsize}, lrank={self.lrank})"
