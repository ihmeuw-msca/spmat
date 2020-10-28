"""
Sum of diagonal and low rank matrices
"""
from __future__ import annotations
from typing import List, Iterable
import numpy as np
from . import utils


class ILMat:
    def __init__(self,
                 lmat: np.ndarray):
        self.lmat = lmat
        self.dsize = self.lmat.shape[0]
        self.lrank = min(self.lmat.shape)
        self.check_attr()

        if self.has_altmat():
            self.altmat = ILMat(self.lmat.T)

    @property
    def mat(self) -> np.ndarray:
        return np.identity(self.dsize) + self.lmat.dot(self.lmat.T)

    @property
    def invmat(self) -> np.ndarray:
        if self.has_altmat():
            result = np.identity(self.dsize) - self.lmat.dot(
                self.altmat.invmat.dot(self.lmat.T)
            )
        else:
            result = np.linalg.inv(self.mat)
        return result

    def check_attr(self):
        if self.lmat.ndim != 2:
            raise ValueError("`lmat` must be a matrix.")

    def has_altmat(self) -> bool:
        return self.lrank < self.dsize

    def dot(self, array: Iterable) -> np.ndarray:
        array = utils.to_numpy(array, ndim=(1, 2))
        result = array + self.lmat.dot(self.lmat.T.dot(array))
        return result

    def invdot(self, array: Iterable) -> np.ndarray:
        array = utils.to_numpy(array, ndim=(1, 2))
        if self.has_altmat():
            result = array - self.lmat.dot(
                self.altmat.invdot(self.lmat.T.dot(array))
            )
        else:
            result = np.linalg.solve(self.mat, array)
        return result

    def logdet(self) -> float:
        if self.has_altmat():
            result = self.altmat.logdet()
        else:
            result = np.log(np.linalg.eigvals(self.mat)).sum()
        return result

    def __repr__(self) -> str:
        return f"ILMat(dsize={self.dsize}, lrank={self.lrank})"


class DLMat:
    def __init__(self,
                 diag: np.ndarray,
                 lmat: np.ndarray):
        self.diag = diag
        self.lmat = lmat
        self.dsize = self.diag.size
        self.lrank = min(self.lmat.shape)
        self.check_attr()

        scaled_lmat = self.dscale(self.lmat, by="row", inv=True)
        self.coremat = ILMat(scaled_lmat)

    @property
    def mat(self) -> np.ndarray:
        return np.diag(self.diag) + self.lmat.dot(self.lmat.T)

    @property
    def invmat(self) -> np.ndarray:
        return self.dscale(self.coremat.invmat, inv=True)

    def check_attr(self):
        if self.diag.ndim != 1:
            raise ValueError("`diag` has to be a vector.")
        if self.lmat.ndim != 2:
            raise ValueError("`lmat` has to be a matrix.")
        if self.dsize != self.lmat.shape[0]:
            raise ValueError("`diag` and `lmat` size not match.")
        if any(self.diag <= 0.0):
            raise ValueError("`diag` can only contain positive numbers.")

    def dscale(self,
               array: Iterable,
               by: str = "both",
               inv: bool = False) -> np.ndarray:
        if not isinstance(array, np.ndarray):
            array = np.to_numpy(array)
        if by not in ["row", "col", "both"]:
            raise ValueError("`by` can only be selected from 'row', 'col' and 'both'.")
        d = 1/np.sqrt(self.diag) if inv else np.sqrt(self.diag)

        if array.ndim == 1:
            result = array*d**(2 if by == "both" else 1)
        elif array.ndim == 2:
            if by == "row":
                result = d[:, None]*array
            elif by == "col":
                result = array*d
            else:
                result = d[:, None]*array*d
        else:
            raise ValueError("`array` must be a vector or matrx.")
        return result

    def dot(self, array: Iterable) -> np.ndarray:
        return self.dscale(self.coremat.dot(
            self.dscale(array, by="row")
        ), by="row")

    def invdot(self, array: Iterable) -> np.ndarray:
        return self.dscale(self.coremat.invdot(
            self.dscale(array, by="row", inv=True)
        ), by="row", inv=True)

    def logdet(self) -> float:
        return np.log(self.diag).sum() + self.coremat.logdet()

    def __repr__(self) -> str:
        return f"DLMat(dsize={self.dsize}, lrank={self.lrank})"


class BDLMat:
    def __init__(self,
                 dlmats: List[DLMat]):
        self.dlmats = dlmats
        self.num_blocks = len(self.dlmats)
        self.dsizes = np.array([dlmat.dsize for dlmat in self.dlmats])
        self.dsize = self.dsizes.sum()

    @property
    def mat(self) -> np.ndarray:
        return utils.create_bdiag_mat([dlmat.mat for dlmat in self.dlmats])

    @property
    def invmat(self) -> np.ndarray:
        return utils.create_bdiag_mat([dlmat.invmat for dlmat in self.dlmats])

    def dot(self, array: Iterable) -> np.ndarray:
        array = utils.to_numpy(array, ndim=(1, 2))
        arrays = utils.split(array, self.dsizes)
        return np.concatenate([
            dlmat.dot(arrays[i])
            for i, dlmat in enumerate(self.dlmats)
        ], axis=0)

    def invdot(self, array: Iterable) -> np.ndarray:
        array = utils.to_numpy(array, ndim=(1, 2))
        arrays = utils.split(array, self.dsizes)
        return np.concatenate([
            dlmat.invdot(arrays[i])
            for i, dlmat in enumerate(self.dlmats)
        ], axis=0)

    def logdet(self) -> float:
        return sum([dlmat.logdet() for dlmat in self.dlmats])

    def __repr__(self) -> str:
        return f"BDLMat(dsize={self.dsize}, num_blocks={self.num_blocks})"

    @classmethod
    def create_bdlmat(cls,
                      diag: np.ndarray,
                      lmat: np.ndarray,
                      dsizes: Iterable[int]) -> BDLMat:
        diags = utils.split(diag, dsizes)
        lmats = utils.split(lmat, dsizes)
        return cls([DLMat(*args) for args in zip(diags, lmats)])
