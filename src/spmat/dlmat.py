"""
Sum of diagonal and low rank matrices
"""
from typing import Iterable, List

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import block_diag

from spmat import linalg, utils


class ILMat:
    """
    Identity plus outer product of low rank matrix, I + L @ L.T

    Attributes
    ----------
    lmat : NDArray
        Low rank matrix.
    tmat : ILMat
        Alternative ILMat, construct by transpose of ``lmat``. It is not
        ``None`` when ``lmat`` is a 'fat' matrix.
    dsize : int
        Size of the diagonal.
    lrank : int
        Rank of the low rank matrix.

    Methods
    -------
    dot(x)
        Dot product with vector or matrix.
    invdot(x)
        Inverse dot product with vector or matrix.
    logdet()
        Log determinant of the matrix.

    """

    def __init__(self, lmat: Iterable):
        """
        Parameters
        ----------
        lmat : Iterable

        Raises
        ------
        ValueError
            When ``lmat`` is not a matrix.

        """
        self.lmat = utils.to_numpy(lmat, ndim=(2,))
        self.dsize = self.lmat.shape[0]
        self.lrank = min(self.lmat.shape)

        self._u, s, _ = np.linalg.svd(self.lmat, full_matrices=False)
        self._v = s**2
        self._w = -self._v / (1 + self._v)

    @property
    def mat(self) -> NDArray:
        return np.identity(self.dsize) + (self._u * self._v) @ self._u.T

    @property
    def invmat(self) -> NDArray:
        return np.identity(self.dsize) + (self._u * self._w) @ self._u.T

    def dot(self, x: Iterable) -> NDArray:
        """Dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        NDArray

        """
        x = utils.to_numpy(x, ndim=(1, 2))
        return x + (self._u * self._v) @ (self._u.T @ x)

    def invdot(self, x: Iterable) -> NDArray:
        """Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        NDArray
        """
        x = utils.to_numpy(x, ndim=(1, 2))
        return x + (self._u * self._w) @ (self._u.T @ x)

    def logdet(self) -> float:
        """Log determinant

        Returns
        -------
        float

        """
        return np.log(1 + self._v).sum()

    def diag(self) -> NDArray:
        """Diagonal of the matrix

        Returns
        -------
        NDArray

        """
        return 1.0 + (self.lmat**2).sum(axis=1)

    def invdiag(self) -> NDArray:
        """Diagonal of the inverse of the matrix

        Returns
        -------
        NDArray

        """
        return 1.0 + (self._u**2 * self._w).sum(axis=1)

    def __repr__(self) -> str:
        return f"ILMat(dsize={self.dsize}, lrank={self.lrank})"


class BILMat:
    """
    Block ILMat.
    """

    def __init__(self, lmats: Iterable, dsizes: Iterable):
        self.lmats = np.ascontiguousarray(lmats)
        self.dsizes = np.asarray(dsizes).astype(int)
        self.lranks = np.minimum(self.dsizes, self.lmats.shape[1])
        self.dsize = self.dsizes.sum()

        if self.dsizes.sum() != self.lmats.shape[0]:
            raise ValueError("Sizes of blocks do not match shape of matrix.")

        self._u, s = linalg.block_lsvd(self.lmats.copy(), self.dsizes, self.lranks)
        self._v = s**2
        self._w = -self._v / (1 + self._v)

    @property
    def lmat_blocks(self) -> List[NDArray]:
        return np.split(self.lmats, np.cumsum(self.dsizes)[:-1], axis=0)

    @property
    def mat(self) -> NDArray:
        return block_diag(
            *[
                np.identity(self.dsizes[i]) + lmat.dot(lmat.T)
                for i, lmat in enumerate(self.lmat_blocks)
            ]
        )

    @property
    def invmat(self) -> NDArray:
        return block_diag(
            *[
                np.linalg.inv(np.identity(self.dsizes[i]) + lmat.dot(lmat.T))
                for i, lmat in enumerate(self.lmat_blocks)
            ]
        )

    def dot(self, x: Iterable) -> NDArray:
        x = np.ascontiguousarray(x)
        dotfun = linalg.block_mvdot if x.ndim == 1 else linalg.block_mmdot
        return dotfun(self._u, self._v, x, self.dsizes, self.lranks)

    def invdot(self, x: Iterable) -> NDArray:
        x = np.ascontiguousarray(x)
        dotfun = linalg.block_mvdot if x.ndim == 1 else linalg.block_mmdot
        return dotfun(self._u, self._w, x, self.dsizes, self.lranks)

    def logdet(self) -> float:
        return np.log(1 + self._v).sum()

    def diag(self) -> NDArray:
        return 1.0 + (self.lmats**2).sum(axis=1)

    def invdiag(self) -> NDArray:
        return 1.0 + linalg.block_rowsum(
            self._u**2, self._w, self.dsizes, self.lranks
        )

    def __repr__(self) -> str:
        return f"BILMat(dsize={self.dsize}, num_blocks={self.dsizes.size})"


class DLMat:
    """
    Diagonal plus outer product of low rank matrix, D + L @ L.T

    Attributes
    ----------
    diag : NDArray
        Diagonal vector.
    lmat : NDArray
        Low rank matrix.
    dsize : int
        Size of the diagonal.
    lrank : int
        Rank of the low rank matrix.
    sdiag : NDArray
        Square root of diagonal vector.
    ilmat : ILMat
        Inner ILMat after strip off the diagonal vector.

    Methods
    -------
    dot(x)
        Dot product with vector or matrix.
    invdot(x)
        Inverse dot product with vector or matrix.
    logdet()
        Log determinant of the matrix.
    """

    def __init__(self, dvec: Iterable, lmat: Iterable):
        """
        Parameters
        ----------
        dvec : Iterable
            Diagonal vector.
        lmat : Iterable
            Low rank matrix.

        Raises
        ------
        ValueError
            If length of ``dvec`` not match with number of rows of ``lmat``.
        ValueError
            If there are non-positive numbers in ``diag``.
        """
        dvec = utils.to_numpy(dvec, ndim=(1,))
        lmat = utils.to_numpy(lmat, ndim=(2,))
        if dvec.size != lmat.shape[0]:
            raise ValueError("`diag` and `lmat` size not match.")
        if any(dvec <= 0.0):
            raise ValueError("`diag` must be all positive.")

        self.dvec = dvec
        self.lmat = lmat

        self.dsize = self.dvec.size
        self.lrank = min(self.lmat.shape)

        self.sdvec = np.sqrt(self.dvec)
        self.ilmat = ILMat(self.lmat / self.sdvec[:, np.newaxis])

    @property
    def mat(self) -> NDArray:
        return np.diag(self.dvec) + self.lmat.dot(self.lmat.T)

    @property
    def invmat(self) -> NDArray:
        return self.ilmat.invmat / (self.sdvec[:, np.newaxis] * self.sdvec)

    def dot(self, x: Iterable) -> NDArray:
        """Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        NDArray
        """
        x = utils.to_numpy(x, ndim=(1, 2))
        x = (x.T * self.sdvec).T
        x = self.ilmat.dot(x)
        x = (x.T * self.sdvec).T
        return x

    def invdot(self, x: Iterable) -> NDArray:
        """Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        NDArray

        """
        x = utils.to_numpy(x, ndim=(1, 2))
        x = (x.T / self.sdvec).T
        x = self.ilmat.invdot(x)
        x = (x.T / self.sdvec).T
        return x

    def logdet(self) -> float:
        """Log determinant

        Returns
        -------
        float
            Log determinant of the matrix.

        """
        return np.log(self.dvec).sum() + self.ilmat.logdet()

    def diag(self) -> NDArray:
        """Diagonal of the matrix

        Returns
        -------
        NDArray

        """
        return self.dvec + (self.lmat**2).sum(axis=1)

    def invdiag(self) -> NDArray:
        """Diagonal of the inverse of the matrix

        Returns
        -------
        NDArray

        """
        return self.ilmat.invdiag() / self.dvec

    def __repr__(self) -> str:
        return f"DLMat(dsize={self.dsize}, lrank={self.lrank})"


class BDLMat:
    """
    Block diagonal low rank matrix, [... D + L @ L.T ...]

    Attributes
    ----------
    diags : NDArray
        Diagonal component of the matrix.
    lmats : NDArray
        L-Matrix component of the matrix.
    dsizes : NDArray
        An array contains ``dsize`` for each block.
    dsize : int
        Overall diagonal size of the matrix.
    lranks : int
        Ranks of each l-matrix block.
    sdiags : NDArray
        Square root of the diagonal.
    bilmat : BILMat
        Block ILMat for easier computation

    Methods
    -------
    dot(x)
        Dot product with vector or matrix.
    invdot(x)
        Inverse dot product with vector or matrix.
    logdet()
        Log determinant of the matrix.
    """

    def __init__(self, dvecs: Iterable, lmats: Iterable, dsizes: Iterable):
        self.dvecs = np.ascontiguousarray(dvecs)
        self.lmats = np.ascontiguousarray(lmats)
        self.dsizes = np.ascontiguousarray(dsizes)
        self.lranks = np.minimum(self.dsizes, self.lmats.shape[1])
        self.sdvecs = np.sqrt(self.dvecs)

        self.bilmat = BILMat(self.lmats / self.sdvecs[:, np.newaxis], self.dsizes)
        self.dsize = self.dsizes.sum()

    @property
    def mat(self) -> NDArray:
        return self.bilmat.mat * (self.sdvecs[:, np.newaxis] * self.sdvecs)

    @property
    def invmat(self) -> NDArray:
        return self.bilmat.invmat / (self.sdvecs[:, np.newaxis] * self.sdvecs)

    @property
    def num_blocks(self) -> int:
        return self.dsizes.size

    def dot(self, x: NDArray) -> NDArray:
        """Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        NDArray

        """
        x = np.ascontiguousarray(x)
        x = (x.T * self.sdvecs).T
        x = self.bilmat.dot(x)
        x = (x.T * self.sdvecs).T
        return x

    def invdot(self, x: NDArray) -> NDArray:
        """Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        NDArray

        """
        x = np.ascontiguousarray(x)
        x = (x.T / self.sdvecs).T
        x = self.bilmat.invdot(x)
        x = (x.T / self.sdvecs).T
        return x

    def logdet(self) -> float:
        """Log determinant

        Returns
        -------
        float

        """
        return np.log(self.dvecs).sum() + self.bilmat.logdet()

    def diag(self) -> NDArray:
        """Diagonal of the matrix

        Returns
        -------
        NDArray

        """
        return self.dvecs + (self.lmats**2).sum(axis=1)

    def invdiag(self) -> NDArray:
        """Diagonal of the inverse of the matrix

        Returns
        -------
        NDArray

        """
        return self.bilmat.invdiag() / self.dvecs

    def __repr__(self) -> str:
        return f"BDLMat(dsize={self.dsize}, num_blocks={self.num_blocks})"
