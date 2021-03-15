"""
Sum of diagonal and low rank matrices
"""
from typing import List, Iterable
import numpy as np
from numpy import ndarray
from scipy.linalg import block_diag
from spmat import utils, linalg


class ILMat:
    """
    Identity plus outer product of low rank matrix, I + L @ L.T

    Attributes
    ----------
    lmat : ndarray
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
        self._w = -self._v/(1 + self._v)

    @property
    def mat(self) -> ndarray:
        return np.identity(self.dsize) + (self._u*self._v) @ self._u.T

    @property
    def invmat(self) -> ndarray:
        return np.identity(self.dsize) + (self._u*self._w) @ self._u.T

    def dot(self, x: Iterable) -> ndarray:
        """
        Dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        ndarray
        """
        x = utils.to_numpy(x, ndim=(1, 2))
        return x + (self._u*self._v) @ (self._u.T @ x)

    def invdot(self, x: Iterable) -> ndarray:
        """
        Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        ndarray
        """
        x = utils.to_numpy(x, ndim=(1, 2))
        return x + (self._u*self._w) @ (self._u.T @ x)

    def logdet(self) -> float:
        """
        Log determinant

        Returns
        -------
        float
            Log determinant of the matrix.
        """
        return np.log(1 + self._v).sum()

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

        self._u, s = linalg.block_lsvd(self.lmats.copy(),
                                       self.dsizes,
                                       self.lranks)
        self._v = s**2
        self._w = -self._v/(1 + self._v)

    @property
    def lmat_blocks(self) -> List[ndarray]:
        return np.split(self.lmats, np.cumsum(self.dsizes)[:-1], axis=0)

    @property
    def mat(self) -> ndarray:
        return block_diag(*[
            np.identity(self.dsizes[i]) + lmat.dot(lmat.T)
            for i, lmat in enumerate(self.lmat_blocks)
        ])

    @property
    def invmat(self) -> ndarray:
        return block_diag(*[
            np.linalg.inv(np.identity(self.dsizes[i]) + lmat.dot(lmat.T))
            for i, lmat in enumerate(self.lmat_blocks)
        ])

    def dot(self, x: Iterable) -> ndarray:
        x = np.ascontiguousarray(x)
        dotfun = linalg.block_mvdot if x.ndim == 1 else linalg.block_mmdot
        return dotfun(self._u, self._v, x, self.dsizes, self.lranks)

    def invdot(self, x: Iterable) -> ndarray:
        x = np.ascontiguousarray(x)
        dotfun = linalg.block_mvdot if x.ndim == 1 else linalg.block_mmdot
        return dotfun(self._u, self._w, x, self.dsizes, self.lranks)

    def logdet(self) -> float:
        return np.log(1 + self._v).sum()

    def __repr__(self) -> str:
        return f"BILMat(dsize={self.dsize}, num_blocks={self.dsizes.size})"


class DLMat:
    """
    Diagonal plus outer product of low rank matrix, D + L @ L.T

    Attributes
    ----------
    diag : ndarray
        Diagonal vector.
    lmat : ndarray
        Low rank matrix.
    dsize : int
        Size of the diagonal.
    lrank : int
        Rank of the low rank matrix.
    sdiag : ndarray
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

    def __init__(self, diag: Iterable, lmat: Iterable):
        """
        Parameters
        ----------
        diag : Iterable
            Diagonal vector.
        lmat : Iterable
            Low rank matrix.

        Raises
        ------
        ValueError
            If length of ``diag`` not match with number of rows of ``lmat``.
        ValueError
            If there are non-positive numbers in ``diag``.
        """
        diag = utils.to_numpy(diag, ndim=(1,))
        lmat = utils.to_numpy(lmat, ndim=(2,))
        if diag.size != lmat.shape[0]:
            raise ValueError("`diag` and `lmat` size not match.")
        if any(diag <= 0.0):
            raise ValueError("`diag` must be all positive.")

        self.diag = diag
        self.lmat = lmat

        self.dsize = self.diag.size
        self.lrank = min(self.lmat.shape)

        self.sdiag = np.sqrt(self.diag)
        self.ilmat = ILMat(self.lmat/self.sdiag[:, np.newaxis])

    @property
    def mat(self) -> ndarray:
        return np.diag(self.diag) + self.lmat.dot(self.lmat.T)

    @property
    def invmat(self) -> ndarray:
        return self.ilmat.invmat/(self.sdiag[:, np.newaxis] * self.sdiag)

    def dot(self, x: Iterable) -> ndarray:
        """
        Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        ndarray
        """
        x = utils.to_numpy(x, ndim=(1, 2))
        x = (x.T*self.sdiag).T
        x = self.ilmat.dot(x)
        x = (x.T*self.sdiag).T
        return x

    def invdot(self, x: Iterable) -> ndarray:
        """
        Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        ndarray
        """
        x = utils.to_numpy(x, ndim=(1, 2))
        x = (x.T/self.sdiag).T
        x = self.ilmat.invdot(x)
        x = (x.T/self.sdiag).T
        return x

    def logdet(self) -> float:
        """
        Log determinant

        Returns
        -------
        float
            Log determinant of the matrix.
        """
        return np.log(self.diag).sum() + self.ilmat.logdet()

    def __repr__(self) -> str:
        return f"DLMat(dsize={self.dsize}, lrank={self.lrank})"


class BDLMat:
    """
    Block diagonal low rank matrix, [... D + L @ L.T ...]

    Attributes
    ----------
    diags : ndarray
        Diagonal component of the matrix.
    lmats : ndarray
        L-Matrix component of the matrix.
    dsizes : ndarray
        An array contains ``dsize`` for each block.
    dsize : int
        Overall diagonal size of the matrix.
    lranks : int
        Ranks of each l-matrix block.
    sdiags : ndarray
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

    def __init__(self, diags: Iterable, lmats: Iterable, dsizes: Iterable):
        self.diags = np.ascontiguousarray(diags)
        self.lmats = np.ascontiguousarray(lmats)
        self.dsizes = np.ascontiguousarray(dsizes)
        self.lranks = np.minimum(self.dsizes, self.lmats.shape[1])
        self.sdiags = np.sqrt(self.diags)

        self.bilmat = BILMat(self.lmats/self.sdiags[:, np.newaxis], self.dsizes)
        self.dsize = self.dsizes.sum()

    @property
    def mat(self) -> ndarray:
        return self.bilmat.mat * (self.sdiags[:, np.newaxis] * self.sdiags)

    @property
    def invmat(self) -> ndarray:
        return self.bilmat.invmat / (self.sdiags[:, np.newaxis] * self.sdiags)

    @property
    def num_blocks(self) -> int:
        return self.dsizes.size

    def dot(self, x: ndarray) -> ndarray:
        """
        Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        ndarray
        """
        x = np.ascontiguousarray(x)
        x = (x.T*self.sdiags).T
        x = self.bilmat.dot(x)
        x = (x.T*self.sdiags).T
        return x

    def invdot(self, x: ndarray) -> ndarray:
        """
        Inverse dot product with vector or matrix

        Parameters
        ----------
        x : Iterable
            Vector or matrix

        Returns
        -------
        ndarray
        """
        x = np.ascontiguousarray(x)
        x = (x.T/self.sdiags).T
        x = self.bilmat.invdot(x)
        x = (x.T/self.sdiags).T
        return x

    def logdet(self) -> float:
        """
        Log determinant

        Returns
        -------
        float
            Log determinant of the matrix.
        """
        return np.log(self.diags).sum() + self.bilmat.logdet()

    def __repr__(self) -> str:
        return f"BDLMat(dsize={self.dsize}, num_blocks={self.num_blocks})"
