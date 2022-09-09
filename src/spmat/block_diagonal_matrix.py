from typing import List
import numpy as np
import scipy

from spmat.utils import flatten, compute_svd


class Block:

    def __init__(self, data: np.array):
        self.data = data
        self.validate()

    @property
    def shape(self):
        return self.data.shape

    @property
    def svd(self):
        if not hasattr(self, '_svd'):
            u, s, _ = scipy.linalg.svd(self.data)
            self._svd = u, s
        return self._svd

    @property
    def inverse(self):
        if not hasattr(self, '_inverse'):
            u, s = self.svd
            self._inverse = u.dot(
                np.diag(1.0 / s)
            ).dot(
                u.T
            )
        return self._inverse

    @property
    def log_det(self) -> float:
        if not hasattr(self, '_log_det'):
            _, s = self.svd
            self._log_det = np.sum(np.log(s))
        return self._log_det

    def dot(self, other: np.array):
        return self.data.dot(other)

    def validate(self, rtol: float = 1e-05, atol: float = 1e-08) -> None:
        """Blocks are always assumed to be square and symmetric."""
        if not np.allclose(self.data, self.data.T, rtol=rtol, atol=atol):
            raise ValueError(f"Provided data of {self.data} is not symmetric")


class BDMatrix:
    """Block diagonal matrix.

    If this class (or any class) has a matrix attribute, and implements the dot, inv_dot,
    and log_determinant methods, then we can say it implements the SpecialMatrix protocol."""

    def __init__(self, data: np.array, block_sizes: List[int]):
        self.data = data
        # Flatten the matrix into a 1-D vector according to block sizes
        # Why flatten? Store as array of blocks?
        # self.flat_matrix = flatten(data, block_sizes)
        self.block_sizes = block_sizes

    @property
    def blocks(self) -> List[Block]:
        """Lazy loads a list of block classes."""
        if not hasattr(self, '_blocks'):
            self._blocks = flatten(self.data, self.block_sizes)
        return self._blocks

    def dot(self, other: np.array) -> np.array:
        # Flatten the other array and do the dot in loops
        # First pass: Assume other is always a 1D numpy array, organized into the same blocks.
        result = np.array([])
        curr_idx = 0
        for block in self.blocks:
            # Can we always assume "other" is a 1 or 2D array?
            rows, _ = block.shape
            dot_subset = block.dot(other[curr_idx:curr_idx + rows])
            curr_idx += rows
            result = np.vstack(result, dot_subset)
        return result

    def inv_dot(self, other: np.array) -> np.array:
        """Returns the inverse of this matrix dot the input."""
        curr_idx = 0
        result = np.array([])
        for block in self.blocks:
            rows, _ = block.shape
            invdot_subset = block.inverse.dot(other[curr_idx:curr_idx + rows])
            result = np.vstack([result, invdot_subset])
            curr_idx += rows
        return result

    @property
    def log_determinant(self) -> float:
        """Sum of the logs."""
        if not hasattr(self, '_log_determinant'):
            self._log_determinant = np.sum([block.log_det for block in self.blocks])
        return self._log_determinant
