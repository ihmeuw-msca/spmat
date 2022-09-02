from typing import List
import numpy as np

from spmat.utils import flatten, compute_svd


class BDMatrix:
    """Block diagonal matrix.

    If this class (or any class) has a matrix attribute, and implements the dot, inv_dot,
    and log_determinant methods, then we can say it implements the SpecialMatrix protocol."""

    def __init__(self, data: np.array, block_sizes: List[int]):
        self.matrix = data
        # Flatten the matrix into a 1-D vector according to block sizes
        self.flat_matrix = flatten(data, block_sizes)
        self.block_sizes = block_sizes

    @property
    def block_svd(self) -> np.array:
        """Lazy loads the SVD of each block in the array."""
        # Lazy loading is often a useful paradigm to compute something expensive
        # the first time it's needed, not necessarily on startup.
        if not hasattr(self, '_block_svd'):
            self._block_svd = compute_svd(self.flat_matrix, self.block_sizes)
        return self._block_svd

    def dot(self, other: np.array) -> np.array:
        # Flatten the other array and do the dot in loops
        # First pass: Assume other is always a 1D numpy array, organized into the same blocks.
        result = np.array([])
        curr_start = 0
        for block in self.block_sizes:
            curr_end = curr_start + block
            dot_prod = self.matrix[curr_start:curr_end, curr_start:curr_end].reshape(
                (block, block)
            ).dot(
                other[curr_start:curr_end]
            )
            result = np.hstack(result, dot_prod)
            curr_start = curr_end
        return result

    def inv_dot(self, other: np.array) -> np.array:
        """Returns the inverse of this matrix dot the input."""
        u, s = self.block_svd
        curr_start = 0
        result = np.array([])
        for block in self.block_sizes:
            # invert the s and multiply?
            curr_end = curr_start + block
            block_inverse = np.invert(
                                s[curr_start:curr_end]
                            ).dot(
                                other[curr_start:curr_end]
                            )
            result = np.hstack(result, block_inverse)
            curr_start = curr_end
        return result

    def log_determinant(self) -> np.array:
        # Multiply the singular matrix diagonal by block
        # Return type? Array or a single value?
        _, s = self.block_svd
        result = np.array([])
        curr_start = 0
        for block in self.block_sizes:
            curr_end = curr_start + block
            np.append(result, s[curr_start:curr_end].prod())
            curr_start = curr_end
        return result


