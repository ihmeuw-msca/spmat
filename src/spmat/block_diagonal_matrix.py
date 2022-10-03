import numpy as np
import scipy


class Block:

    def __init__(self, data: np.array, validate: bool = False):
        self.data = data
        if validate:
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

    def inv_dot(self, other: np.array):
        return self.inverse.dot(other)

    def validate(self, rtol: float = 1e-05, atol: float = 1e-08) -> None:
        """Blocks are always assumed to be square and symmetric.

        Is this validation necessary? For many blocks the one-off validation
        might not be worth it."""
        if not np.allclose(self.data, self.data.T, rtol=rtol, atol=atol):
            raise ValueError(f"Provided data of {self.data} is not symmetric")

    def __repr__(self):
        return self.data.__repr__()


class BDMatrix:
    """Block diagonal matrix.

    If this class (or any class) has a matrix attribute, and implements the dot, inv_dot,
    and log_determinant methods, then we can say it implements the SpecialMatrix protocol."""

    def __init__(self, data: np.array, block_sizes: list[int]):
        self.data = data
        self.block_sizes = block_sizes
        # Validate that shapes match
        self.validate()

    @property
    def shape(self) -> tuple[int]:
        """Shape of the input data."""
        return self.data.shape

    @property
    def blocks(self) -> list[Block]:
        """Lazy loads a list of block classes."""
        if not hasattr(self, '_blocks'):
            self._blocks = []
            curr_idx = 0
            for nvals in self.block_sizes:
                flat_mat = self.data[curr_idx:curr_idx + nvals, curr_idx:curr_idx + nvals]
                self._blocks.append(Block(flat_mat))
                curr_idx += nvals
        return self._blocks

    @property
    def log_determinant(self) -> float:
        """Sum of the logs."""
        if not hasattr(self, '_log_determinant'):
            self._log_determinant = sum([block.log_det for block in self.blocks])
        return self._log_determinant

    def validate(self):
        block_sum = np.sum(self.block_sizes)
        if (block_sum, block_sum) != self.data.shape:
            raise ValueError(f"Input matrix has dimensions {self.data.shape}, "
                             f"but provided blocks sum to shape {block_sum}. "
                             f"The shapes must exactly match.")

    def dot(self, other: np.array) -> np.array:
        """Calculate the dot product of the block diagonal array with another array."""
        # First pass: Assume other is always a 1D numpy array, organized into the same blocks.

        # Mismatched lengths will break the dot product
        if len(other) != self.shape[0]:
            raise ValueError(f"Provided vector has length {len(other)}, but the data has "
                             f"shape {self.shape}. The dimensions must match.")

        # Split the other array by index. np.cumsum gets the indices to split on
        # exclude the last value to avoid returning a single empty array at the end
        split_array = np.split(other, np.cumsum(self.block_sizes[:-1]))
        result = np.concatenate(
            [block.dot(vector) for block, vector in zip(self.blocks, split_array)]
        )
        return result

    def inv_dot(self, other: np.array) -> np.array:
        """Returns the inverse of this matrix dot the input."""
        # Mismatched lengths will break the dot product
        if len(other) != self.shape[0]:
            raise ValueError(f"Provided vector has length {len(other)}, but the data has "
                             f"shape {self.shape}. The dimensions must match.")

        # Split the other array by index. np.cumsum gets the indices to split on
        # exclude the last value to avoid returning a single empty array at the end
        split_array = np.split(other, np.cumsum(self.block_sizes[:-1]))
        result = np.concatenate(
            [block.inv_dot(vector) for block, vector in zip(self.blocks, split_array)]
        )
        return result

    def __repr__(self):
        return self.data.__repr__()
