"""
Utility functions
"""
from typing import Iterable, Tuple, List
import numpy as np
from scipy.linalg import svd


def to_numpy(array: Iterable,
             ndim: Tuple[int] = None) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if ndim is not None:
        if array.ndim not in ndim:
            raise ValueError(f"`array` ndim must be in {ndim}.")

    return array


def flatten(matrix: np.array, block_sizes: List[int]) -> np.array:
    """Flattens a block diagonal matrix into a 1-d vector.

    Assumes every block is square"""

    # Validate that shapes match
    block_sum = np.sum(np.array(block_sizes), axis=0)
    if (block_sum, block_sum) != matrix.shape:
        raise ValueError(f"Input matrix has dimensions {matrix.shape}, "
                         f"but provided blocks sum to shape {[block_sum] * 2}. "
                         f"The shapes must exactly match.")

    result = np.array([])
    curr_idx = 0
    for nvals in block_sizes:
        flat_mat = matrix[curr_idx:curr_idx + nvals, curr_idx:curr_idx + nvals].flatten()
        result = np.hstack((result, flat_mat))
        curr_idx += nvals
    return result


def split(array: Iterable,
          sizes: Iterable[int],
          axis: int = 0) -> List[np.ndarray]:
    array = to_numpy(array)
    if array.shape[axis] != sum(sizes):
        raise ValueError("`array` not match `sizes`.")
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)


def create_bdiag_mat(mats: List[np.ndarray]) -> np.ndarray:
    if not all([mat.ndim == 2 for mat in mats]):
        raise ValueError("`mats` must be a list of matrices.")
    row_sizes = [mat.shape[0] for mat in mats]
    col_sizes = [mat.shape[1] for mat in mats]
    row_size = sum(row_sizes)
    col_size = sum(col_sizes)

    bdiag_mat = np.zeros((row_size, col_size), dtype=mats[0].dtype)
    row_idx = split(np.arange(row_size), row_sizes)
    col_idx = split(np.arange(col_size), col_sizes)

    for i, mat in enumerate(mats):
        bdiag_mat[np.ix_(row_idx[i], col_idx[i])] = mat

    return bdiag_mat


def compute_svd(matrix: np.array, block_sizes: List[int]) -> np.array:
    """Compute the SVD components by block.

    matrix: 1D array, unrolled block diagonal matrix
    block_sizes: a list of ints detailing block sizes, to help us unroll"""

    total_blocks = sum(block_sizes)

    if matrix.size != total_blocks:
        raise ValueError(f"Provided array is length {matrix.size} but provided blocks are "
                         f"total {total_blocks}. The values must match.")

    curr_start = 0
    u_vector = np.array([])
    s_vector = np.array([])
    for block in block_sizes:
        # Square since block size implies block^2 values when unrolled
        curr_end = curr_start + block ** 2
        block_mat = matrix[curr_start:curr_end].reshape(block, block)
        u, s, _ = svd(block_mat)

        u_vector = np.hstack(u_vector, u.flatten())
        s_vector = np.hstack(s_vector, np.diag(s))
        curr_start = curr_end

    return u_vector, s_vector
