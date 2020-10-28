"""
Utility functions
"""
from typing import Iterable, Tuple, List
import numpy as np


def to_numpy(array: Iterable,
             ndim: Tuple[int] = None) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if ndim is not None:
        if array.ndim not in ndim:
            raise ValueError(f"`array` ndim must be in {ndim}.")

    return array


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
