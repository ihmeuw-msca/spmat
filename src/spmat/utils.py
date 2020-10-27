"""
Utility functions
"""
from typing import Iterable, Tuple, List
import numpy as np


def toarray(array: Iterable,
            ndim: Tuple[int] = None) -> np.ndarray:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if ndim is not None:
        if array.ndim not in ndim:
            raise ValueError(f"`array` ndim must be in {ndim}.")

    return array


def splitarray(array: Iterable,
               sizes: Iterable[int],
               axis: int = 0) -> List[np.ndarray]:
    array = toarray(array)
    if array.shape[axis] != sum(sizes):
        raise ValueError("`array` not match `sizes`.")
    return np.split(array, np.cumsum(sizes)[:-1], axis=axis)
