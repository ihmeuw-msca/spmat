"""
Test utility functions
"""
import pytest
import numpy as np
from spmat import utils


@pytest.mark.parametrize("array", [np.arange(3),
                                   [0, 1, 2],
                                   (0, 1, 2)])
def test_toarray(array):
    nparray = utils.toarray(array)
    assert isinstance(nparray, np.ndarray)


@pytest.mark.parametrize("array", [[[1, 2], [3, 4]]])
def test_toarray_ndim(array):
    with pytest.raises(ValueError):
        utils.toarray(array, ndim=(1,))


@pytest.mark.parametrize("array", [np.ones(5),
                                   np.ones((5, 3))])
@pytest.mark.parametrize("sizes", [[1, 1, 3], [2, 2, 1]])
def test_splitarray(array, sizes):
    arrays = utils.splitarray(array, sizes)
    assert len(arrays) == len(sizes)
    assert all(len(arrays[i]) == size for i, size in enumerate(sizes))


@pytest.mark.parametrize("array", [np.ones((5, 4))])
@pytest.mark.parametrize(("sizes", "axis"),
                         [([2, 2, 1], 0),
                          ([2, 2], 1)])
def test_splitarray_axis(array, sizes, axis):
    arrays = utils.splitarray(array, sizes, axis=axis)
    assert len(arrays) == len(sizes)
    assert all(arrays[i].shape[axis] == size for i, size in enumerate(sizes))
