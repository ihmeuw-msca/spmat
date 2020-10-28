"""
Test utility functions
"""
import pytest
import numpy as np
from spmat import utils


@pytest.mark.parametrize("array", [np.arange(3),
                                   [0, 1, 2],
                                   (0, 1, 2)])
def test_to_numpy(array):
    nparray = utils.to_numpy(array)
    assert isinstance(nparray, np.ndarray)


@pytest.mark.parametrize("array", [[[1, 2], [3, 4]]])
def test_to_numpy_ndim(array):
    with pytest.raises(ValueError):
        utils.to_numpy(array, ndim=(1,))


@pytest.mark.parametrize("array", [np.ones(5),
                                   np.ones((5, 3))])
@pytest.mark.parametrize("sizes", [[1, 1, 3], [2, 2, 1]])
def test_split(array, sizes):
    arrays = utils.split(array, sizes)
    assert len(arrays) == len(sizes)
    assert all(len(arrays[i]) == size for i, size in enumerate(sizes))


@pytest.mark.parametrize("array", [np.ones((5, 4))])
@pytest.mark.parametrize(("sizes", "axis"),
                         [([2, 2, 1], 0),
                          ([2, 2], 1)])
def test_split_axis(array, sizes, axis):
    arrays = utils.split(array, sizes, axis=axis)
    assert len(arrays) == len(sizes)
    assert all(arrays[i].shape[axis] == size for i, size in enumerate(sizes))


@pytest.mark.parametrize("mats", [[np.ones((3, 3))]*4])
def test_create_bdiag_mat(mats):
    lmat = utils.create_bdiag_mat([np.ones((3, 1))]*4)
    my_bdiag_mat = utils.create_bdiag_mat(mats)
    tr_bdiag_mat = lmat.dot(lmat.T)
    assert np.allclose(my_bdiag_mat, tr_bdiag_mat)
