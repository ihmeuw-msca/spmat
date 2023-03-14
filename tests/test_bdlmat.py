"""
Test BDLMat
"""
import numpy as np
import pytest

from spmat import BDLMat

# pylint: disable=redefined-outer-name


SHAPE = (5, 3)
NUM_BLOCKS = 3


@pytest.fixture
def bdlmat():
    diag = np.random.rand(SHAPE[0] * NUM_BLOCKS) + 0.1
    lmat = np.random.randn(SHAPE[0] * NUM_BLOCKS, SHAPE[1])
    dsizes = [SHAPE[0]] * NUM_BLOCKS
    return BDLMat(diag, lmat, dsizes)


def test_bdlmat(bdlmat):
    my_result = bdlmat.mat.dot(bdlmat.invmat)
    tr_result = np.identity(bdlmat.dsize)
    assert np.allclose(my_result, tr_result)


@pytest.mark.parametrize(
    "array",
    [
        np.random.randn(SHAPE[0] * NUM_BLOCKS),
        np.random.randn(SHAPE[0] * NUM_BLOCKS, SHAPE[1]),
    ],
)
def test_dot(bdlmat, array):
    my_result = bdlmat.dot(array)
    tr_result = bdlmat.mat.dot(array)
    assert np.allclose(my_result, tr_result)


@pytest.mark.parametrize(
    "array",
    [
        np.random.randn(SHAPE[0] * NUM_BLOCKS),
        np.random.randn(SHAPE[0] * NUM_BLOCKS, SHAPE[1]),
    ],
)
def test_invdot(bdlmat, array):
    my_result = bdlmat.invdot(array)
    tr_result = np.linalg.solve(bdlmat.mat, array)
    assert np.allclose(my_result, tr_result)


def test_logdet(bdlmat):
    my_result = bdlmat.logdet()
    tr_result = np.linalg.slogdet(bdlmat.mat)[1]
    assert np.isclose(my_result, tr_result)


def test_diag(bdlmat):
    my_result = bdlmat.diag()
    tr_result = np.diag(bdlmat.mat)
    assert np.allclose(my_result, tr_result)


def test_invdiag(bdlmat):
    my_result = bdlmat.invdiag()
    tr_result = np.diag(np.linalg.inv(bdlmat.mat))
    assert np.allclose(my_result, tr_result)
