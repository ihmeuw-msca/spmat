"""
Test DLMat
"""
import numpy as np
import pytest

from spmat import DLMat

# pylint: disable=redefined-outer-name


SHAPE = (5, 3)


@pytest.fixture
def dlmat():
    diag = np.random.rand(SHAPE[0]) + 0.1
    lmat = np.random.randn(*SHAPE)
    return DLMat(diag, lmat)


def test_dlmat(dlmat):
    my_result = dlmat.mat.dot(dlmat.invmat)
    tr_result = np.identity(dlmat.dsize)
    assert np.allclose(my_result, tr_result)


@pytest.mark.parametrize("array", [np.random.randn(SHAPE[0]), np.random.randn(*SHAPE)])
def test_dot(dlmat, array):
    my_result = dlmat.dot(array)
    tr_result = dlmat.mat.dot(array)
    assert np.allclose(my_result, tr_result)


@pytest.mark.parametrize("array", [np.random.randn(SHAPE[0]), np.random.randn(*SHAPE)])
def test_invdot(dlmat, array):
    my_result = dlmat.invdot(array)
    tr_result = np.linalg.solve(dlmat.mat, array)
    assert np.allclose(my_result, tr_result)


def test_logdet(dlmat):
    my_result = dlmat.logdet()
    tr_result = np.linalg.slogdet(dlmat.mat)[1]
    assert np.isclose(my_result, tr_result)


def test_diag(dlmat):
    my_result = dlmat.diag()
    tr_result = np.diag(dlmat.mat)
    assert np.allclose(my_result, tr_result)


def test_invdiag(dlmat):
    my_result = dlmat.invdiag()
    tr_result = np.diag(np.linalg.inv(dlmat.mat))
    assert np.allclose(my_result, tr_result)
