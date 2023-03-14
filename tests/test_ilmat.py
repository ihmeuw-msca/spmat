"""
Test ILMat
"""
import numpy as np
import pytest

from spmat import ILMat

# pylint: disable=redefined-outer-name


SHAPE = (5, 3)


@pytest.fixture
def ilmat():
    lmat = np.random.randn(*SHAPE)
    return ILMat(lmat)


def test_ilmat(ilmat):
    my_result = ilmat.mat.dot(ilmat.invmat)
    tr_result = np.identity(ilmat.dsize)
    assert np.allclose(my_result, tr_result)


@pytest.mark.parametrize("array", [np.random.randn(SHAPE[0]), np.random.randn(*SHAPE)])
def test_dot(ilmat, array):
    my_result = ilmat.dot(array)
    tr_result = ilmat.mat.dot(array)
    assert np.allclose(my_result, tr_result)


@pytest.mark.parametrize("array", [np.random.randn(SHAPE[0]), np.random.randn(*SHAPE)])
def test_invdot(ilmat, array):
    my_result = ilmat.invdot(array)
    tr_result = np.linalg.solve(ilmat.mat, array)
    assert np.allclose(my_result, tr_result)


def test_logdet(ilmat):
    my_result = ilmat.logdet()
    tr_result = np.linalg.slogdet(ilmat.mat)[1]
    assert np.isclose(my_result, tr_result)


def test_diag(ilmat):
    my_result = ilmat.diag()
    tr_result = np.diag(ilmat.mat)
    assert np.allclose(my_result, tr_result)


def test_invdiag(ilmat):
    my_result = ilmat.invdiag()
    tr_result = np.diag(np.linalg.inv(ilmat.mat))
    assert np.allclose(my_result, tr_result)
