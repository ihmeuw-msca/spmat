from typing import Callable
import time

import numpy as np
import pytest
from scipy.linalg import block_diag

from spmat.dlmat import BDLMat


def timer(func: Callable):
    """Quick and dirty wrapper to log execution time of a given function.

    Nice to have a decorator implementation since it's easy to wrap existing functions with it.
    """

    def inner_func(*args, **kwargs):
        start = time.time()
        val = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {end - start:.2f}s to complete")
        return val

    return inner_func


class MockBDLMat(BDLMat):
    """Mocks the BDLMat class to wrap the 3 methods of interest in a timer method."""

    @timer
    def dot(self, x):
        return super().dot(x)

    @timer
    def invdot(self, x):
        return super().invdot(x)

    @timer
    def logdet(self):
        return super().logdet()


@pytest.fixture(scope='module')
def large_matrix():
    # Arbitrary decision: Create 1_000 2D arrays, each dimension ranges from 1 to 1000
    # Expected output: will get a block diagonal matrix with average dimensions of
    # (500_000, 500_000), along with appropriate dimensions
    arrays = []
    for _ in range(200_000):
        size = np.random.randint(1, 5)
        arrays.append(np.random.randn(size, size))

    return block_diag(arrays)


@pytest.fixture(scope='module')
def vector(large_matrix):
    """Given a large matrix from above, calculate a vector of usable size."""
    return np.random.randn(large_matrix.shape[0])


@pytest.mark.performance_tests
def test_bdlmat_dot(large_matrix, vector):
    # Dot a super large sparse array
    mat = MockBDLMat(large_matrix)
    mat.dot(vector)
    mat.invdot(vector)
    mat.logdet()


@pytest.mark.performance_tests
def test_numpy_implementation(large_matrix, vector):

    large_matrix @ vector
    np.linalg.solve(large_matrix, vector)
    np.linalg.slogdet(large_matrix)


def test_scipy_sparse(large_matrix):
    # Look into this
    scipy.sparse.linalg.svds(large_matrix)


