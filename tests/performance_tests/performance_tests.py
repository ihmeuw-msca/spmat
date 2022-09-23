import argparse
from collections import defaultdict
import json
import numpy as np
import scipy
import time
from typing import Callable

from spmat.block_diagonal_matrix import BDMatrix

# Note that this module is not named test_... to avoid being run by pytest

# 2 axes to test over: block size and number of blocks
# Test scaling up both parameters. Meaningful tests are to
# 1. Large number of small blocks
# 2. Small number of large blocks

FIXED_SIZE = 4  # Dimension of the parameter to be fixed
VARIABLE_SIZE = [1000, 2000, 5000, 10000, 20000, 50000]  # Values to test scaling over


class Timer:

    """A simple decorator class that can be used to record elapsed runtimes for various
    matrix functions. A matrix with a shape must be the first positional argument."""

    # Method cache intended to be reused across timer instances, so used as a class variable.
    method_cache: defaultdict = defaultdict(dict)

    def __init__(self, func: Callable):
        self.func = func

    def __call__(self, num_blocks, block_size, *args, **kwargs):
        """Wrapper around a call to record elapsed runtime.
        Must provide the number of blocks and block size as the first two arguments,
        for caching purposes."""
        start = time.time()
        res = self.func(*args, **kwargs)
        end = time.time()
        # Store the elapsed runtime in a cache on the class
        # Note that this caching expects uniqueness by function name
        self.method_cache[self.func.__name__][(num_blocks, block_size)] = end - start
        return res

    @classmethod
    def to_json(cls, outfile: str):
        # Serializes the mapping of runtimes to a json file
        with open(outfile, 'w') as f:
            json.dump(cls.method_cache, f)


def create_symmetric_matrix(size: int):
    mat = np.random.randn(size, size)
    # Multiply by the transpose to make symmetric
    mat = mat.T.dot(mat)
    return mat


def create_diagonal_matrix(num_blocks, block_size):

    blocks = [create_symmetric_matrix(block_size) for _ in range(num_blocks)]
    return scipy.linalg.block_diag(*blocks)


class NumpyTest:

    @staticmethod
    @Timer
    def numpy_dot(mat1: np.array, mat2: np.array):
        return mat1.dot(mat2)

    @staticmethod
    @Timer
    def numpy_inv_dot(mat1: np.array, mat2: np.array):
        return np.linalg.inv(mat1).dot(mat2)

    @staticmethod
    @Timer
    def numpy_log_det(mat1: np.array):
        return np.linalg.slogdet(mat1)


class BDMatrixTest:

    @staticmethod
    @Timer
    def bdmat_dot(bdmat: BDMatrix, mat2: np.array):
        return bdmat.dot(mat2)

    @staticmethod
    @Timer
    def bdmat_inv_dot(bdmat: BDMatrix, mat2: np.array):
        return bdmat.inv_dot(mat2)

    @staticmethod
    @Timer
    def bdmat_log_det(bdmat: BDMatrix):
        return bdmat.log_determinant


def perf_test():

    for num in VARIABLE_SIZE:

        for num_blocks, block_size in [(num, FIXED_SIZE), (FIXED_SIZE, num)]:

            mat = create_diagonal_matrix(num_blocks, block_size)
            # Test dot and invdot methods on a nx2 matrix
            other_mat = np.random.randn(num * FIXED_SIZE, 2)

            # Compute and store numpy results
            NumpyTest.numpy_dot(num_blocks, block_size, mat, other_mat)
            NumpyTest.numpy_inv_dot(num_blocks, block_size, mat, other_mat)
            NumpyTest.numpy_log_det(num_blocks, block_size, mat)

            # Do the same for BDMatrix results
            bdmat = BDMatrix(mat, [block_size] * num_blocks)
            BDMatrixTest.bdmat_dot(num_blocks, block_size, bdmat, other_mat)
            # Note that results might potentially be confounded slightly. Both invdot and log_det
            # involve use of the matrix's singular values, which are lazily calculated. Therefore
            # whichever method is profiled first will incur the extra runtime of calculating the
            # SVD, and the second method will likely be much faster.
            BDMatrixTest.bdmat_inv_dot(num_blocks, block_size, bdmat, other_mat)
            BDMatrixTest.bdmat_log_det(num_blocks, block_size, bdmat)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filepath")
    args = argparser.parse_args()
    filepath = args.filepath
    perf_test()  # Runs all the tests
    # Serialize here, plot in a different step
    Timer.to_json(filepath)
