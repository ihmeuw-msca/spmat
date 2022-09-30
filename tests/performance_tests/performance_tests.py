from abc import abstractmethod
import argparse
from collections import defaultdict
import json
import logging
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
VARIABLE_SIZE = [1000, 2000, 5000, 10000, 15000]  # Values to test scaling over
NUM_ITERATIONS = 10


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                              "%Y-%m-%d %H:%M:%S")
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)



class Timer:

    """A simple decorator class that can be used to record elapsed runtimes for various
    matrix functions. A matrix with a shape must be the first positional argument."""

    # Method cache intended to be reused across timer instances, so used as a class variable.
    method_cache: defaultdict = defaultdict(list)

    def __init__(self, implementation: str):
        self.implementation = implementation

    def __call__(self, func: Callable):
        """Wrapper around a call to record elapsed runtime.
        Must provide the number of blocks and block size as the first two arguments,
        for caching purposes."""

        def wrapper(num_blocks: int, block_size: int, *args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            # Store the elapsed runtime in a cache on the class
            # Log to notify of progress
            self.method_cache['algorithm'].append(func.__name__)
            self.method_cache['implementation'].append(self.implementation)
            self.method_cache['num_blocks'].append(num_blocks)
            self.method_cache['block_size'].append(block_size)
            self.method_cache['runtime'].append(end - start)
            return res
        return wrapper

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


class Test:

    @abstractmethod
    def dot():
        NotImplemented

    @abstractmethod
    def inv_dot():
        NotImplemented

    @abstractmethod
    def log_det():
        NotImplemented

    @classmethod
    def run_benchmarks(cls, num_blocks: int, block_size: int, mat1, mat2) -> None:
        # Run multiple times so we can take averages
        for i in range(NUM_ITERATIONS):
            # Give percentages every 10 percent
            chunk = NUM_ITERATIONS // 10
            if i % chunk == 0 and i > 0:
                logger.info(f"Profiling for (num_blocks, block_size) {(num_blocks, block_size)}: "
                            f"{(i / NUM_ITERATIONS) * 100}% done")
            cls.dot(num_blocks, block_size, mat1, mat2)
            cls.inv_dot(num_blocks, block_size, mat1, mat2)
            cls.log_det(num_blocks, block_size, mat1)


class ScipySparseTest:
    # TODO: add this scipy sparse benchmark
    pass


class NumpyTest(Test):

    @staticmethod
    @Timer(implementation='numpy')
    def dot(mat1: np.array, mat2: np.array):
        return mat1.dot(mat2)

    @staticmethod
    @Timer(implementation='numpy')
    def inv_dot(mat1: np.array, mat2: np.array):
        return np.linalg.solve(mat1, mat2)

    @staticmethod
    @Timer(implementation='numpy')
    def log_det(mat1: np.array):
        return np.linalg.slogdet(mat1)


class BDMatrixTest(Test):

    @staticmethod
    @Timer(implementation='raw_python')
    def dot(bdmat: BDMatrix, mat2: np.array):
        return bdmat.dot(mat2)

    @staticmethod
    @Timer(implementation='raw_python')
    def inv_dot(bdmat: BDMatrix, mat2: np.array):
        return bdmat.inv_dot(mat2)

    @staticmethod
    @Timer(implementation='raw_python')
    def log_det(bdmat: BDMatrix):
        return bdmat.log_determinant


def perf_test():

    for num in VARIABLE_SIZE:

        for num_blocks, block_size in [(num, FIXED_SIZE), (FIXED_SIZE, num)]:

            mat = create_diagonal_matrix(num_blocks, block_size)
            # Test dot and invdot methods on a nx2 matrix
            other_mat = np.random.randn(num * FIXED_SIZE, 2)

            # Compute and store numpy results
            NumpyTest.run_benchmarks(num_blocks, block_size, mat, other_mat)

            # Do the same for BDMatrix results
            # Note that results might potentially be confounded slightly. Both invdot and log_det
            # involve use of the matrix's singular values, which are lazily calculated. Therefore
            # whichever method is profiled first will incur the extra runtime of calculating the
            # SVD, and the second method will likely be much faster.
            bdmat = BDMatrix(mat, [block_size] * num_blocks)
            BDMatrixTest.run_benchmarks(num_blocks, block_size, bdmat, other_mat)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--filepath", default=None)
    args = argparser.parse_args()
    filepath = args.filepath
    perf_test()  # Runs all the tests
    # Serialize here, plot in a different step
    if filepath:
        Timer.to_json(filepath)
