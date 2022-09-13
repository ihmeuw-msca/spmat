import numpy as np
import pytest
from spmat.block_diagonal_matrix import Block, BDMatrix


def create_block(shape) -> Block:
    block = np.random.randn(*shape)
    # Make symmetric
    block = block.T.dot(block)
    return Block(block)


def test_block():
    block = create_block((3, 3))
    assert block.shape == (3, 3)

    # Check the log det is computable, lazy loading works properly
    assert np.isclose(block.log_det, np.linalg.slogdet(block.data)[1])
    assert hasattr(block, '_svd')

    other_vec = np.array([1, 1, 1])

    assert np.allclose(block.dot(other_vec), block.data.dot(other_vec))

    assert np.allclose(block.inverse, np.linalg.inv(block.data))

def test_block_matrix():

    pass