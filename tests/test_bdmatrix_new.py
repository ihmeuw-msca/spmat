import numpy as np
import pytest
import scipy
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

    # Create a series of blocks
    blocks = [create_block((i, i)).data for i in range(1, 4)]
    # Expected size: 3 blocks of 1x1, 2x2, 3x3 dimensions. End matrix is going to be 6x6
    diagonal_mat = scipy.linalg.block_diag(*blocks)

    with pytest.raises(ValueError):
        # Create a matrix with incorrect dimensions.
        BDMatrix(data=diagonal_mat, block_sizes=[1])

    # with correct dimensions:
    bd_matrix = BDMatrix(data=diagonal_mat, block_sizes=[1, 2, 3])
    assert bd_matrix.shape == (6, 6)

    other_mat = np.random.randn(6)

    # Calculate the dot
    # Should exactly equal the full dot product, without computing the extra 0's
    assert np.allclose(bd_matrix.dot(other_mat), bd_matrix.data.dot(other_mat))

    # Inverse dot
    assert np.allclose(bd_matrix.inv_dot(other_mat),
                       np.linalg.inv(bd_matrix.data).dot(other_mat))

    # Test with a 2d array too
    other_mat_2d = np.random.randn(6, 2)
    assert np.allclose(bd_matrix.dot(other_mat_2d), bd_matrix.data.dot(other_mat_2d))

    # Inverse dot
    assert np.allclose(bd_matrix.inv_dot(other_mat_2d),
                       np.linalg.inv(bd_matrix.data).dot(other_mat_2d))

    # Assert the SVD has been calculated dynamically and cached
    assert all([hasattr(block, '_svd') for block in bd_matrix.blocks])

    # Assert the log determinant exists
    assert np.isclose(bd_matrix.log_determinant, np.linalg.slogdet(bd_matrix.data)[1])
