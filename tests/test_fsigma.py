import numpy as np
import pytest
from crtools import fsigma


def test_fsigma_center_exclude():
    arr = np.array([[1., 2., 3.], [4., 100., 6.], [7., 8., 9.]])
    out = np.zeros_like(arr)
    fsigma(arr, out, 1, 1, 1)
    neighbors = np.array([1., 2., 3., 4., 6., 7., 8., 9.])
    expected = np.std(neighbors, ddof=0)
    assert np.allclose(out[1, 1], expected)


def test_fsigma_shape_mismatch():
    arr = np.ones((3, 3), dtype=np.float64)
    out = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        fsigma(arr, out, 1, 1, 1)
