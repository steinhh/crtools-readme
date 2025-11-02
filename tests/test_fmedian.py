import numpy as np
import pytest
from crtools import fmedian


def test_fmedian_center_exclude():
    arr = np.array([[1., 2., 3.], [4., 100., 6.], [7., 8., 9.]])
    out = np.zeros_like(arr)
    # exclude center: median of all neighbors except center
    fmedian(arr, out, 1, 1, 1)
    neighbors = np.array([1., 2., 3., 4., 6., 7., 8., 9.])
    expected = np.median(neighbors)
    assert out[1, 1] == expected


def test_fmedian_shape_mismatch():
    arr = np.ones((3, 3), dtype=np.float64)
    out = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        fmedian(arr, out, 1, 1, 1)
