import numpy as np
import pytest

from crtools import fmedian


def test_median_excludes_nan_neighbors():
    """Neighbors that are NaN should be ignored when computing the median."""
    arr = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],
        [7.0, 8.0, 9.0]
    ], dtype=np.float64)

    out = np.zeros_like(arr)
    # 3x3 window excluding the center from neighbors
    fmedian(arr, out, 1, 1, 1)

    # For the center pixel, neighbors excluding center are [1,2,3,4,6,7,8,9]
    # median = (4 + 6) / 2 = 5.0
    # Use allclose to avoid fragile exact-equality on floating point results.
    np.testing.assert_allclose(out[1, 1], 5.0, rtol=0, atol=1e-12)


def test_median_with_all_nan_window_writes_nan():
    """If the whole neighborhood (and center) are NaN, output should be NaN."""
    arr = np.array([[np.nan]], dtype=np.float64)
    out = np.zeros_like(arr)

    # Window 1x1, exclude center -> no neighbors; center is NaN -> output should be NaN
    fmedian(arr, out, 0, 0, 1)
    assert np.isnan(out[0, 0])


def test_include_center_nan_is_ignored():
    """When center is NaN but neighbors are finite, including center should still ignore the NaN."""
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],  # center NaN
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    out = np.zeros_like(a)

    # 3x3 window including center
    fmedian(a, out, 1, 1, 0)
    # Values considered: [1,2,3,4,6,7,8,9] (center NaN ignored) -> median = (4+6)/2 = 5
    np.testing.assert_allclose(out[1, 1], 5.0, rtol=0, atol=1e-12)


def test_1x1_excluding_center_uses_center_when_finite():
    """With a 1x1 window and center excluded, no neighbors means we fall back to the center value if finite."""
    a = np.array([[42.0]], dtype=np.float64)
    out = np.zeros_like(a)
    fmedian(a, out, 0, 0, 1)
    np.testing.assert_allclose(out[0, 0], 42.0, rtol=0, atol=1e-12)


def test_corner_partial_window_even_count():
    """At image borders, the window is truncated; check median with an even number of samples."""
    a = np.array([[1.0, 2.0],
                  [3.0, 4.0]], dtype=np.float64)
    out = np.zeros_like(a)

    # 3x3 window including center at top-left corner -> values: [1,2,3,4]
    fmedian(a, out, 1, 1, 0)
    np.testing.assert_allclose(out[0, 0], 2.5, rtol=0, atol=1e-12)


def test_include_center_changes_result_with_outlier():
    """Including the center outlier changes the median compared to excluding it."""
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 999.0, 6.0],  # outlier
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    out = np.zeros_like(a)

    # Excluding center -> neighbors are [1,2,3,4,6,7,8,9] -> median = 5
    fmedian(a, out, 1, 1, 1)
    med_excl = out[1, 1]

    # Including center -> values [1,2,3,4,6,7,8,9,999] -> median = 6
    fmedian(a, out, 1, 1, 0)
    med_incl = out[1, 1]

    np.testing.assert_allclose(med_excl, 5.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(med_incl, 6.0, rtol=0, atol=1e-12)


def test_single_valid_neighbor_median():
    """If only a single finite neighbor remains, the median equals that neighbor."""
    a = np.array([
        [10.0, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
    ], dtype=np.float64)
    out = np.zeros_like(a)

    # Exclude center, compute at (1,1); only (0,0) is valid within the 3x3 window
    fmedian(a, out, 1, 1, 1)
    np.testing.assert_allclose(out[1, 1], 10.0, rtol=0, atol=1e-12)


def test_dtype_enforced_float64_fmedian():
    """fmedian requires float64 input and output arrays."""
    good = np.ones((2, 2), dtype=np.float64)
    out = np.zeros_like(good)

    # Works with float64
    fmedian(good, out, 1, 1, 1)

    # Fails with wrong input dtype
    with pytest.raises(TypeError):
        bad_in = good.astype(np.float32)
        fmedian(bad_in, out, 1, 1, 1)

    # Fails with wrong output dtype
    with pytest.raises(TypeError):
        bad_out = out.astype(np.float32)
        fmedian(good, bad_out, 1, 1, 1)


def test_dimension_checks_fmedian():
    """Non-2D arrays or mismatched shapes should raise errors for fmedian."""
    a = np.ones((2, 3), dtype=np.float64)
    out = np.zeros_like(a)

    # Happy path
    fmedian(a, out, 1, 1, 1)

    # Mismatched shape
    with pytest.raises(ValueError):
        out_bad = np.zeros((3, 3), dtype=np.float64)
        fmedian(a, out_bad, 1, 1, 1)

    # 1D array should fail
    with pytest.raises(ValueError):
        a1 = np.ones(3, dtype=np.float64)
        out1 = np.zeros(3, dtype=np.float64)
        fmedian(a1, out1, 1, 1, 1)
