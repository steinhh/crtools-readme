import numpy as np
import pytest

from crtools import fsigma


def test_sigma_zero_on_constant_window():
    """Sigma of a constant-valued neighborhood should be 0.0."""
    a = np.full((5, 5), 7.0, dtype=np.float64)
    out = np.empty_like(a)

    # 3x3 window including center
    fsigma(a, out, 1, 1, 0)
    assert np.allclose(out, 0.0)

    # 3x3 window excluding center
    fsigma(a, out, 1, 1, 1)
    assert np.allclose(out, 0.0)


def test_center_exclusion_reduces_sigma_with_outlier():
    """Excluding the center outlier should reduce local sigma at the center pixel."""
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 999.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    out = np.zeros_like(a)

    # With center included
    fsigma(a, out, 1, 1, 0)
    sigma_with = out[1, 1]

    # With center excluded
    fsigma(a, out, 1, 1, 1)
    sigma_without = out[1, 1]

    assert sigma_with > sigma_without


def test_nan_values_are_ignored():
    """NaN values should be ignored in sigma calculation (center or neighbors)."""
    a = np.array([
        [1.0, 2.0, 3.0],
        [4.0, np.nan, 6.0],  # center NaN
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    out = np.zeros_like(a)

    # Center included: NaN should be ignored; sigma at center should equal
    # population std of [1,2,3,4,6,7,8,9] which is sqrt(7.5)
    fsigma(a, out, 1, 1, 0)
    assert np.isclose(out[1, 1], np.sqrt(7.5))

    # Make a neighbor NaN (not center) and exclude center; sigma should be finite
    b = np.array([
        [np.nan, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ], dtype=np.float64)
    fsigma(b, out, 1, 1, 1)
    # Same neighborhood as above but without 5.0 and with NaN ignored ->
    # values are [1,2,3,4,6,7,8,9] for center (5 ignored via exclude_center, 1 is NaN)
    assert np.isfinite(out[1, 1])


def test_1x1_excluding_center_yields_zero():
    """With a 1x1 window and center excluded (no neighbors), sigma is defined as 0.0."""
    a = np.array([[42.0]], dtype=np.float64)
    out = np.zeros_like(a)
    fsigma(a, out, 0, 0, 1)
    assert np.isclose(out[0, 0], 0.0)


def test_dtype_enforced_float64():
    """fsigma requires float64 input and output arrays."""
    good = np.ones((2, 2), dtype=np.float64)
    out = np.zeros_like(good)

    # Works with float64
    fsigma(good, out, 1, 1, 1)

    # Fails with wrong input dtype
    with pytest.raises(TypeError):
        bad_in = good.astype(np.float32)
        fsigma(bad_in, out, 1, 1, 1)

    # Fails with wrong output dtype
    with pytest.raises(TypeError):
        bad_out = out.astype(np.float32)
        fsigma(good, bad_out, 1, 1, 1)


def test_dimension_checks():
    """Non-2D arrays or mismatched shapes should raise errors."""
    a = np.ones((2, 3), dtype=np.float64)
    out = np.zeros_like(a)

    # Happy path
    fsigma(a, out, 1, 1, 1)

    # Mismatched shape
    with pytest.raises(ValueError):
        out_bad = np.zeros((3, 3), dtype=np.float64)
        fsigma(a, out_bad, 1, 1, 1)

    # 1D array should fail
    with pytest.raises(ValueError):
        a1 = np.ones(3, dtype=np.float64)
        out1 = np.zeros(3, dtype=np.float64)
        fsigma(a1, out1, 1, 1, 1)
