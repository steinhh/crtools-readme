import numpy as np
import pytest
from crtools import fmedian, fsigma


def _get_neighbors(arr, iy, ix, xsize, ysize, exclude_center):
    ny, nx = arr.shape
    vals = []
    for dy in range(-ysize, ysize + 1):
        yy = iy + dy
        if yy < 0 or yy >= ny:
            continue
        for dx in range(-xsize, xsize + 1):
            xx = ix + dx
            if xx < 0 or xx >= nx:
                continue
            if exclude_center and dx == 0 and dy == 0:
                continue
            vals.append(arr[yy, xx])
    return np.array(vals, dtype=np.float64)


def reference_fmedian(arr, xsize, ysize, exclude_center):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for iy in range(ny):
        for ix in range(nx):
            vals = _get_neighbors(arr, iy, ix, xsize, ysize, exclude_center)
            if vals.size == 0:
                out[iy, ix] = 0.0
            else:
                out[iy, ix] = np.median(vals)
    return out


def reference_fsigma(arr, xsize, ysize, exclude_center):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    for iy in range(ny):
        for ix in range(nx):
            vals = _get_neighbors(arr, iy, ix, xsize, ysize, exclude_center)
            if vals.size == 0:
                out[iy, ix] = 0.0
            else:
                out[iy, ix] = float(np.std(vals, ddof=0))
    return out


def test_include_center_vs_exclude_center_random():
    rng = np.random.default_rng(12345)
    arr = rng.normal(100, 10, (20, 17)).astype(np.float64)
    for exclude in (0, 1):
        fm_ref = reference_fmedian(arr, 2, 1, exclude)
        sg_ref = reference_fsigma(arr, 2, 1, exclude)
        out_fm = np.zeros_like(arr)
        out_sg = np.zeros_like(arr)
        fmedian(arr, out_fm, 2, 1, exclude)
        fsigma(arr, out_sg, 2, 1, exclude)
        assert np.allclose(out_fm, fm_ref)
        assert np.allclose(out_sg, sg_ref)


def test_corner_and_edge_behavior():
    arr = np.arange(1., 10.).reshape(3, 3)
    # small window, exclude center
    fm_ref = reference_fmedian(arr, 1, 1, 1)
    sg_ref = reference_fsigma(arr, 1, 1, 1)
    out_fm = np.zeros_like(arr)
    out_sg = np.zeros_like(arr)
    fmedian(arr, out_fm, 1, 1, 1)
    fsigma(arr, out_sg, 1, 1, 1)
    assert np.allclose(out_fm, fm_ref)
    assert np.allclose(out_sg, sg_ref)


def test_1d_input_raises():
    arr = np.array([1., 2., 3.])
    out = np.zeros_like(arr)
    with pytest.raises(ValueError):
        fmedian(arr, out, 1, 1, 1)
    with pytest.raises(ValueError):
        fsigma(arr, out, 1, 1, 1)


def test_1x1_exclude_center_returns_zero():
    arr = np.array([[42.0]])
    out = np.zeros_like(arr)
    fmedian(arr, out, 1, 1, 1)
    assert np.isclose(out[0, 0], 0.0)
    out[:] = 0.0
    fsigma(arr, out, 1, 1, 1)
    assert np.isclose(out[0, 0], 0.0)


def test_input_not_modified():
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(8, 8)).astype(np.float64)
    arr_copy = arr.copy()
    out = np.zeros_like(arr)
    fmedian(arr, out, 1, 1, 0)
    fsigma(arr, out, 1, 1, 0)
    assert np.allclose(arr, arr_copy)
