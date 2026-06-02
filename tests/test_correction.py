"""Tests for the 3D error-correction module."""

from __future__ import annotations

import numpy as np

from deeperfly.correction import (
    OneEuroFilter,
    drop_outliers,
    flag_outliers,
    smooth_gaussian,
    smooth_one_euro,
)


# -- outliers ----------------------------------------------------------------


def test_flag_outliers_threshold_and_nan():
    err = np.array([[5.0, 50.0, np.nan], [41.0, 39.0, 100.0]])
    mask = flag_outliers(err, threshold=40.0)
    np.testing.assert_array_equal(mask, [[False, True, False], [True, False, True]])


def test_drop_outliers_sets_nan():
    pts2d = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    mask = np.zeros((2, 3), dtype=bool)
    mask[0, 1] = True
    out = drop_outliers(pts2d, mask)
    assert np.isnan(out[0, 1]).all()
    assert not np.isnan(out[0, 0]).any()
    np.testing.assert_array_equal(out[~mask], pts2d[~mask])


# -- smoothing ---------------------------------------------------------------


def test_one_euro_reduces_jitter(rng):
    signal = 1.0 + rng.normal(scale=0.5, size=(300, 1, 3))
    filtered = smooth_one_euro(signal, fps=100.0, mincutoff=0.3, beta=0.0)
    assert filtered.shape == signal.shape
    assert filtered[50:].var() < 0.3 * signal[50:].var()


def test_one_euro_filter_callable_elementwise():
    filt = OneEuroFilter(100.0, mincutoff=1.0)
    out = filt(np.zeros((2, 3)))
    assert out.shape == (2, 3)


def test_smooth_gaussian_reduces_jitter(rng):
    signal = 2.0 + rng.normal(scale=0.4, size=(200, 2, 3))
    filtered = smooth_gaussian(signal, sigma=3.0)
    assert filtered.shape == signal.shape
    assert filtered.var() < signal.var()


def test_smooth_gaussian_nan_aware():
    seq = np.ones((10, 1, 3))
    seq[5] = np.nan  # isolated gap
    out = smooth_gaussian(seq, sigma=1.0)
    assert np.isfinite(out).all()  # neighbors fill the gap
    np.testing.assert_allclose(out[5], 1.0, atol=1e-9)


def test_smooth_gaussian_all_nan_stays_nan():
    seq = np.full((4, 1, 3), np.nan)
    out = smooth_gaussian(seq, sigma=1.0)
    assert np.isnan(out).all()
