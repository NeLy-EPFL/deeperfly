"""Tests for the 3D error-correction module."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import geometry as geom
from deeperfly.correction import (
    OneEuroFilter,
    align_to_template,
    drop_outliers,
    flag_outliers,
    procrustes_align,
    smooth_gaussian,
    smooth_one_euro,
)
from deeperfly.skeleton import Skeleton


def _rotation(rng):
    return np.asarray(geom.rvec_to_rmat(rng.normal(size=3)))


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


# -- Procrustes --------------------------------------------------------------


def test_procrustes_recovers_known_similarity(rng):
    x = rng.normal(size=(15, 3))
    s_true, r_true, t_true = 1.7, _rotation(rng), rng.normal(size=3)
    y = s_true * (x @ r_true.T) + t_true
    aligned, (s, r, t) = procrustes_align(x, y)
    np.testing.assert_allclose(aligned, y, atol=1e-9)
    assert s == pytest.approx(s_true, rel=1e-6)
    np.testing.assert_allclose(r, r_true, atol=1e-6)
    np.testing.assert_allclose(t, t_true, atol=1e-6)


def test_procrustes_rigid_ignores_scale(rng):
    x = rng.normal(size=(10, 3))
    r_true, t_true = _rotation(rng), rng.normal(size=3)
    y = 3.0 * (x @ r_true.T) + t_true  # scaled target
    _, (s, _, _) = procrustes_align(x, y, scale=False)
    assert s == 1.0


def test_procrustes_ignores_nan_rows(rng):
    x = rng.normal(size=(15, 3))
    r_true, t_true = _rotation(rng), rng.normal(size=3)
    y = x @ r_true.T + t_true
    x_nan = x.copy()
    x_nan[3] = np.nan
    aligned, (_, r, _) = procrustes_align(x_nan, y)
    np.testing.assert_allclose(r, r_true, atol=1e-6)
    assert np.isnan(aligned[3]).all()  # NaN row stays NaN


def test_align_to_template_per_side(rng):
    skel = Skeleton.fly()
    template = rng.normal(size=(skel.n_points, 3))
    pts = template.copy()
    for idx in (skel.left_idx, skel.right_idx):
        r, t = _rotation(rng), rng.normal(size=3)
        pts[idx] = 1.3 * (template[idx] @ r.T) + t
    aligned = align_to_template(pts, template, skel)
    np.testing.assert_allclose(
        aligned[skel.left_idx], template[skel.left_idx], atol=1e-6
    )
    np.testing.assert_allclose(
        aligned[skel.right_idx], template[skel.right_idx], atol=1e-6
    )


def test_align_to_template_sequence(rng):
    skel = Skeleton.fly()
    template = rng.normal(size=(skel.n_points, 3))
    seq = np.stack(
        [template + rng.normal(scale=0.01, size=template.shape) for _ in range(4)]
    )
    aligned = align_to_template(seq, template, skel)
    assert aligned.shape == seq.shape


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
    assert np.isfinite(out).all()  # neighbours fill the gap
    np.testing.assert_allclose(out[5], 1.0, atol=1e-9)


def test_smooth_gaussian_all_nan_stays_nan():
    seq = np.full((4, 1, 3), np.nan)
    out = smooth_gaussian(seq, sigma=1.0)
    assert np.isnan(out).all()
