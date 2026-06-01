"""Tests for skeleton-aware triangulation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.cameras import CameraGroup
from deeperfly.skeleton import Skeleton
from deeperfly.triangulate import (
    apply_visibility,
    merge_points,
    merge_sources,
    reprojection_error,
    triangulate,
)
from helpers import CAMERA_NAMES


@pytest.fixture
def cameras(rig) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


def _fly_cloud(rng, n=38, scale=2.0):
    """A small random 3D cloud near the world origin (within the rig's view)."""
    return rng.normal(scale=scale, size=(n, 3))


def test_roundtrip_recovers_points(cameras, rng):
    pts3d = _fly_cloud(rng)
    pts2d = np.asarray(cameras.project(pts3d))  # (V, N, 2)
    recovered = triangulate(cameras, pts2d)
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)


def test_apply_visibility_nans_invisible(cameras, fly, rng):
    pts3d = _fly_cloud(rng)
    pts2d = np.asarray(cameras.project(pts3d))
    masked = apply_visibility(pts2d, fly, CAMERA_NAMES)
    vis = fly.visibility_mask(CAMERA_NAMES)
    # Exactly the invisible (view, point) entries became NaN.
    assert np.isnan(masked).any(axis=-1).tolist() == (~vis).tolist()
    # Visible entries are untouched.
    np.testing.assert_array_equal(masked[vis], pts2d[vis])


def test_triangulation_after_visibility(cameras, fly, rng):
    pts3d = _fly_cloud(rng)
    pts2d = apply_visibility(np.asarray(cameras.project(pts3d)), fly, CAMERA_NAMES)
    recovered = triangulate(cameras, pts2d)
    # Every fly point is seen by >= 2 cameras, so all are recovered.
    assert not np.isnan(recovered).any()
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)


def test_point_with_one_view_is_nan(cameras, rng):
    pts3d = _fly_cloud(rng, n=4)
    pts2d = np.array(cameras.project(pts3d))
    # Leave point 0 visible in a single camera only.
    pts2d[1:, 0] = np.nan
    recovered = triangulate(cameras, pts2d)
    assert np.isnan(recovered[0]).all()
    assert not np.isnan(recovered[1:]).any()


def test_reprojection_error_zero_at_truth(cameras, rng):
    pts3d = _fly_cloud(rng)
    pts2d = np.asarray(cameras.project(pts3d))
    err = reprojection_error(cameras, pts3d, pts2d)
    assert err.shape == (len(cameras), pts3d.shape[0])
    assert np.nanmax(err) < 1e-6


def test_reprojection_error_nan_where_unobserved(cameras, rng):
    pts3d = _fly_cloud(rng)
    pts2d = np.array(cameras.project(pts3d))
    pts2d[0, 0] = np.nan
    err = reprojection_error(cameras, pts3d, pts2d)
    assert np.isnan(err[0, 0])
    assert not np.isnan(err[1:, 0]).any()


def test_sequence_layout(cameras, rng):
    # (V, T, N, 2) observations triangulate to (T, N, 3).
    pts3d = rng.normal(scale=2.0, size=(5, 38, 3))
    pts2d = np.asarray(cameras.project(pts3d))
    assert pts2d.shape == (len(cameras), 5, 38, 2)
    recovered = triangulate(cameras, pts2d)
    assert recovered.shape == (5, 38, 3)
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)


# -- stripe merge ------------------------------------------------------------


def test_merge_points_routes_each_view_to_its_visible_side(fly, rng):
    merged, remap = fly.merge_lr_stripes()
    vis = fly.visibility_mask(CAMERA_NAMES)
    src = merge_sources(remap, vis, merged.n_points)
    assert src.shape == (7, 35)

    pts2d = apply_visibility(rng.normal(size=(7, 3, 38, 2)), fly, CAMERA_NAMES)
    out = merge_points(pts2d, src, axis=2)
    assert out.shape == (7, 3, 35, 2)
    rh, lh, f = (CAMERA_NAMES.index(n) for n in ("rh", "lh", "f"))
    # Stripe0 (merged idx 16): right cameras keep the right column (old 16),
    # left cameras pull from the left column (old 35), the front sees neither.
    np.testing.assert_array_equal(out[rh, :, 16], pts2d[rh, :, 16])
    np.testing.assert_array_equal(out[lh, :, 16], pts2d[lh, :, 35])
    assert np.isnan(out[f, :, 16]).all()
    # A non-stripe column passes through unchanged.
    np.testing.assert_array_equal(out[:, :, 5], pts2d[:, :, 5])


def test_merge_points_handles_candidate_axes(fly, rng):
    merged, remap = fly.merge_lr_stripes()
    src = merge_sources(remap, fly.visibility_mask(CAMERA_NAMES), merged.n_points)
    score = rng.uniform(size=(7, 2, 38, 4))  # (V, T, N, K)
    out = merge_points(score, src, axis=2)
    assert out.shape == (7, 2, 35, 4)
    lh = CAMERA_NAMES.index("lh")
    np.testing.assert_array_equal(out[lh, :, 16], score[lh, :, 35])


def test_merged_stripe_triangulates_from_four_cameras(cameras, fly, rng):
    pts3d = _fly_cloud(rng)
    pts3d[35:38] = pts3d[16:19]  # left/right stripes are the same physical markers
    pts2d = apply_visibility(np.asarray(cameras.project(pts3d)), fly, CAMERA_NAMES)

    merged, remap = fly.merge_lr_stripes()
    src = merge_sources(remap, fly.visibility_mask(CAMERA_NAMES), merged.n_points)
    merged2d = merge_points(pts2d, src, axis=1)  # single frame: (V, N, 2)
    recovered = triangulate(cameras, merged2d)
    # The merged stripes recover the shared 3D point from all four side cameras.
    np.testing.assert_allclose(recovered[16:19], pts3d[16:19], atol=1e-6)
    assert np.isfinite(recovered).all()
