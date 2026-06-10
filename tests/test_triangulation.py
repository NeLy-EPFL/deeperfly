"""Tests for skeleton-aware triangulation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.cameras import CameraGroup
from deeperfly.triangulation import (
    reprojection_error,
    triangulate,
    triangulate_ransac,
)


@pytest.fixture
def cameras(rig) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )


def _fly_cloud(rng, n=38, scale=2.0):
    """A small random 3D cloud near the world origin (within the rig's view)."""
    return rng.normal(scale=scale, size=(n, 3))


def test_roundtrip_recovers_points(cameras, rng):
    pts3d = _fly_cloud(rng)
    pts2d = np.asarray(cameras.project(pts3d))  # (V, N, 2)
    recovered = triangulate(cameras, pts2d)
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)


def test_triangulation_with_nan_observations(cameras, rng):
    # Visibility now travels purely as NaN (no separate mask): drop a couple of
    # (view, point) observations and triangulation still recovers them from the
    # remaining views.
    pts3d = _fly_cloud(rng)
    pts2d = np.array(cameras.project(pts3d))
    pts2d[0, :5] = np.nan  # camera 0 does not see the first five points
    pts2d[3, 7] = np.nan
    recovered = triangulate(cameras, pts2d)
    assert not np.isnan(recovered).any()  # >= 2 views remain for every point
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


# -- RANSAC -----------------------------------------------------------------


def test_ransac_matches_dlt_when_clean(cameras, rng):
    # With no outliers RANSAC recovers the truth and trusts every view.
    pts3d = _fly_cloud(rng)
    pts2d = np.asarray(cameras.project(pts3d))
    recovered, inliers = triangulate_ransac(cameras, pts2d, threshold=1.0)
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)
    assert inliers.shape == (len(cameras), pts3d.shape[0])
    assert inliers.all()


def test_ransac_rejects_gross_outlier(cameras, rng):
    # A single mislocated detection drags plain DLT but not RANSAC.
    pts3d = _fly_cloud(rng, n=6)
    pts2d = np.array(cameras.project(pts3d))
    pts2d[2, 0] += 300.0  # shift one view of point 0 far off

    dlt = triangulate(cameras, pts2d)
    assert np.linalg.norm(dlt[0] - pts3d[0]) > 0.1  # corrupted

    recovered, inliers = triangulate_ransac(cameras, pts2d, threshold=5.0)
    np.testing.assert_allclose(recovered[0], pts3d[0], atol=1e-6)
    assert not inliers[2, 0]  # the bad view was flagged
    assert inliers[:, 0].sum() == len(cameras) - 1
    # Untouched points keep all their views and match the truth.
    assert inliers[:, 1:].all()
    np.testing.assert_allclose(recovered[1:], pts3d[1:], atol=1e-6)


def test_ransac_unobserved_views_are_not_inliers(cameras, rng):
    pts3d = _fly_cloud(rng, n=4)
    pts2d = np.array(cameras.project(pts3d))
    pts2d[0, 0] = np.nan  # camera 0 does not see point 0
    recovered, inliers = triangulate_ransac(cameras, pts2d, threshold=1.0)
    assert not inliers[0, 0]
    assert inliers[1:, 0].all()
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)


def test_ransac_too_few_views_is_nan(cameras, rng):
    pts3d = _fly_cloud(rng, n=4)
    pts2d = np.array(cameras.project(pts3d))
    pts2d[1:, 0] = np.nan  # point 0 seen by a single camera
    recovered, inliers = triangulate_ransac(cameras, pts2d, threshold=1.0)
    assert np.isnan(recovered[0]).all()
    assert not np.isnan(recovered[1:]).any()


def test_ransac_min_inliers_gate(cameras, rng):
    # Point 0 has only two clean views; the rest are gross outliers.
    pts3d = _fly_cloud(rng, n=3)
    pts2d = np.array(cameras.project(pts3d))
    pts2d[2:, 0] += rng.uniform(200, 400, size=(len(cameras) - 2, 2))

    ok, inliers = triangulate_ransac(cameras, pts2d, threshold=5.0, min_inliers=2)
    np.testing.assert_allclose(ok[0], pts3d[0], atol=1e-6)
    assert inliers[:, 0].sum() == 2

    # Demanding three agreeing views rejects point 0 but keeps the clean ones.
    gated, _ = triangulate_ransac(cameras, pts2d, threshold=5.0, min_inliers=3)
    assert np.isnan(gated[0]).all()
    assert not np.isnan(gated[1:]).any()


def test_ransac_sequence_layout(cameras, rng):
    pts3d = rng.normal(scale=2.0, size=(4, 38, 3))
    pts2d = np.asarray(cameras.project(pts3d))
    recovered, inliers = triangulate_ransac(cameras, pts2d, threshold=1.0)
    assert recovered.shape == (4, 38, 3)
    assert inliers.shape == (len(cameras), 4, 38)
    np.testing.assert_allclose(recovered, pts3d, atol=1e-6)


def test_ransac_min_inliers_below_two_raises(cameras, rng):
    pts2d = np.asarray(cameras.project(_fly_cloud(rng, n=2)))
    with pytest.raises(ValueError, match="min_inliers"):
        triangulate_ransac(cameras, pts2d, min_inliers=1)


def test_ransac_tie_break_prefers_tighter_consensus(cameras):
    # Two disjoint, equal-size (3-view) consensus sets compete for one point: a
    # well-separated P (views 0,1,2) and Q (views 3,4,5), with view 6 unobserved.
    # One of Q's views is nudged off, so both sets have the same inlier *count*
    # but P's reprojects tighter -- the tie must break toward P.
    assert len(cameras) >= 6
    P = np.array([0.3, -0.2, 0.5])
    Q = np.array([-0.8, 0.6, -0.4])
    proj = np.asarray(cameras.project(np.stack([P, Q])))  # (V, 2, 2)
    pP, pQ = proj[:, 0], proj[:, 1]

    pts2d = np.full((len(cameras), 1, 2), np.nan)
    pts2d[0, 0], pts2d[1, 0], pts2d[2, 0] = pP[0], pP[1], pP[2]  # exact -> err 0
    pts2d[3, 0], pts2d[4, 0] = pQ[3], pQ[4]
    pts2d[5, 0] = pQ[5] + 2.0  # Q's consensus is 2 px looser than P's

    recovered, inliers = triangulate_ransac(cameras, pts2d, threshold=5.0)
    np.testing.assert_allclose(recovered[0], P, atol=1e-6)
    assert inliers[:3, 0].all() and not inliers[3:, 0].any()
