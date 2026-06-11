"""Tests for :mod:`deeperfly.geometry`.

The computer-vision primitives are cross-checked against OpenCV
(``cv2.Rodrigues``, ``cv2.projectPoints``, ``cv2.triangulatePoints``), which is
the de-facto reference implementation of these models. Where OpenCV and this
library implement the *same* closed form (Rodrigues, the projection +
distortion model), agreement is expected down to floating-point round-off; the
tolerances below are deliberately tight to catch any drift.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deeperfly import geometry as geom


def random_rvecs(rng, n, scale=1.0):
    return rng.normal(size=(n, 3)) * scale


# -- Rodrigues: rvec <-> rmat ------------------------------------------------


def test_rvec_to_rmat_matches_opencv(rng):
    for rvec in random_rvecs(rng, 25, scale=2.0):
        expected, _ = cv2.Rodrigues(rvec)
        got = np.asarray(geom.rvec_to_rmat(rvec))
        assert np.allclose(got, expected, atol=1e-12)


def test_rvec_to_rmat_is_a_rotation(rng):
    rmats = np.asarray(geom.rvec_to_rmat(random_rvecs(rng, 50, scale=3.0)))
    eye = np.broadcast_to(np.eye(3), rmats.shape)
    assert np.allclose(rmats @ rmats.transpose(0, 2, 1), eye, atol=1e-12)
    assert np.allclose(np.linalg.det(rmats), 1.0, atol=1e-12)


def test_rvec_to_rmat_small_angle_stays_orthogonal():
    # Catastrophic cancellation in (1 - cos theta) would show up here.
    rvecs = np.array([[0.0, 0.0, 0.0], [1e-10, 0.0, 0.0], [1e-6, -2e-6, 3e-7]])
    rmats = np.asarray(geom.rvec_to_rmat(rvecs))
    eye = np.broadcast_to(np.eye(3), rmats.shape)
    assert np.allclose(rmats @ rmats.transpose(0, 2, 1), eye, atol=1e-14)


def test_rmat_to_rvec_matches_opencv(rng):
    for rvec in random_rvecs(rng, 25, scale=2.0):
        rmat, _ = cv2.Rodrigues(rvec)
        expected, _ = cv2.Rodrigues(rmat)
        got = np.asarray(geom.rmat_to_rvec(rmat))
        assert np.allclose(got, expected.ravel(), atol=1e-10)


def test_rvec_rmat_roundtrip(rng):
    rvecs = random_rvecs(rng, 50, scale=2.5)
    # Keep angles strictly below pi so the rvec round-trip is unambiguous.
    rvecs = rvecs / np.maximum(1.0, np.linalg.norm(rvecs, axis=-1, keepdims=True) / 3.0)
    rmats = geom.rvec_to_rmat(rvecs)
    assert np.allclose(np.asarray(geom.rmat_to_rvec(rmats)), rvecs, atol=1e-10)


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
def test_rmat_to_rvec_near_pi(axis):
    # At theta = pi the rvec sign is ambiguous, but R(rvec) is well-defined, so
    # round-trip through the rotation matrix and compare matrices.
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    for theta in (np.pi, np.pi - 1e-7):
        rmat = np.asarray(geom.rvec_to_rmat(axis * theta))
        rvec = np.asarray(geom.rmat_to_rvec(rmat))
        assert np.allclose(np.asarray(geom.rvec_to_rmat(rvec)), rmat, atol=1e-6)


def test_rotation_batched_shapes(rng):
    rvecs = random_rvecs(rng, 6).reshape(2, 3, 3)
    rmats = geom.rvec_to_rmat(rvecs)
    assert rmats.shape == (2, 3, 3, 3)
    assert geom.rmat_to_rvec(rmats).shape == (2, 3, 3)


# -- intrinsics --------------------------------------------------------------


def test_intr_to_kmat_four_vector():
    k = np.asarray(geom.intr_to_kmat(np.array([800.0, 810.0, 320.0, 240.0])))
    expected = np.array([[800.0, 0, 320.0], [0, 810.0, 240.0], [0, 0, 1.0]])
    assert np.allclose(k, expected)


def test_intr_to_kmat_three_vector_shares_focal():
    k = np.asarray(geom.intr_to_kmat(np.array([800.0, 320.0, 240.0])))
    assert k[0, 0] == k[1, 1] == 800.0
    assert (k[0, 2], k[1, 2]) == (320.0, 240.0)


def test_intr_to_kmat_batched():
    intrs = np.array([[800.0, 810.0, 320.0, 240.0], [700.0, 700.0, 1.0, 2.0]])
    kmats = np.asarray(geom.intr_to_kmat(intrs))
    assert kmats.shape == (2, 3, 3)
    assert kmats[1, 0, 0] == 700.0 and kmats[1, 1, 2] == 2.0


# -- projection & distortion -------------------------------------------------


def _opencv_project(pts3d, rvec, tvec, intr, dist):
    k = np.asarray(geom.intr_to_kmat(intr))
    out, _ = cv2.projectPoints(pts3d, rvec, tvec, k, dist)
    return out.reshape(-1, 2)


@pytest.mark.parametrize("ncoef", [0, 4, 5, 8, 12])
def test_project_full_matches_opencv(rng, ncoef):
    rvec = rng.normal(size=3) * 0.3
    tvec = np.array([0.1, -0.2, 4.0])
    intr = np.array([800.0, 810.0, 320.0, 240.0])
    dist = rng.normal(size=ncoef) * 0.01
    cloud = rng.normal(size=(30, 3)) + np.array([0.0, 0.0, 5.0])

    got = np.asarray(
        geom.project_full(cloud, rvec[None], tvec[None], intr[None], dist[None])
    )[0]
    expected = _opencv_project(cloud, rvec, tvec, intr, dist)
    assert np.allclose(got, expected, atol=1e-9)


def test_project_full_shared_intr_and_dist_broadcast(rng):
    # 1-D intrs/dists are shared across all views.
    rvecs = random_rvecs(rng, 3, 0.2)
    tvecs = rng.normal(size=(3, 3)) + np.array([0, 0, 5.0])
    intr = np.array([800.0, 320.0, 240.0])
    dist = np.array([0.01, -0.02, 0.001, 0.0, 0.003])
    cloud = rng.normal(size=(10, 3)) + np.array([0, 0, 5.0])

    shared = np.asarray(geom.project_full(cloud, rvecs, tvecs, intr, dist))
    per_view = np.asarray(
        geom.project_full(
            cloud,
            rvecs,
            tvecs,
            np.broadcast_to(intr, (3, 3)),
            np.broadcast_to(dist, (3, 5)),
        )
    )
    assert shared.shape == (3, 10, 2)
    assert np.allclose(shared, per_view)


def test_project_full_preserves_point_batch_dims(rng):
    rvecs = random_rvecs(rng, 2, 0.2)
    tvecs = rng.normal(size=(2, 3)) + np.array([0, 0, 5.0])
    intr = np.array([800.0, 800.0, 1.0, 2.0])
    cloud = rng.normal(size=(4, 5, 3)) + np.array([0, 0, 5.0])
    out = geom.project_full(cloud, rvecs, tvecs, intr, np.zeros(0))
    assert out.shape == (2, 4, 5, 2)


@pytest.mark.parametrize("ncoef", [4, 5, 8, 12])
def test_distort_matches_opencv(rng, ncoef):
    # With K = I, rvec = tvec = 0 and z = 1, cv2.projectPoints returns exactly
    # the distorted normalized coordinates -- a direct check of ``distort``.
    xy = rng.uniform(-0.4, 0.4, size=(50, 2))
    dist = rng.normal(size=ncoef) * 0.01
    pts3d = np.column_stack([xy, np.ones(len(xy))])
    expected, _ = cv2.projectPoints(pts3d, np.zeros(3), np.zeros(3), np.eye(3), dist)
    expected = expected.reshape(-1, 2)
    got = np.asarray(geom.distort(xy[None], dist[None]))[0]
    assert np.allclose(got, expected, atol=1e-10)


def test_distort_empty_is_identity(rng):
    xy = rng.uniform(-1, 1, size=(3, 7, 2))
    out = geom.distort(xy, np.zeros((3, 0)))
    assert np.allclose(np.asarray(out), xy)


# -- projection matrices -----------------------------------------------------


def test_project_pmat_matches_project_full(rng, rig):
    cloud = rng.uniform(-0.5, 0.5, size=(12, 3))
    rtmat = np.concatenate(
        [np.asarray(geom.rvec_to_rmat(rig["rvecs"])), rig["tvecs"][..., None]], axis=-1
    )
    pmats = np.asarray(geom.intr_to_kmat(rig["intrs"])) @ rtmat
    via_pmat = np.asarray(geom.project_pmat(cloud, pmats))
    via_full = np.asarray(
        geom.project_full(cloud, rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"])
    )
    assert via_pmat.shape == via_full.shape == (7, 12, 2)
    assert np.allclose(via_pmat, via_full, atol=1e-6)


# -- triangulation -----------------------------------------------------------


def _pmats(rig):
    rtmat = np.concatenate(
        [np.asarray(geom.rvec_to_rmat(rig["rvecs"])), rig["tvecs"][..., None]], axis=-1
    )
    return np.asarray(geom.intr_to_kmat(rig["intrs"])) @ rtmat


def test_triangulate_dlt_matches_opencv(rng, rig):
    cloud = rng.uniform(-0.5, 0.5, size=(20, 3))
    pmats = _pmats(rig)[:2]
    pts2d = np.asarray(geom.project_pmat(cloud, pmats))

    homog = cv2.triangulatePoints(pmats[0], pmats[1], pts2d[0].T, pts2d[1].T)
    expected = (homog[:3] / homog[3]).T
    got = np.asarray(geom.triangulate_dlt(pts2d, pmats))
    assert np.allclose(got, expected, atol=1e-8)


def test_triangulate_dlt_recovers_points(rng, rig):
    cloud = rng.uniform(-0.5, 0.5, size=(50, 3))
    pmats = _pmats(rig)
    pts2d = np.asarray(geom.project_pmat(cloud, pmats))
    got = np.asarray(geom.triangulate_dlt(pts2d, pmats))
    assert np.allclose(got, cloud, atol=1e-6)


def test_triangulate_dlt_handles_missing_observations(rng, rig):
    cloud = rng.uniform(-0.5, 0.5, size=(5, 3))
    pmats = _pmats(rig)
    pts2d = np.array(geom.project_pmat(cloud, pmats))

    # Point 0 seen in only one view -> NaN; point 1 missing two views but still
    # has >= 2 and should be recovered.
    pts2d[1:, 0] = np.nan
    pts2d[:2, 1] = np.nan
    got = np.asarray(geom.triangulate_dlt(pts2d, pmats))

    assert np.all(np.isnan(got[0]))
    assert np.allclose(got[1:], cloud[1:], atol=1e-6)


def test_triangulate_dlt_batched_point_shape(rng, rig):
    cloud = rng.uniform(-0.5, 0.5, size=(3, 4, 3))
    pmats = _pmats(rig)
    pts2d = np.asarray(geom.project_pmat(cloud, pmats))
    got = np.asarray(geom.triangulate_dlt(pts2d, pmats))
    assert got.shape == (3, 4, 3)
    assert np.allclose(got, cloud, atol=1e-6)


def test_triangulate_dlt_weighted_zero_weight_matches_drop(rng, rig):
    # Weighting a view to zero is exactly the NaN-drop of that view; clamping
    # also covers non-positive / non-finite weights (sqrt never sees a negative).
    cloud = rng.uniform(-0.5, 0.5, size=(5, 3))
    pmats = _pmats(rig)
    pts2d = np.array(geom.project_pmat(cloud, pmats))
    pts2d[0] += 30.0  # corrupt view 0

    dropped = pts2d.copy()
    dropped[0] = np.nan
    via_nan = np.asarray(geom.triangulate_dlt(dropped, pmats))

    for bad in (0.0, -2.0, np.nan, np.inf):
        w = np.ones(pts2d.shape[:-1])
        w[0] = bad
        got = np.asarray(geom.triangulate_dlt(pts2d, pmats, w))
        assert np.isfinite(got).all()
        assert np.allclose(got, via_nan, atol=1e-6)


# -- undistortion / back-projection ------------------------------------------


def test_undistort_inverts_distort(rng):
    # Round-trip through the full radial + tangential + thin-prism model.
    dist = np.array([0.12, -0.05, 0.001, -0.0015, 0.01, 0.02, -0.001, 0.0005])
    for _ in range(50):
        xy = rng.uniform(-0.3, 0.3, size=2)
        xy_d = np.asarray(geom.distort_one(xy, dist))
        xy_rec = np.asarray(geom.undistort_one(xy_d, dist))
        assert np.allclose(xy_rec, xy, atol=1e-9)


def test_undistort_empty_is_identity(rng):
    xy = rng.uniform(-1.0, 1.0, size=2)
    assert np.allclose(np.asarray(geom.undistort_one(xy, np.zeros(0))), xy)


def test_undistort_matches_opencv(rng):
    # Cross-check against cv2.undistortPoints for a 5-coefficient model.
    dist = np.array([0.1, -0.03, 0.001, -0.002, 0.005])
    kmat = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    for _ in range(25):
        xy = rng.uniform(-0.25, 0.25, size=2)
        xy_d = np.asarray(geom.distort_one(xy, dist))
        pixel = (kmat @ np.array([xy_d[0], xy_d[1], 1.0]))[:2]
        expected = cv2.undistortPoints(
            pixel.reshape(1, 1, 2), kmat, dist.reshape(1, -1)
        ).reshape(2)
        got = np.asarray(geom.undistort_one(xy_d, dist))
        assert np.allclose(got, expected, atol=1e-6)


def test_backproject_ray_reprojects_to_pixel(cameras, rng):
    cam = next(iter(cameras))
    for _ in range(20):
        pt3d = rng.uniform(-1.5, 1.5, size=3)
        pixel = cam.project(pt3d)
        origin, direction = cam.backproject_ray(pixel)
        # the original point lies on its own back-projection ray
        on_ray = np.asarray(geom.closest_point_on_ray(origin, direction, pt3d))
        assert np.allclose(on_ray, pt3d, atol=1e-6)
        assert np.allclose(cam.project(on_ray), pixel, atol=1e-6)


def test_closest_point_on_ray_reprojects_and_is_closest(cameras, rng):
    cam = next(iter(cameras))
    for _ in range(20):
        x_old = rng.uniform(-1.5, 1.5, size=3)
        # drag to a pixel away from x_old's projection
        pixel = cam.project(x_old) + rng.uniform(-20.0, 20.0, size=2)
        origin, direction = cam.backproject_ray(pixel)
        x_new = np.asarray(geom.closest_point_on_ray(origin, direction, x_old))
        # (1) reprojects exactly onto the dragged pixel
        assert np.allclose(cam.project(x_new), pixel, atol=1e-5)
        # (2) no other point on the ray is closer to x_old
        d_new = np.linalg.norm(x_new - x_old)
        for s in (-0.7, -0.2, 0.3, 1.1):
            other = x_new + s * np.asarray(direction)
            assert d_new <= np.linalg.norm(other - x_old) + 1e-9
