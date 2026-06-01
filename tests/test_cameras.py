"""Tests for :mod:`deeperfly.cameras`.

Covers the extrinsic resolver (the small spec grammar of rvec / rotation
matrix / forward-up / look-at orbit), the :class:`Camera` conveniences, and the
:class:`CameraGroup` config loader and geometry round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import geometry as geom
from deeperfly.cameras import Camera, CameraGroup, resolve_extrinsics
from helpers import (
    AZIMUTHS_DEG,
    CAMERA_NAMES,
    DISTANCE_MM,
    FOCAL_PX,
    HEIGHT,
    WIDTH,
    reference_rmat,
)


@pytest.fixture
def config() -> dict:
    """Config dict equivalent to examples/cameras.toml (cameras only)."""
    return {
        "camera_defaults": {
            "focal_length_px": [FOCAL_PX, FOCAL_PX],
            "principal_point_px": [(WIDTH - 1) / 2, (HEIGHT - 1) / 2],
            "distortion_coefficients": [],
            "look_at": [0.0, 0.0, 0.0],
            "distance": DISTANCE_MM,
            "elevation_deg": 0.0,
            "roll_deg": 0.0,
        },
        "cameras": {
            name: {"azimuth_deg": az} for name, az in zip(CAMERA_NAMES, AZIMUTHS_DEG)
        },
    }


# -- resolve_extrinsics ------------------------------------------------------


def test_resolve_orbit_matches_reference_convention():
    for az in AZIMUTHS_DEG:
        rvec, tvec = resolve_extrinsics(
            {"look_at": [0, 0, 0], "distance": DISTANCE_MM, "azimuth_deg": az}
        )
        expected_rmat = reference_rmat(np.deg2rad(az))
        assert np.allclose(
            np.asarray(geom.rvec_to_rmat(rvec)), expected_rmat, atol=1e-12
        )
        assert np.allclose(tvec, [0.0, 0.0, DISTANCE_MM], atol=1e-9)


def test_resolve_rvec_passthrough(rng):
    rvec_in = rng.normal(size=3) * 0.5
    rvec, tvec = resolve_extrinsics({"rvec": rvec_in.tolist(), "tvec": [1.0, 2.0, 3.0]})
    assert np.allclose(rvec, rvec_in, atol=1e-12)
    assert np.allclose(tvec, [1.0, 2.0, 3.0])


def test_resolve_rotation_matrix_and_position():
    rmat = reference_rmat(np.deg2rad(30.0))
    rvec, tvec = resolve_extrinsics(
        {"rotation_matrix": rmat.tolist(), "position": [0.0, 0.0, 5.0]}
    )
    assert np.allclose(np.asarray(geom.rvec_to_rmat(rvec)), rmat, atol=1e-12)
    # tvec encodes the world camera center: center == -R^T t.
    center = -np.asarray(geom.rvec_to_rmat(rvec)).T @ tvec
    assert np.allclose(center, [0.0, 0.0, 5.0], atol=1e-9)


def test_resolve_forward_up():
    rvec, _ = resolve_extrinsics(
        {"forward": [0.0, 0.0, 1.0], "up": [0.0, -1.0, 0.0], "position": [0, 0, -3.0]}
    )
    rmat = np.asarray(geom.rvec_to_rmat(rvec))
    assert np.allclose(rmat[2], [0.0, 0.0, 1.0], atol=1e-12)  # optical axis = forward


def test_resolve_roll_composes_about_optical_axis():
    # forward must not be parallel to the default up (+z) for look-at to be well posed.
    base, _ = resolve_extrinsics({"forward": [1, 0, 0.0], "tvec": [0, 0, 0]})
    rolled, _ = resolve_extrinsics(
        {"forward": [1, 0, 0.0], "tvec": [0, 0, 0], "roll_deg": 90.0}
    )
    rb = np.asarray(geom.rvec_to_rmat(base))
    rr = np.asarray(geom.rvec_to_rmat(rolled))
    # optical axis unchanged by roll
    assert np.allclose(rb[2], rr[2], atol=1e-12)
    # a 90 deg roll maps the camera x-axis onto +/- the camera y-axis
    assert np.allclose(np.abs(rr[0] @ rb[1]), 1.0, atol=1e-12)


@pytest.mark.parametrize(
    "spec",
    [
        {"rvec": [0, 0, 0], "rotation_matrix": np.eye(3).tolist(), "tvec": [0, 0, 0]},
        {"position": [0, 0, 1], "center": [0, 0, 1], "rvec": [0, 0, 0]},
        {"tvec": [0, 0, 0], "position": [0, 0, 1], "rvec": [0, 0, 0]},
    ],
)
def test_resolve_conflicting_keys_raise(spec):
    with pytest.raises(ValueError):
        resolve_extrinsics(spec)


def test_resolve_missing_rotation_raises():
    with pytest.raises(ValueError, match="rotation"):
        resolve_extrinsics({"position": [0, 0, 1]})


def test_resolve_lookat_without_position_raises():
    with pytest.raises(ValueError, match="position"):
        resolve_extrinsics({"look_at": [0, 0, 0]})


# -- Camera ------------------------------------------------------------------


def test_camera_position_roundtrip(rng):
    rvec = rng.normal(size=3) * 0.4
    tvec = rng.normal(size=3)
    cam = Camera(
        rvec=rvec, tvec=tvec, intr=np.array([800.0, 800, 1, 2]), dist=np.zeros(0)
    )
    # center = -R^T t  =>  t = -R @ center
    assert np.allclose(cam.rmat @ cam.position, -tvec, atol=1e-10)


def test_camera_project_matches_geometry(rng):
    cam = Camera.from_spec(
        {
            "rvec": (rng.normal(size=3) * 0.2).tolist(),
            "tvec": [0.1, 0.2, 5.0],
            "focal_length_px": [800.0, 810.0],
            "principal_point_px": [320.0, 240.0],
            "distortion_coefficients": [0.01, -0.02, 0.001, 0.0],
        }
    )
    cloud = rng.normal(size=(15, 3)) + np.array([0, 0, 5.0])
    expected = np.asarray(
        geom.project_full(
            cloud, cam.rvec[None], cam.tvec[None], cam.intr[None], cam.dist[None]
        )
    )[0]
    assert np.allclose(cam.project(cloud), expected, atol=1e-12)


def test_parse_intrinsics_scalar_focal():
    cam = Camera.from_spec(
        {
            "rvec": [0, 0, 0],
            "tvec": [0, 0, 1],
            "focal_length_px": 700.0,
            "principal_point_px": [10.0, 20.0],
        }
    )
    assert cam.intr.tolist() == [700.0, 700.0, 10.0, 20.0]


# -- CameraGroup -------------------------------------------------------------


def test_group_from_config_dict(config):
    group = CameraGroup.from_config(config)
    assert group.names == CAMERA_NAMES
    rmats = np.array([reference_rmat(np.deg2rad(az)) for az in AZIMUTHS_DEG])
    assert np.allclose(np.asarray(geom.rvec_to_rmat(group.rvecs)), rmats, atol=1e-12)
    assert np.allclose(group.tvecs, [[0, 0, DISTANCE_MM]] * len(group), atol=1e-9)
    assert np.allclose(
        group.intrs, [[FOCAL_PX, FOCAL_PX, (WIDTH - 1) / 2, (HEIGHT - 1) / 2]] * 7
    )


def test_group_from_config_toml_file(tmp_path):
    toml = """
    [camera_defaults]
    focal_length_px = 800.0
    principal_point_px = [320.0, 240.0]

    [cameras.left]
    rvec = [0.0, 0.0, 0.0]
    tvec = [0.0, 0.0, 5.0]

    [cameras.right]
    rvec = [0.0, 0.1, 0.0]
    tvec = [1.0, 0.0, 5.0]
    """
    path = tmp_path / "cams.toml"
    path.write_text(toml)
    group = CameraGroup.from_config(path)
    assert group.names == ["left", "right"]
    assert np.allclose(group["left"].intr, [800.0, 800.0, 320.0, 240.0])


def test_group_empty_config_raises():
    with pytest.raises(ValueError, match="no cameras"):
        CameraGroup.from_config({"cameras": {}})


def test_group_project_triangulate_roundtrip(config, rng):
    group = CameraGroup.from_config(config)
    cloud = rng.uniform(-0.5, 0.5, size=(60, 3))
    pts2d = group.project(cloud)
    assert pts2d.shape == (7, 60, 2)
    recovered = group.triangulate(pts2d)
    assert np.allclose(recovered, cloud, atol=1e-6)


def test_group_from_arrays_roundtrip(rig):
    group = CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )
    assert group.names == rig["names"]
    assert np.allclose(group.rvecs, rig["rvecs"])
    assert group["f"].name == "f"


def test_group_dists_zero_padded():
    cams = {
        "a": Camera(
            np.zeros(3),
            np.array([0, 0, 1.0]),
            np.array([1.0, 1, 0, 0]),
            np.array([0.1, 0.2]),
        ),
        "b": Camera(
            np.zeros(3),
            np.array([0, 0, 1.0]),
            np.array([1.0, 1, 0, 0]),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        ),
    }
    group = CameraGroup(cams)
    dists = group.dists
    assert dists.shape == (2, 5)
    assert np.allclose(dists[0], [0.1, 0.2, 0, 0, 0])
    assert np.allclose(dists[1], [0.1, 0.2, 0.3, 0.4, 0.5])
