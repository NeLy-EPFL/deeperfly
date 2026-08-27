"""Tests for :mod:`deeperfly.cameras`.

Covers the extrinsic resolver (the orbit spec: look_at / distance /
azimuth_deg / elevation_deg / roll_deg), the :class:`Camera` conveniences, and
the :class:`CameraGroup` config loader and geometry round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import (
    AZIMUTHS_DEG,
    CAMERA_NAMES,
    DISTANCE_MM,
    FOCAL_PX,
    HEIGHT,
    WIDTH,
    reference_rmat,
)

from deeperfly import geometry as geom
from deeperfly.cameras import Camera, CameraGroup, resolve_extrinsics
from deeperfly.config import Config


@pytest.fixture
def config() -> Config:
    """Reference camera rig config (cameras only)."""
    return Config.from_dict(
        {
            "default_camera": {
                "focal_length_px": [FOCAL_PX, FOCAL_PX],
                "principal_point_px": [(WIDTH - 1) / 2, (HEIGHT - 1) / 2],
                "distortion_coefficients": [],
                "look_at": [0.0, 0.0, 0.0],
                "distance": DISTANCE_MM,
                "elevation_deg": 0.0,
                "roll_deg": 0.0,
            },
            "cameras": {
                **{
                    name: {"azimuth_deg": az}
                    for name, az in zip(CAMERA_NAMES, AZIMUTHS_DEG)
                },
            },
        }
    )


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


def test_resolve_defaults_look_at_origin_and_zero_angles():
    # Only distance given: camera at [d, 0, 0] looking back at the origin.
    rvec, tvec = resolve_extrinsics({"distance": DISTANCE_MM})
    rmat = np.asarray(geom.rvec_to_rmat(rvec))
    assert np.allclose(rmat, reference_rmat(0.0), atol=1e-12)
    assert np.allclose(-rmat.T @ tvec, [DISTANCE_MM, 0.0, 0.0], atol=1e-9)


def test_resolve_orbit_places_camera_around_look_at():
    target = np.array([1.0, -2.0, 3.0])
    rvec, tvec = resolve_extrinsics(
        {
            "look_at": target.tolist(),
            "distance": 5.0,
            "azimuth_deg": 30.0,
            "elevation_deg": 20.0,
        }
    )
    rmat = np.asarray(geom.rvec_to_rmat(rvec))
    center = -rmat.T @ tvec
    # camera sits `distance` from the target with its optical axis pointing at it
    assert np.isclose(np.linalg.norm(center - target), 5.0, atol=1e-9)
    assert np.allclose(rmat[2], (target - center) / 5.0, atol=1e-12)


def test_resolve_roll_composes_about_optical_axis():
    base_spec = {"distance": 3.0, "azimuth_deg": 25.0, "elevation_deg": 10.0}
    base, _ = resolve_extrinsics(base_spec)
    rolled, _ = resolve_extrinsics({**base_spec, "roll_deg": 90.0})
    rb = np.asarray(geom.rvec_to_rmat(base))
    rr = np.asarray(geom.rvec_to_rmat(rolled))
    # optical axis unchanged by roll
    assert np.allclose(rb[2], rr[2], atol=1e-12)
    # a 90 deg roll maps the camera x-axis onto +/- the camera y-axis
    assert np.allclose(np.abs(rr[0] @ rb[1]), 1.0, atol=1e-12)


def test_resolve_missing_distance_raises():
    with pytest.raises(ValueError, match="distance"):
        resolve_extrinsics({"look_at": [0, 0, 0]})


@pytest.mark.parametrize(
    "key",
    ["rvec", "tvec", "rotation_matrix", "forward", "up", "position", "center", "eye"],
)
def test_resolve_removed_keys_raise(key):
    with pytest.raises(ValueError, match="orbit"):
        resolve_extrinsics({"distance": 1.0, key: [0, 0, 0]})


def test_resolve_straight_down_is_ambiguous():
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_extrinsics({"distance": 1.0, "elevation_deg": 90.0})


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
            "distance": 5.0,
            "azimuth_deg": 30.0,
            "elevation_deg": 10.0,
            "roll_deg": 5.0,
            "focal_length_px": [800.0, 810.0],
            "principal_point_px": [320.0, 240.0],
            "distortion_coefficients": [0.01, -0.02, 0.001, 0.0],
        }
    )
    cloud = rng.normal(size=(15, 3)) * 0.3  # near the look_at target (origin)
    expected = np.asarray(
        geom.project_full(
            cloud, cam.rvec[None], cam.tvec[None], cam.intr[None], cam.dist[None]
        )
    )[0]
    assert np.allclose(cam.project(cloud), expected, atol=1e-12)


def test_parse_intrinsics_scalar_focal():
    cam = Camera.from_spec(
        {
            "distance": 1.0,
            "focal_length_px": 700.0,
            "principal_point_px": [10.0, 20.0],
        }
    )
    assert cam.intr.tolist() == [700.0, 700.0, 10.0, 20.0]


def test_from_spec_infers_principal_point_from_image_size():
    # No principal_point_px -> image center ((w-1)/2, (h-1)/2) from (height, width).
    cam = Camera.from_spec(
        {"distance": 1.0, "focal_length_px": 700.0},
        image_size=(HEIGHT, WIDTH),
    )
    assert cam.intr.tolist() == [700.0, 700.0, (WIDTH - 1) / 2, (HEIGHT - 1) / 2]


def test_from_spec_explicit_principal_point_overrides_image_size():
    # An explicit principal point wins even when an image size is available.
    cam = Camera.from_spec(
        {
            "distance": 1.0,
            "focal_length_px": 700.0,
            "principal_point_px": [10.0, 20.0],
        },
        image_size=(HEIGHT, WIDTH),
    )
    assert cam.intr.tolist() == [700.0, 700.0, 10.0, 20.0]


def test_from_spec_missing_principal_point_without_image_size_raises():
    with pytest.raises(ValueError, match="principal_point_px"):
        Camera.from_spec({"distance": 1.0, "focal_length_px": 700.0})


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
    [default_camera]
    focal_length_px = 800.0
    principal_point_px = [320.0, 240.0]
    distance = 5.0

    [cameras.left]
    azimuth_deg = 0.0

    [cameras.right]
    azimuth_deg = 90.0
    """
    path = tmp_path / "cams.toml"
    path.write_text(toml)
    group = CameraGroup.from_config(Config.from_toml(path))
    assert group.names == ["left", "right"]
    assert np.allclose(group["left"].intr, [800.0, 800.0, 320.0, 240.0])


def test_group_empty_config_raises():
    with pytest.raises(ValueError, match="no cameras"):
        CameraGroup.from_config(Config.from_dict({"cameras": {}}))


def test_group_from_config_infers_principal_point_per_view():
    # Defaults omit principal_point_px; each view's center comes from image_sizes.
    config = {
        "default_camera": {"focal_length_px": 800.0, "distance": 5.0},
        "cameras": {
            "left": {"azimuth_deg": 0.0},
            "right": {"azimuth_deg": 90.0},
        },
    }
    image_sizes = {"left": (512, 1024), "right": (480, 640)}
    group = CameraGroup.from_config(Config.from_dict(config), image_sizes=image_sizes)
    assert np.allclose(
        group["left"].intr, [800.0, 800.0, (1024 - 1) / 2, (512 - 1) / 2]
    )
    assert np.allclose(
        group["right"].intr, [800.0, 800.0, (640 - 1) / 2, (480 - 1) / 2]
    )


def test_group_from_config_ignores_non_rig_keys():
    """A camera is pure geometry: the footage pattern beside it is not part of the rig.

    And its intrinsics describe the RAW frame, which is what lets the detector window a
    camera however it likes -- a detection is mapped back through the window before it
    meets a camera, so it always arrives in raw pixels.
    """
    config = {
        "default_camera": {"focal_length_px": 800.0, "distance": 5.0},
        "cameras": {
            "left": {"azimuth_deg": 0.0, "video": "cam0.mp4"},
        },
    }
    image_sizes = {"left": (100, 100)}
    group = CameraGroup.from_config(Config.from_dict(config), image_sizes=image_sizes)
    # Principal point is the raw image center, unaffected by the (ignored) footage key.
    assert np.allclose(group["left"].intr, [800.0, 800.0, 49.5, 49.5])


def test_a_retired_per_camera_preprocess_key_is_refused_not_ignored():
    """The key cropped a view once; [pose2d.crops] does it now, and both cannot.

    A detection window is inverted on the way back, so detections land in raw footage
    pixels and the camera keeps raw intrinsics. The retired key instead moved the CAMERA
    into cropped-pixel space -- so honoring both would double-correct by exactly the crop
    offset, with nothing in the output to point at.

    It has to REFUSE rather than ignore, and that is the whole point of the test: a crop is
    what a badly-framed axial camera needs, so a silently-dropped crop key is wrong in the
    one situation where it costs most. The message has to carry the replacement, because
    "your crop did nothing" is not actionable on its own.
    """
    config = {
        "default_camera": {"focal_length_px": 800.0, "distance": 5.0},
        "cameras": {
            "left": {"azimuth_deg": 0.0, "preprocess": [{"op": "fliplr"}]},
        },
    }
    with pytest.raises(
        ValueError, match=r"\[cameras\.left\] carries 'preprocess'"
    ) as e:
        Config.from_dict(config).camera_table()
    assert "[pose2d.crops]" in str(e.value)


def test_the_retired_key_is_refused_under_the_shared_table_too():
    """`[default_camera]` is where someone would put it to crop every camera."""
    config = {
        "default_camera": {
            "focal_length_px": 800.0,
            "distance": 5.0,
            "preprocess": [{"op": "fliplr"}],
        },
        "cameras": {
            "left": {"azimuth_deg": 0.0},
        },
    }
    with pytest.raises(ValueError, match=r"\[default_camera\] carries 'preprocess'"):
        Config.from_dict(config).camera_table()


def test_a_leftover_mirror_key_is_refused_by_name():
    """It named the camera seeing this one's mirror image, for flip augmentation.

    Refused rather than ignored, and that is the whole point: the previous release kept
    it as "a fact about how the rig was built" with no reader left, so a config carrying
    it was already running without it -- and silence there is exactly what a reader would
    take for "still honored". The message has to say what replaces it, which is nothing
    in the config: the mirror is the camera at the negated azimuth.
    """
    config = {
        "default_camera": {"focal_length_px": 800.0, "distance": 5.0},
        "cameras": {"left": {"azimuth_deg": 45.0, "mirror": "right"}},
    }
    with pytest.raises(ValueError, match=r"\[cameras\.left\] carries 'mirror'") as e:
        Config.from_dict(config).camera_table()
    assert "azimuth_deg" in str(e.value)


def test_a_camera_called_defaults_is_refused_by_name():
    """The v1 spelling of the shared values, and now an illegal camera name.

    [cameras] is a pure name -> camera map, which is what makes it safe to read without
    knowing a reserved word; a `defaults` sub-table there would parse as a camera with no
    azimuth and fail somewhere else entirely.
    """
    with pytest.raises(ValueError, match=r"\[cameras\] carries 'defaults'") as e:
        Config.from_dict({"cameras": {"defaults": {"distance": 5.0}}}).camera_table()
    assert "[default_camera]" in str(e.value)


def test_a_calibration_key_under_cameras_is_refused_by_name():
    with pytest.raises(ValueError, match=r"\[cameras\] carries 'calibration'") as e:
        Config.from_dict({"cameras": {"calibration": "cal.toml"}}).camera_table()
    assert "[calibration]" in str(e.value)


def test_a_bare_key_under_cameras_is_refused():
    """Not a camera and not a reserved word -- so it can only be a mistake."""
    with pytest.raises(ValueError, match=r"non-table key\(s\) \['fps'\]"):
        Config.from_dict({"cameras": {"fps": 100}}).camera_table()


def test_group_from_config_missing_principal_point_no_sizes_raises():
    config = {
        "default_camera": {"focal_length_px": 800.0},
        "cameras": {
            "left": {"distance": 5.0},
        },
    }
    with pytest.raises(ValueError, match="principal_point_px"):
        CameraGroup.from_config(Config.from_dict(config))


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
