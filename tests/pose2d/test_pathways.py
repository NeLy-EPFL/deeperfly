"""Tests for the detection plan: what the synthesis produces, the (i, v, p) scatter and
the coordinate inverse that maps a model peak back into raw footage pixels.

The plan is no longer parsed, it is **synthesized** from the camera table -- one source,
one preprocessor, one identity-mapped pathway per camera, through the single detector
``[pose2d] class`` / ``weights`` names. So most of what this file used to test (every
cross-reference between ``[[sources]]``, ``[[pose2d.preprocessors]]``,
``[[pose2d.models]]``, ``[[pose2d.pathways]]`` and ``[pose2d.output_points]``) has no
config surface left to get wrong. What survives is the arithmetic -- the scatter and the
coordinate inverse -- plus the two things a config can still name that does not exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.pose2d.pathways import (
    normalized_peaks_to_original_pixels,
    route_channels_to_points_in_views,
)
from deeperfly.preprocessing import Crop, FrameTransform, Resize

WEIGHTS = "mvt_r28_pad48_gray_fly38.pth"


def _config(cameras=None, **pose2d) -> Config:
    """A minimal two-camera config, with ``pose2d`` merged into ``[pose2d]``."""
    return Config.from_dict(
        {
            "default_camera": {"distance": 100, "focal_length_px": 1},
            "cameras": cameras
            or {"rh": {"azimuth_deg": -120}, "lf": {"azimuth_deg": 45}},
            "pose2d": {"class": "mvt", "weights": WEIGHTS, **pose2d},
        }
    )


# -- coordinate inverse (normalized_peaks_to_original_pixels) ----------------


def test_normalized_peaks_to_original_pixels_plain_scales_to_source_pixels():
    # No window: a model peak at normalized (x, y) maps to ~ (x*W, y*H) in raw
    # pixels (within the half-pixel resize convention).
    pts = np.array([[0.5, 0.5]])
    out = normalized_peaks_to_original_pixels(
        pts, FrameTransform(()), (256, 512), (480, 960)
    )
    np.testing.assert_allclose(out, [[0.5 * 960, 0.5 * 480]], atol=1.0)


def test_normalized_peaks_to_original_pixels_undoes_the_window():
    """The whole reason a camera keeps RAW intrinsics: the window is inverted here.

    A peak decoded inside a crop has to come back out at ``crop origin + offset``, or
    every detection through a windowed camera lands short by exactly the crop offset --
    which reprojects plausibly and is what makes the bug quiet.
    """
    src = (480, 960)
    window = FrameTransform((Crop(x=100, y=40, width=480, height=240),))
    plain = normalized_peaks_to_original_pixels(
        np.array([[0.5, 0.5]]), FrameTransform(()), (256, 512), src
    )
    cropped = normalized_peaks_to_original_pixels(
        np.array([[0.5, 0.5]]), window, (256, 512), src
    )
    np.testing.assert_allclose(plain[0], [(960 - 1) / 2, (480 - 1) / 2], atol=1.0)
    np.testing.assert_allclose(
        cropped[0], [100 + (480 - 1) / 2, 40 + (240 - 1) / 2], atol=1.0
    )


def test_normalized_peaks_to_original_pixels_roundtrips_with_map_points():
    transform = FrameTransform((Crop(x=8, y=4, width=480, height=240),))
    src = (480, 960)
    norm = np.array([[0.2, 0.7], [0.9, 0.1]])
    view_px = normalized_peaks_to_original_pixels(norm, transform, (256, 512), src)
    # Forward map (raw -> windowed -> model input) then normalize recovers norm.
    windowed_px = transform.map_points(view_px, src)
    resize = FrameTransform((Resize(width=512, height=256),))
    back = resize.map_points(windowed_px, transform.output_size(src)) / np.array(
        [512, 256]
    )
    np.testing.assert_allclose(back, norm, atol=1e-6)


# -- scatter ------------------------------------------------------------------


def test_route_channels_to_points_in_views_routes_channels_and_leaves_nan():
    raw = np.array([[1.0, 2.0], [3.0, 4.0]])
    conf = np.array([0.5, 0.75])
    mapping = np.array([[0, 0, 1], [1, 1, 0]])  # ch0 -> (v0,p1), ch1 -> (v1,p0)
    pts = np.full((2, 2, 2), np.nan)
    out_conf = np.zeros((2, 2))
    route_channels_to_points_in_views(raw, conf, mapping, pts, out_conf)
    np.testing.assert_allclose(pts[0, 1], [1.0, 2.0])
    np.testing.assert_allclose(pts[1, 0], [3.0, 4.0])
    assert np.isnan(pts[0, 0]).all() and np.isnan(pts[1, 1]).all()
    assert out_conf[0, 1] == 0.5 and out_conf[1, 0] == 0.75
    assert out_conf[0, 0] == 0.0 and out_conf[1, 1] == 0.0


def test_route_channels_to_points_in_views_candidate_axis():
    """The trailing K axis rides along, so one scatter serves peaks and candidates."""
    raw = np.arange(2 * 3 * 2, dtype=float).reshape(2, 3, 2)
    conf = np.arange(2 * 3, dtype=float).reshape(2, 3)
    mapping = np.array([[0, 0, 0], [1, 0, 1]])
    pts = np.full((1, 2, 3, 2), np.nan)
    out_conf = np.zeros((1, 2, 3))
    route_channels_to_points_in_views(raw, conf, mapping, pts, out_conf)
    np.testing.assert_allclose(pts[0, 0], raw[0])
    np.testing.assert_allclose(pts[0, 1], raw[1])
    np.testing.assert_allclose(out_conf[0], conf)


# -- what the synthesis produces ----------------------------------------------


def test_a_camera_is_a_source_is_a_pathway_is_a_view():
    """The invariant the whole schema rests on, asserted on the names themselves."""
    plan = _config().detection_plan()
    assert plan.view_names == ["rh", "lf"]
    assert [s.name for s in plan.sources] == ["rh", "lf"]
    assert [pw.name for pw in plan.pathways] == ["rh", "lf"]
    assert [pw.source for pw in plan.pathways] == ["rh", "lf"]
    assert plan.view_sources() == {"rh": "rh", "lf": "lf"}


def test_the_mapping_is_the_identity_into_the_cameras_own_view():
    """Channel i -> point i of this camera -- what DENSE means, and 38 x V rows saved."""
    plan = _config().detection_plan()
    for v, pw in enumerate(plan.pathways):
        assert pw.mapping.shape == (plan.n_points, 3)
        np.testing.assert_array_equal(pw.mapping[:, 0], np.arange(plan.n_points))
        np.testing.assert_array_equal(pw.mapping[:, 1], v)
        np.testing.assert_array_equal(pw.mapping[:, 2], np.arange(plan.n_points))
    # Every (view, point) is written, which is the other face of the same fact.
    assert plan.visibility_mask().all()


def test_one_model_for_the_whole_run():
    plan = _config().detection_plan()
    assert list(plan.models) == ["mvt"]
    spec = plan.models["mvt"]
    assert spec.cls == "mvt" and spec.weights == WEIGHTS
    # The channel count comes from the SKELETON, so a config cannot get it wrong.
    assert spec.n_out_channels == plan.n_points
    # And every pathway forwards through it.
    assert {pw.model for pw in plan.pathways} == {"mvt"}


def test_a_camera_with_no_crop_has_no_preprocessor():
    plan = _config().detection_plan()
    assert plan.preprocessors == {}
    assert all(pw.preprocessor is None for pw in plan.pathways)
    assert all(pw.transform.is_identity() for pw in plan.pathways)


def test_a_fixed_crop_becomes_that_cameras_window():
    plan = _config(crops={"rh": [10, 20, 300, 200]}).detection_plan()
    assert list(plan.preprocessors) == ["rh"]
    rh = next(pw for pw in plan.pathways if pw.name == "rh")
    assert rh.preprocessor == "rh"
    assert rh.transform.ops == (Crop(x=10, y=20, width=300, height=200),)
    assert next(pw for pw in plan.pathways if pw.name == "lf").preprocessor is None


def test_a_searched_camera_gets_an_unresolved_window():
    plan = _config(auto_crops=["rh"]).detection_plan()
    rh = plan.preprocessors["rh"]
    assert rh.needs_auto_crop
    assert rh.auto_crop.seed is None
    # And the plan reports it as a window, which is what a panel borrowing one reads.
    assert set(plan.view_transforms()) == {"rh"}


def test_a_box_plus_auto_crops_is_a_seeded_search():
    """The third v1 form (`{ op = "crop", auto = true, x = ... }`) falls out for free."""
    plan = _config(auto_crops=["rh"], crops={"rh": [10, 20, 300, 200]}).detection_plan()
    auto = plan.preprocessors["rh"].auto_crop
    assert auto is not None and not auto.resolved
    assert auto.seed == (10, 20, 300, 200)


def test_model_precision_defaults_to_the_class_requirement():
    """mvt only runs in float32 and its loader refuses anything else."""
    assert _config().detection_plan().models["mvt"].precision == "float32"
    assert (
        _config(precision="bfloat16").detection_plan().models["mvt"].precision
        == "bfloat16"
    )


# -- the two things a config can still get wrong ------------------------------


@pytest.mark.parametrize(
    "pose2d, message",
    [
        ({"crops": {"nope": [0, 0, 10, 10]}}, r"\[pose2d.crops\] names 'nope'"),
        ({"crops": {"rh": [0, 0, 10]}}, r"must be \[x, y, width, height\]"),
        ({"crops": [0, 0, 10, 10]}, r"\[pose2d.crops\] must be a table"),
        ({"auto_crops": ["nope"]}, r"auto_crops names 'nope'"),
        ({"auto_crops": "rh"}, r"auto_crops must be a list"),
    ],
)
def test_a_crop_naming_a_camera_that_does_not_exist_is_refused(pose2d, message):
    with pytest.raises(ValueError, match=message):
        _config(**pose2d).detection_plan()


def test_a_plan_with_no_cameras_is_refused():
    with pytest.raises(ValueError, match="needs cameras"):
        Config.from_dict(
            {"cameras": {}, "pose2d": {"class": "mvt", "weights": WEIGHTS}}
        ).detection_plan()


def test_a_plan_with_no_detector_class_is_refused():
    with pytest.raises(ValueError, match="needs a string 'class'"):
        Config.from_dict(
            {
                "default_camera": {"distance": 100, "focal_length_px": 1},
                "cameras": {"rh": {"azimuth_deg": 0}},
                "pose2d": {"weights": WEIGHTS},
            }
        ).detection_plan()


@pytest.mark.parametrize(
    "key",
    ["models", "model", "pathways", "preprocessors", "output_points", "autocrop"],
)
def test_a_v1_pose2d_key_is_refused_by_name(key):
    """Named, never ignored: every one of these would otherwise change nothing at all."""
    with pytest.raises(ValueError, match=key):
        _config(**{key: "whatever"}).pose2d()


# -- footage by view ----------------------------------------------------------


def test_footage_by_view_is_now_the_identity_but_still_recorded():
    """The pipeline resolves footage keyed by camera and records it keyed by view.

    Those are the same key now, so this can no longer go wrong -- but the recording is
    what the viewer matches each camera's frames against, and the regression it was
    written for (a blank frame per camera) is worth keeping a gate on.
    """
    from pathlib import Path

    from deeperfly.pipeline.run import _footage_by_view

    config = _config()
    footage = {"rh": [Path("/foot/camera_RH.mp4")], "lf": [Path("/foot/camera_LF.mp4")]}
    assert _footage_by_view(config, footage) == footage
    assert _footage_by_view(config, None) is None
    assert _footage_by_view(config, {}) is None
