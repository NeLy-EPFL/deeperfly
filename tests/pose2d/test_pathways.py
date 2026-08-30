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
    check_render_aspect,
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


# -- the rendering aspect check (check_render_aspect) ------------------------


class _Model:
    """The only thing the aspect check reads off a loaded model."""

    def __init__(self, input_size=(256, 512)):
        self.input_size = input_size


def _aspect_messages(sizes, **pose2d):
    plan = _config(**pose2d).detection_plan()
    return check_render_aspect(plan, {"mvt": _Model()}, sizes)


def test_the_rigs_that_exist_do_not_warn():
    """The tolerance has to admit today's corpus or the check is noise.

    Every window in ~/fly-pose-data/project is 2.0000 (the axial ones) or 1.8750 (the
    side cameras, and the flywheel strips cut to match them). 960x512 into a 2:1 network
    is a 6.7% stretch, and it is the same stretch in training and at inference, which is
    the only reason it has never cost anything.
    """
    assert _aspect_messages({"rh": (512, 960), "lf": (512, 960)}) == []
    assert _aspect_messages({"rh": (480, 960), "lf": (480, 960)}) == []


def test_a_new_rig_dropped_in_whole_warns():
    """A 1.6:1 frame resized to 2:1 is a 25% squeeze -- four times anything trained on."""
    (message,) = _aspect_messages({"rh": (800, 1280), "lf": (512, 960)})
    assert "'rh'" in message and "the full frame" in message
    assert "25% vertical squeeze" in message


def test_a_fixed_window_of_the_wrong_shape_warns():
    (message,) = _aspect_messages(
        {"rh": (1008, 1600), "lf": (512, 960)}, crops={"rh": [200, 100, 800, 700]}
    )
    assert "its window" in message and "800x700" in message


def test_a_window_at_the_models_aspect_does_not_warn():
    assert (
        _aspect_messages(
            {"rh": (1008, 1600), "lf": (512, 960)}, crops={"rh": [291, 313, 1040, 520]}
        )
        == []
    )


def test_a_seeded_search_is_checked_and_a_blind_one_is_not():
    """The search locks its aspect to the seed, so a bad seed is never corrected."""
    sizes = {"rh": (1008, 1600), "lf": (512, 960)}
    (message,) = _aspect_messages(
        sizes, auto_crops=["rh"], crops={"rh": [200, 100, 800, 700]}
    )
    assert "the seed of its automatic crop" in message
    # Blind: the search adopts the model's own aspect, so there is nothing to warn about.
    assert _aspect_messages(sizes, auto_crops=["rh"]) == []


# -- [pose2d] fit: never stretch ---------------------------------------------


def test_fit_defaults_to_stretch_so_nothing_moves():
    """Every shipped checkpoint was trained through the per-axis resize."""
    plan = _config().detection_plan()
    assert plan.preprocessors == {}  # no window, no pad, no change


def test_fit_pad_puts_every_camera_at_the_models_aspect():
    """A camera not at 2:1 gains a border; one already at 2:1 gains nothing."""
    plan = _config(fit="pad", crops={"lf": [291, 313, 1040, 520]}).detection_plan()
    sizes = {"rh": (512, 960), "lf": (1008, 1600)}
    for pw in plan.pathways:
        h, w = pw.transform.output_size(sizes[pw.name])
        assert w / h == pytest.approx(512 / 256), pw.name
    # the axial window is untouched, the side camera is padded 960 -> 1024
    assert plan.preprocessors["lf"].output_size((1008, 1600)) == (520, 1040)
    assert plan.preprocessors["rh"].output_size((512, 960)) == (512, 1024)


def test_fit_pad_silences_the_aspect_warning_it_exists_to_answer():
    """The guard and `fit` are one policy from two ends, so pinning them together.

    A 1.6:1 rig dropped in whole is the 25% squeeze `check_render_aspect` was written to
    catch; under `fit = "pad"` there is nothing left to catch.
    """
    sizes = {"rh": (800, 1280), "lf": (512, 960)}
    assert len(_aspect_messages(sizes)) == 1
    plan = _config(fit="pad").detection_plan()
    assert check_render_aspect(plan, {"mvt": _Model()}, sizes) == []


def test_fit_pad_still_lets_a_searched_window_be_resolved():
    """The pad is appended AFTER the automatic crop, so the search is untouched."""
    plan = _config(fit="pad", auto_crops=["rh"]).detection_plan()
    rh = plan.preprocessors["rh"]
    assert rh.needs_auto_crop and rh.auto_crop.seed is None
    resolved = rh.resolve_auto_crop((100, 50, 600, 300))
    assert resolved.output_size((512, 960)) == (
        300,
        600,
    )  # 2:1 already, pad adds nothing


@pytest.mark.parametrize("bad", ["letterbox", "", 2])
def test_fit_rejects_anything_it_does_not_implement(bad):
    with pytest.raises(ValueError, match="fit must be"):
        _config(fit=bad).detection_plan()
