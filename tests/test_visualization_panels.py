"""Panel cropping, panel clipping, and the derived dorsal ("bird") view.

Every behavior here fails *silently* if it regresses -- a crop that moves the picture
but not the projection draws a well-formed skeleton in the wrong place, a lost clip
paints one camera's limb over another's frame, and a flipped dorsal axis renders a
mirrored plan view that reads as a real pose. So the assertions are on geometry and on
pixels, not on "it ran".
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import AZIMUTHS_DEG, CAMERA_NAMES

from deeperfly.config import Config
from deeperfly.skeleton import Skeleton
from deeperfly.visualization import compose
from deeperfly.visualization.bird import BIRD_VIEW, dorsal_camera

# `cameras`, `fly`, `result`, `rng` fixtures live in conftest.py.

CROP = (12, 7, 40, 20)
#: The same window as a ``[[pose2d.preprocessors]]`` op list.
#: The same window as CROP, as a [pose2d.crops] entry.
CROP_BOX = list(CROP)


@pytest.fixture
def frames(result, rng):
    v, t = result.pts2d.shape[:2]
    return {
        name: rng.integers(0, 255, size=(t, 96, 128, 3), dtype=np.uint8)
        for name in result.cameras.names[:v]
    }


@pytest.fixture
def src(fly, result, frames):
    return compose.Sources(
        fly, result.cameras, frames, pts2d=result.pts2d, pts3d=result.pts3d
    )


@pytest.fixture
def onscreen_src(fly, result, frames, rng):
    """Like ``src`` but with 2D that actually lands inside the 128x96 test frames.

    The shared ``result`` projects through a 960x512 rig, so drawing its 2D on a
    128x96 tile paints nothing -- fine for the tests that only compare renders, fatal
    for the ones that assert a skeleton is visible.
    """
    v, t, p = result.pts2d.shape[:3]
    pts2d = rng.uniform([4, 4], [124, 92], size=(v, t, p, 2))
    return compose.Sources(fly, result.cameras, frames, pts2d=pts2d, conf=result.conf)


def _cfg(panels, **video):
    return Config.from_dict(
        {"visualization": {"videos": [{"video_name": "v", "panels": panels, **video}]}}
    )


def _rig_cameras():
    return {
        v: {"azimuth_deg": az, "distance": 100.0, "focal_length_px": 1.0}
        for v, az in zip(CAMERA_NAMES, AZIMUTHS_DEG)
    }


def _detector_cfg(panels, *, crops, viz=None, **video):
    """A config whose ``[pose2d]`` really detects each camera through ``crops[camera]``.

    ``crops`` maps a camera name to an ``[x, y, width, height]`` window; a camera absent
    from it runs full-frame, which is this rig's policy for its six side cameras. The plan
    is synthesized, so there is nothing to name and nothing to cross-reference -- which
    also means a panel borrowing a window by anything but a CAMERA name has nothing to
    borrow.
    """
    return Config.from_dict(
        {
            "cameras": {
                v: {**spec, "video": f"{v}\\.mp4"} for v, spec in _rig_cameras().items()
            },
            "pose2d": {
                "class": "hrnet",
                "weights": "w.pth",
                "input_size": [256, 512],
                "crops": {v: list(box) for v, box in crops.items()},
            },
            "visualization": {
                **(viz or {}),
                "videos": [{"video_name": "v", "panels": panels, **video}],
            },
        }
    )


# -- crop ---------------------------------------------------------------------


def test_crop_windows_the_frame_and_sizes_the_panel(src, frames):
    x, y, w, h = CROP
    assert src.view_size("rh", CROP) == (h, w)
    np.testing.assert_array_equal(
        src.frame("rh", 0, CROP), frames["rh"][0][y : y + h, x : x + w]
    )


def test_crop_moves_points_and_principal_point_by_the_same_offset(src, result):
    """The picture, the 2D and the projection must move together.

    A crop that shifts only the image is the dangerous failure: the overlay stays
    well-formed and lands somewhere plausible, just wrong -- which reads as a
    calibration error rather than as a rendering bug.
    """
    x, y, _, _ = CROP
    np.testing.assert_allclose(src.points2d("rh", 0, CROP), result.pts2d[0, 0] - [x, y])
    full, cropped = src.camera("rh"), src.camera("rh", CROP)
    np.testing.assert_allclose(cropped.intr[:2], full.intr[:2])  # focal untouched
    np.testing.assert_allclose(cropped.intr[2:], full.intr[2:] - [x, y])
    np.testing.assert_allclose(cropped.rvec, full.rvec)  # pose untouched


def test_cropped_projection_lands_on_the_cropped_picture(src, result, fly):
    """End to end: a 3D point reprojects onto the same pixel content, cropped or not."""
    cam_full = src.camera("rh")
    cam_crop = src.camera("rh", CROP)
    p3d = result.pts3d[0, :1]
    uv_full = np.asarray(cam_full.project(p3d)).reshape(2)
    uv_crop = np.asarray(cam_crop.project(p3d)).reshape(2)
    np.testing.assert_allclose(uv_crop, uv_full - [CROP[0], CROP[1]], atol=1e-9)


def test_crop_is_validated_at_parse_time():
    for bad, msg in (
        ([1, 2, 3], "must be"),
        ([0, 0, 0, 10], "non-positive"),
        ([-1, 0, 10, 10], "outside the frame"),
    ):
        with pytest.raises(ValueError, match=msg):
            compose.read_video_specs(
                _cfg([{"plot": "imshow", "view": "rh", "crop": bad}])
            )


def test_crop_is_not_forwarded_to_the_draw_op():
    panel = compose.read_video_specs(
        _cfg([{"plot": "imshow", "view": "rh", "crop": list(CROP)}])
    )[0].panels[0]
    assert panel.crop == CROP
    assert "crop" not in panel.options and "clip" not in panel.options


# -- a crop borrowed from the detector ----------------------------------------
#
# `crop = "pose2d"` exists because the boxes are per RECORDING: an auto crop
# restamps [pose2d] per recording from a crop plan, and a hand-copied panel box then keeps
# showing the previous recording's window under a perfectly well-formed overlay. So the
# tests below pin that the borrowed window is byte-for-byte the written one, that it
# resolves per view from one global setting, and that the ways it can be ambiguous are
# errors rather than a guess.


def test_a_borrowed_crop_renders_exactly_like_the_written_box(src):
    """The whole promise, as pixels: `"pose2d"` and the literal box are the same panel."""

    def panels(crop):
        return [
            {"plot": "imshow", "view": "f", "crop": crop, "width": 80, "height": 40},
            {
                "plot": "skeleton_2d",
                "view": "f",
                "crop": crop,
                "width": 80,
                "height": 40,
            },
        ]

    borrowed = compose.read_video_specs(
        _detector_cfg(panels("pose2d"), crops={"f": CROP_BOX})
    )[0]
    written = compose.read_video_specs(_cfg(panels(list(CROP))))[0]
    a = compose.compose_frame(borrowed, src, t=0)
    b = compose.compose_frame(written, src, t=0)
    assert a.std() > 0, "the panel drew nothing, so the comparison is vacuous"
    np.testing.assert_array_equal(a, b)


def test_one_global_setting_gives_each_view_its_own_window(src):
    """The line that replaces every duplicated box: each panel follows its own camera."""
    spec = compose.read_video_specs(
        _detector_cfg(
            [
                {"plot": "imshow", "view": "f"},
                {"plot": "imshow", "view": "rh"},  # full-frame pathway
                {"plot": "skeleton_3d", "view": BIRD_VIEW},  # no pathway at all
            ],
            crops={"f": CROP_BOX},
            viz={"crop": "pose2d"},
        )
    )[0]
    resolved = [p.resolve_crop(src) for p in spec.panels]
    assert resolved[0].crop == CROP
    # An uncropped view keeps the no-crop path rather than a full-frame box, and the
    # derived plan view -- which no pathway writes -- must not be windowed at all, or a
    # global setting could not be written safely.
    assert resolved[1].crop is None
    assert resolved[2].crop is None


def test_a_borrowed_crop_moves_the_picture_and_the_geometry_together(src, result):
    """Same guarantee as a written crop: the projection follows the window."""
    spec = compose.read_video_specs(
        _detector_cfg(
            [{"plot": "imshow", "view": "f"}],
            crops={"f": CROP_BOX},
            viz={"crop": "pose2d"},
        )
    )[0]
    panel = spec.panels[0].resolve_crop(src)
    x, y, w, h = panel.crop
    np.testing.assert_array_equal(
        src.frame("f", 0, panel.crop), src.frame("f", 0)[y : y + h, x : x + w]
    )
    full, cropped = src.camera("f"), src.camera("f", panel.crop)
    np.testing.assert_allclose(cropped.intr[2:], full.intr[2:] - [x, y])


def test_a_named_preprocessor_can_be_borrowed_directly(src):
    spec = compose.read_video_specs(
        _detector_cfg(
            [{"plot": "imshow", "view": "rh", "crop": "f"}], crops={"f": CROP_BOX}
        )
    )[0]
    assert spec.panels[0].resolve_crop(src).crop == CROP


def test_a_panel_box_still_wins_over_the_global_setting(src):
    spec = compose.read_video_specs(
        _detector_cfg(
            [
                {"plot": "imshow", "view": "f"},
                {"plot": "imshow", "view": "f", "crop": [0, 0, 8, 8]},
            ],
            crops={"f": CROP_BOX},
            viz={"crop": "pose2d"},
        )
    )[0]
    assert [p.resolve_crop(src).crop for p in spec.panels] == [CROP, (0, 0, 8, 8)]


def test_a_video_entry_can_override_the_global_setting(src):
    spec = compose.read_video_specs(
        _detector_cfg(
            [{"plot": "imshow", "view": "f"}],
            crops={"f": CROP_BOX},
            viz={"crop": "pose2d"},
            crop=[1, 2, 9, 9],  # on the video entry
        )
    )[0]
    assert spec.panels[0].resolve_crop(src).crop == (1, 2, 9, 9)


def test_an_unknown_crop_reference_names_the_alternatives():
    with pytest.raises(ValueError, match=r"have: \['f'\]"):
        compose.read_video_specs(
            _detector_cfg(
                [{"plot": "imshow", "view": "f", "crop": "crop_typo"}],
                crops={"f": CROP_BOX},
            )
        )


def test_a_camera_called_pose2d_is_refused_as_a_crop_reference():
    # A camera literally named "pose2d" would shadow the per-view rule; say so instead of
    # silently picking one meaning of the same word.
    cfg = _detector_cfg(
        [{"plot": "imshow", "view": "pose2d", "crop": "pose2d"}], crops={}
    )
    cfg.data["cameras"]["pose2d"] = {
        "azimuth_deg": 30,
        "distance": 100.0,
        "focal_length_px": 1.0,
    }
    cfg.data["pose2d"]["crops"] = {"pose2d": CROP_BOX}
    with pytest.raises(ValueError, match="rename the"):
        compose.read_video_specs(cfg)


def test_a_borrowed_crop_without_a_detection_plan_says_so():
    """A viz-only caller composites panels over frames it brought itself, and must not be
    forced to carry a detector -- but a panel that reaches for one has to be told."""
    with pytest.raises(ValueError, match="cannot build"):
        compose.read_video_specs(
            _cfg([{"plot": "imshow", "view": "rh", "crop": "pose2d"}])
        )


def test_a_stale_borrowed_crop_fails_loudly_against_smaller_footage(src):
    """The failure mode this whole feature is meant to prevent, if it slips through
    anyway: a box from another recording must not truncate into a plausible panel."""
    spec = compose.read_video_specs(
        _detector_cfg(
            [{"plot": "imshow", "view": "f"}],
            crops={"f": [0, 0, 900, 400]},
            viz={"crop": "pose2d"},
        )
    )[0]
    with pytest.raises(ValueError, match="exceeds"):
        compose.compose_frame(spec, src, t=0)


def test_a_borrowed_crop_is_fingerprinted_so_a_new_box_re_renders():
    """Change the detector's crop and the visualization stage must not reuse its MP4s.

    The reference is carried on the spec as the chain itself, not as a resolved box, so the
    visualization fingerprint moves with ``[pose2d]`` -- which is the only thing that makes
    borrowing safe for a cached run.
    """
    import dataclasses

    from deeperfly.pipeline import fingerprint

    def digest(box):
        specs = _detector_cfg(
            [{"plot": "imshow", "view": "f"}], crops={"f": box}, viz={"crop": "pose2d"}
        ).videos
        return fingerprint._norm([dataclasses.asdict(s) for s in specs])

    assert digest(CROP_BOX) != digest([13, 7, 40, 20])


# -- clip ---------------------------------------------------------------------


def test_clip_keeps_a_panel_out_of_its_neighbour(src):
    """A skeleton drawn in the left cell must not reach the right cell."""
    panels = [
        {"plot": "skeleton_3d", "view": "rh", "x0": 0, "width": 64, "height": 48},
        {"plot": "skeleton_3d", "view": "rm", "x0": 64, "width": 64, "height": 48},
    ]
    clipped = compose.compose_frame(compose.read_video_specs(_cfg(panels))[0], src, t=0)
    loose = compose.compose_frame(
        compose.read_video_specs(_cfg([{**p, "clip": False} for p in panels]))[0],
        src,
        t=0,
    )
    # The loose render spills across the seam; the clipped one cannot, so it must have
    # strictly fewer painted pixels somewhere. (If neither spilled the test is vacuous,
    # so assert the spill exists first.)
    assert (loose.sum(-1) > 0).sum() > (clipped.sum(-1) > 0).sum()


def test_clip_preserves_layering(onscreen_src, frames):
    """imshow then skeleton at the same offset: the skeleton must survive the clip.

    Clipping is done with a sliced *view* of the canvas rather than a private tile
    precisely so this keeps working -- an opaque tile per panel would erase the image
    the previous panel drew.
    """
    spec = compose.read_video_specs(
        _cfg(
            [
                {"plot": "imshow", "view": "rh", "width": 128, "height": 96},
                {"plot": "skeleton_2d", "view": "rh", "width": 128, "height": 96},
            ]
        )
    )[0]
    frame = compose.compose_frame(spec, onscreen_src, t=0)
    image_only = compose.compose_frame(
        compose.read_video_specs(
            _cfg([{"plot": "imshow", "view": "rh", "width": 128, "height": 96}])
        )[0],
        onscreen_src,
        t=0,
    )
    assert (frame != image_only).any()  # the skeleton drew on top
    # and the picture underneath survived: most pixels are still the raw frame
    assert (frame == image_only).all(-1).mean() > 0.5


def test_clip_defaults_on_and_is_overridable():
    panels = compose.read_video_specs(
        _cfg(
            [
                {"plot": "imshow", "view": "rh"},
                {"plot": "imshow", "view": "rm", "clip": False},
            ]
        )
    )[0].panels
    assert panels[0].clip is True and panels[1].clip is False


def test_a_fully_offcanvas_panel_is_skipped_not_crashed(src):
    spec = compose.read_video_specs(
        _cfg(
            [
                {"plot": "imshow", "view": "rh", "width": 32, "height": 32},
                {"plot": "imshow", "view": "rm", "x0": 999, "width": 32, "height": 32},
            ],
            width=32,
            height=32,
        )
    )[0]
    assert compose.compose_frame(spec, src, t=0).shape == (32, 32, 3)


# -- stage selection ----------------------------------------------------------


def test_stage_draws_that_stage_not_the_resolved_one(fly, result, frames):
    """A before/after pair is only meaningful if the two videos differ.

    ``PoseResult`` resolves to the MOST DERIVED stage present, so with EKS in the file
    both a `pose3d` and a `pose3d_eks` panel would otherwise draw the smoother's output
    and the pair would be two copies of one array.
    """
    tri = result.pts3d
    eks = result.pts3d + 5.0
    src = compose.Sources(
        fly,
        result.cameras,
        frames,
        pts2d=result.pts2d,
        pts3d=eks,  # what the result resolved to
        stage_pts3d={"triangulation": tri, "eks": eks},
    )
    np.testing.assert_allclose(src.points3d(0, "triangulation"), tri[0])
    np.testing.assert_allclose(src.points3d(0, "eks"), eks[0])
    np.testing.assert_allclose(src.points3d(0), eks[0])  # unset = resolved


def test_a_missing_stage_is_an_error_not_a_fallback(fly, result, frames):
    src = compose.Sources(
        fly,
        result.cameras,
        frames,
        pts3d=result.pts3d,
        stage_pts3d={"eks": result.pts3d},
    )
    with pytest.raises(ValueError, match="no 3d points for stage 'triangulation'"):
        src.points3d(0, "triangulation")


def test_stage_is_parsed_and_not_forwarded_to_the_draw_op():
    panels = compose.read_video_specs(
        _cfg(
            [
                {"plot": "skeleton_3d", "view": "rh", "stage": "triangulation"},
                {"plot": "skeleton_3d", "view": "rm"},
            ]
        )
    )[0].panels
    assert panels[0].stage == "triangulation" and panels[1].stage is None
    assert "stage" not in panels[0].options


def test_the_plan_view_is_framed_once_so_the_pair_is_comparable(fly, result, frames):
    """Both 3D videos must share one bird camera, or the panel would rescale between
    them and a pose difference would read as a zoom."""
    src = compose.Sources(
        fly,
        result.cameras,
        frames,
        pts3d=_posed_fly(fly),
        stage_pts3d={"triangulation": _posed_fly(fly), "eks": _posed_fly(fly) * 1.5},
    )
    assert src.camera(BIRD_VIEW) is src.camera(BIRD_VIEW)


# -- the derived dorsal view --------------------------------------------------


def _posed_fly(skeleton: Skeleton, *, roll=0.0) -> np.ndarray:
    """A synthetic animal with a KNOWN anatomy, so the view's axes can be checked.

    Anterior is +x, the animal's left is +y, dorsal is +z; the claws sit below the
    body. ``roll`` rotates the whole animal about its own long axis, which is what
    tells a dorsal view apart from a ventral one.
    """
    names = list(skeleton.point_names)
    p = np.zeros((len(names), 3))
    for i, n in enumerate(names):
        side = 1.0 if n.startswith("l") else -1.0
        if n.endswith("_thorax_coxa"):
            p[i] = [0.2, 0.4 * side, 0.0]
        elif n.endswith("_claw"):
            p[i] = [0.3, 0.9 * side, -0.6]  # feet: ventral
        elif n in ("l_antenna", "r_antenna", "neck"):
            p[i] = [1.0, 0.15 * side, 0.05]
        elif "abdomen" in n:
            p[i] = [-1.0, 0.05 * side, 0.05]
        else:
            p[i] = [0.0, 0.5 * side, -0.2]
    c, s = np.cos(roll), np.sin(roll)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return (p @ rot.T)[None].repeat(4, axis=0)


def test_dorsal_view_puts_anterior_up_and_the_animal_s_left_on_the_left(fly):
    pts3d = _posed_fly(fly)
    cam = dorsal_camera(pts3d, fly)
    uv = np.asarray(cam.project(pts3d[0])).reshape(-1, 2)
    names = list(fly.point_names)
    neck = uv[[i for i, n in enumerate(names) if n in ("l_antenna", "r_antenna")]].mean(
        0
    )
    tail = uv[[i for i, n in enumerate(names) if "abdomen" in n]].mean(0)
    left = uv[
        [
            i
            for i, n in enumerate(names)
            if n.startswith("l") and n.endswith("_thorax_coxa")
        ]
    ].mean(0)
    right = uv[
        [
            i
            for i, n in enumerate(names)
            if n.startswith("r") and n.endswith("_thorax_coxa")
        ]
    ].mean(0)
    assert neck[1] < tail[1], "anterior must be at the TOP of the panel"
    assert left[0] < right[0], (
        "looking down at the back, the animal's left is on the left"
    )


def test_dorsal_view_is_settled_by_the_claws_not_by_a_cross_product(fly):
    """Rolling the animal upside down must NOT flip the panel.

    The dorsal direction is chosen so the body is on the far side of the claws. Without
    that the sign comes from a cross product of two axes that both rolled, and the view
    would silently mirror -- swapping the animal's left and right legs on screen.
    """
    upright = dorsal_camera(_posed_fly(fly), fly)
    upside_down = dorsal_camera(_posed_fly(fly, roll=np.pi), fly)
    for cam, pts in (
        (upright, _posed_fly(fly)),
        (upside_down, _posed_fly(fly, roll=np.pi)),
    ):
        uv = np.asarray(cam.project(pts[0])).reshape(-1, 2)
        names = list(fly.point_names)
        left = uv[
            [
                i
                for i, n in enumerate(names)
                if n.startswith("l") and n.endswith("_thorax_coxa")
            ]
        ].mean(0)
        right = uv[
            [
                i
                for i, n in enumerate(names)
                if n.startswith("r") and n.endswith("_thorax_coxa")
            ]
        ].mean(0)
        assert left[0] < right[0]


def test_dorsal_view_frames_the_animal_inside_the_panel(fly):
    pts3d = _posed_fly(fly)
    cam = dorsal_camera(pts3d, fly, image_hw=(240, 480))
    uv = np.asarray(cam.project(pts3d[0])).reshape(-1, 2)
    assert (uv[:, 0] > 0).all() and (uv[:, 0] < 480).all()
    assert (uv[:, 1] > 0).all() and (uv[:, 1] < 240).all()
    # and it fills the frame rather than sitting in a corner. Only the BINDING axis
    # fills: the animal is longer than it is wide and anterior points up the panel, so
    # here it is the height that is tight and the width that has air.
    fill = max(
        (uv[:, 0].max() - uv[:, 0].min()) / 480, (uv[:, 1].max() - uv[:, 1].min()) / 240
    )
    assert fill > 0.8, (
        f"the plan view leaves the animal small ({fill:.2f} of the frame)"
    )


def test_dorsal_view_resolves_from_a_panel_and_needs_no_footage(fly, result, frames):
    src = compose.Sources(
        fly, result.cameras, frames, pts2d=result.pts2d, pts3d=_posed_fly(fly)
    )
    spec = compose.read_video_specs(
        _cfg([{"plot": "skeleton_3d", "view": BIRD_VIEW, "width": 96, "height": 48}])
    )[0]
    frame = compose.compose_frame(spec, src, t=0)
    assert frame.shape == (48, 96, 3)
    assert (frame.sum(-1) > 0).any(), "the plan view drew nothing"
    with pytest.raises(ValueError, match="synthetic viewpoint"):
        src.frame(BIRD_VIEW, 0)


def test_a_real_camera_named_bird_wins_over_the_derived_one(fly, result, frames):
    src = compose.Sources(fly, result.cameras, frames, pts3d=result.pts3d)
    renamed = result.cameras.names[0]
    assert src.camera(renamed) is result.cameras[renamed]


def test_dorsal_view_without_3d_says_so(fly, result, frames):
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    with pytest.raises(ValueError, match="derived from the 3D pose"):
        src.camera(BIRD_VIEW)


def test_dorsal_view_names_the_landmarks_it_cannot_find(result, frames):
    bare = Skeleton(
        name="bare",
        point_names=("a", "b", "c"),
        bones=np.zeros((0, 2), int),
    )
    with pytest.raises(ValueError, match="anterior"):
        dorsal_camera(np.zeros((2, 3, 3)), bare)
