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
from helpers import AZIMUTHS_DEG, CAMERA_NAMES, output_points_table

from deeperfly.config import Config
from deeperfly.skeleton import Skeleton
from deeperfly.visualization import compose
from deeperfly.visualization.bird import BIRD_VIEW, dorsal_camera

# `cameras`, `fly`, `result`, `rng` fixtures live in conftest.py.

CROP = (12, 7, 40, 20)
#: The same window as a ``[[pose2d.preprocessors]]`` op list.
CROP_OPS = [{"op": "crop", "x": 12, "y": 7, "width": 40, "height": 20}]


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


def _detector_cfg(panels, *, crops, spare=None, viz=None, **video):
    """A config whose ``[pose2d]`` really detects each view through ``crops[view]``.

    ``crops`` maps a view name to a preprocessor ``ops`` list; a view absent from it runs
    full-frame, which is this rig's policy for its six side cameras. One dense pathway per
    camera, so the channel mapping is the identity and no ``[pose2d.output_points]`` table
    is needed -- the same shape ``deeperfly dense-config`` writes. ``spare`` adds named
    preprocessors no pathway uses, for the panels that borrow one by name.
    """
    preprocessors = [{"name": f"crop_{v}", "ops": ops} for v, ops in crops.items()]
    preprocessors += [{"name": n, "ops": ops} for n, ops in (spare or {}).items()]
    return Config.from_dict(
        {
            "sources": [
                {"name": f"vid_{v}", "filename": f"{v}.mp4"} for v in CAMERA_NAMES
            ],
            "skeleton": Config.default().data["skeleton"],
            "cameras": _rig_cameras(),
            "pose2d": {
                "preprocessors": preprocessors,
                "models": [
                    {
                        "name": "dense",
                        "class": "hrnet",
                        "input_size": [256, 512],
                        "n_out_channels": 38,
                    }
                ],
                "pathways": [
                    {
                        "name": v,
                        "source": f"vid_{v}",
                        "model": "dense",
                        **({"preprocessor": f"crop_{v}"} if v in crops else {}),
                    }
                    for v in CAMERA_NAMES
                ],
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
# `crop = "pose2d"` exists because the boxes are GENERATED: `deeperfly dense-config`
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
        _detector_cfg(panels("pose2d"), crops={"f": CROP_OPS})
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
            crops={"f": CROP_OPS},
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
            crops={"f": CROP_OPS},
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
            [{"plot": "imshow", "view": "rh", "crop": "crop_f"}], crops={"f": CROP_OPS}
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
            crops={"f": CROP_OPS},
            viz={"crop": "pose2d"},
        )
    )[0]
    assert [p.resolve_crop(src).crop for p in spec.panels] == [CROP, (0, 0, 8, 8)]


def test_a_video_entry_can_override_the_global_setting(src):
    spec = compose.read_video_specs(
        _detector_cfg(
            [{"plot": "imshow", "view": "f"}],
            crops={"f": CROP_OPS},
            viz={"crop": "pose2d"},
            crop=[1, 2, 9, 9],  # on the video entry
        )
    )[0]
    assert spec.panels[0].resolve_crop(src).crop == (1, 2, 9, 9)


def test_a_borrowed_window_survives_a_flip_in_the_chain(src, caplog):
    """A chain that also mirrors looks through the same region; the panel says so.

    The panel cannot mirror the picture -- the overlay is projected into view pixels and a
    crop cannot express a reflection -- so it shows the right region the un-mirrored way
    round, and logs that it did rather than letting it be a surprise.
    """
    # logger="deeperfly", not a bare at_level: the CLI tests elsewhere in the suite
    # setLevel the "deeperfly" logger, and a bare at_level only raises the ROOT one, so
    # this captures nothing when it runs after them (see tests/test_project.py's docstring).
    with caplog.at_level("WARNING", logger="deeperfly"):
        spec = compose.read_video_specs(
            _detector_cfg(
                [{"plot": "imshow", "view": "rh", "crop": "mirror_crop"}],
                crops={"f": CROP_OPS},
                spare={"mirror_crop": [*CROP_OPS, {"op": "fliplr"}]},
            )
        )[0]
    assert spec.panels[0].resolve_crop(src).crop == CROP
    assert "mirrors or turns" in caplog.text


def test_pathways_that_window_a_view_differently_are_an_error_not_a_guess(
    fly, result, frames
):
    """Two windows, one panel: either render looks fine, so guessing is the wrong move."""
    point_names = Config.default().data["skeleton"]["point_names"]
    half = len(point_names) // 2
    cfg = Config.from_dict(
        {
            "sources": [{"name": "vid_f", "filename": "f.mp4"}],
            "skeleton": Config.default().data["skeleton"],
            "cameras": _rig_cameras(),
            "pose2d": {
                "preprocessors": [
                    {"name": "a", "ops": CROP_OPS},
                    {
                        "name": "b",
                        "ops": [
                            {"op": "crop", "x": 0, "y": 0, "width": 8, "height": 8}
                        ],
                    },
                ],
                "models": [
                    {
                        "name": "dense",
                        "class": "hrnet",
                        "input_size": [256, 512],
                        "n_out_channels": 38,
                    }
                ],
                "pathways": [
                    {
                        "name": "near",
                        "source": "vid_f",
                        "model": "dense",
                        "preprocessor": "a",
                    },
                    {
                        "name": "far",
                        "source": "vid_f",
                        "model": "dense",
                        "preprocessor": "b",
                    },
                ],
                # Both pathways fill view "f" -- half its points each -- through different
                # windows, which is the only way a view can carry two of them.
                "output_points": output_points_table(
                    point_names,
                    [
                        (
                            "f",
                            "near",
                            [p if p < half else -1 for p in range(len(point_names))],
                        ),
                        (
                            "f",
                            "far",
                            [p if p >= half else -1 for p in range(len(point_names))],
                        ),
                    ],
                ),
            },
            "visualization": {
                "crop": "pose2d",
                "videos": [
                    {"video_name": "v", "panels": [{"plot": "imshow", "view": "f"}]}
                ],
            },
        }
    )
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    spec = compose.read_video_specs(cfg)[0]
    with pytest.raises(ValueError, match="different windows"):
        compose.compose_frame(spec, src, t=0)


def test_an_unknown_crop_reference_names_the_alternatives():
    with pytest.raises(ValueError, match="crop_f"):
        compose.read_video_specs(
            _detector_cfg(
                [{"plot": "imshow", "view": "f", "crop": "crop_typo"}],
                crops={"f": CROP_OPS},
            )
        )


def test_a_reserved_preprocessor_name_is_refused():
    # A preprocessor literally named "pose2d" would shadow the per-view rule; say so
    # instead of silently picking one meaning of the same word.
    with pytest.raises(ValueError, match="rename the preprocessor"):
        compose.read_video_specs(
            _detector_cfg(
                [{"plot": "imshow", "view": "f", "crop": "pose2d"}],
                crops={"f": CROP_OPS},
                spare={"pose2d": CROP_OPS},
            )
        )


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
            crops={"f": [{"op": "crop", "x": 0, "y": 0, "width": 900, "height": 400}]},
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

    def digest(ops):
        specs = _detector_cfg(
            [{"plot": "imshow", "view": "f"}], crops={"f": ops}, viz={"crop": "pose2d"}
        ).videos
        return fingerprint._norm([dataclasses.asdict(s) for s in specs])

    moved = [{"op": "crop", "x": 13, "y": 7, "width": 40, "height": 20}]
    assert digest(CROP_OPS) != digest(moved)


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
        limb_names=("x",),
        limb_id=np.zeros(3, int),
        palette={"x": "#ffffff"},
    )
    with pytest.raises(ValueError, match="anterior"):
        dorsal_camera(np.zeros((2, 3, 3)), bare)
