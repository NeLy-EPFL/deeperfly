"""Tests for the OpenCV visualization backend and the panel compositor.

These check geometry/shape/compositing behavior (canvas sizes, that overlays
actually draw, depth ordering, config parsing), not pixel-exact rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly.viz import compose
from deeperfly.viz import opencv as cv
from deeperfly.viz._palette import point_colors_rgb

# `cameras`, `fly`, `result`, `rng` fixtures live in conftest.py.


@pytest.fixture
def frames(result, rng):
    """Synthetic per-view footage matching the result's 2D points."""
    v, t = result.pts2d.shape[:2]
    return {
        name: rng.integers(0, 255, size=(t, 96, 128, 3), dtype=np.uint8)
        for name in result.cameras.names[:v]
    }


# -- palette -----------------------------------------------------------------


def test_palette_matches_matplotlib(fly):
    mpl = pytest.importorskip("matplotlib.colors")
    rgb = point_colors_rgb(fly)
    for limb, hexc in fly.palette.items():
        idx = [i for i, lid in enumerate(fly.limb_id) if fly.limb_names[lid] == limb]
        for i in idx:
            np.testing.assert_allclose(rgb[i], mpl.to_rgba(hexc)[:3], atol=1e-6)


# -- primitives --------------------------------------------------------------


def test_new_canvas_background():
    assert (cv.new_canvas(4, 5, "black") == 0).all()
    assert (cv.new_canvas(4, 5, "white") == 255).all()
    assert (cv.new_canvas(4, 5, (10, 20, 30))[0, 0] == (10, 20, 30)).all()
    with pytest.raises(ValueError, match="background must be one of"):
        cv.new_canvas(4, 5, "navy")


def test_draw_image_blits_with_offset_and_clips():
    canvas = cv.new_canvas(10, 10, "black")
    img = np.full((4, 4, 3), 200, np.uint8)
    cv.draw_image(canvas, img, x0=8, y0=8)  # partly off-canvas -> clipped, no error
    assert (canvas[8:10, 8:10] == 200).all()
    assert (canvas[:8, :8] == 0).all()


def test_draw_image_non_uniform_scale_resizes_axes_independently():
    canvas = cv.new_canvas(50, 60, "black")
    img = np.full((40, 40, 3), 200, np.uint8)
    cv.draw_image(canvas, img, x0=0, y0=0, scale=(1.0, 0.5))  # 40 wide, 20 tall
    assert (canvas[:20, :40] == 200).all()
    assert (canvas[20:, :] == 0).all()
    assert (canvas[:, 40:] == 0).all()


def test_draw_skeleton_2d_draws_pixels(result, fly):
    canvas = cv.new_canvas(96, 128, "black")
    # rescale the synthetic 2D points into the canvas so they land in-frame
    pts = result.pts2d[0, 0].copy()
    pts -= np.nanmin(pts, axis=0)
    pts /= np.nanmax(pts, axis=0) + 1e-9
    pts *= [120, 90]
    out = cv.draw_skeleton_2d(canvas, pts, fly, point_radius=2, line_thickness=1)
    assert out.any(), "nothing was drawn"


def test_draw_skeleton_3d_depth_orders_and_drops_behind_camera(cameras, fly):
    cam = cameras["rf"]
    # one point far behind the camera -> must be dropped (not drawn / no crash)
    pts3d = np.zeros((fly.n_points, 3))
    pts3d[:] = np.linspace(-1, 1, fly.n_points)[:, None]
    canvas = cv.new_canvas(512, 1024, "black")
    out = cv.draw_skeleton_3d(canvas, pts3d, cam, fly)
    assert out.shape == (512, 1024, 3)


# -- compositor --------------------------------------------------------------


def _two_panel_config(plot):
    return {
        "pipeline": {
            "visualization": {
                "videos": [
                    {
                        "video_name": f"test_{plot}",
                        "panels": [
                            {"plot": "imshow", "view": "rh", "x0": 0, "y0": 0},
                            {"plot": plot, "view": "rh", "point_radius": 2},
                            {"plot": "imshow", "view": "rm", "x0": 128, "y0": 0},
                            {"plot": plot, "view": "rm", "x0": 128, "y0": 0},
                        ],
                    }
                ]
            }
        }
    }


def test_read_video_specs_parses_panels_and_options():
    specs = compose.read_video_specs(_two_panel_config("skeleton_2d"))
    assert len(specs) == 1
    spec = specs[0]
    assert spec.video_name == "test_skeleton_2d"
    assert [p.plot for p in spec.panels] == [
        "imshow",
        "skeleton_2d",
        "imshow",
        "skeleton_2d",
    ]
    assert spec.panels[1].options == {"point_radius": 2}  # extra keys forwarded
    assert spec.panels[2].x0 == 128


def test_op_kwargs_merge_three_levels():
    cfg = {
        "pipeline": {
            "visualization": {
                "kwargs": {  # 1. global, all videos
                    "skeleton_2d": {"line_thickness": 2, "point_radius": 1},
                    "skeleton_3d": {"line_thickness": 2},
                },
                "videos": [
                    {
                        "video_name": "v",
                        "kwargs": {"skeleton_2d": {"point_radius": 7}},  # 2. this video
                        "panels": [
                            {"plot": "skeleton_2d", "view": "rh"},
                            # 3. panel-level key overrides both broader levels
                            {"plot": "skeleton_2d", "view": "rm", "line_thickness": 9},
                            {"plot": "skeleton_3d", "view": "rf"},
                            {"plot": "imshow", "view": "rh"},  # unrelated op untouched
                        ],
                    }
                ],
            }
        }
    }
    panels = compose.read_video_specs(cfg)[0].panels
    # global line_thickness, video point_radius overrides global point_radius
    assert panels[0].options == {"line_thickness": 2, "point_radius": 7}
    # panel line_thickness wins over global; video point_radius still applies
    assert panels[1].options == {"line_thickness": 9, "point_radius": 7}
    # skeleton_3d only sees its own global entry, not skeleton_2d's
    assert panels[2].options == {"line_thickness": 2}
    # imshow has no kwargs at any level
    assert panels[3].options == {}


def test_op_kwargs_must_be_a_table():
    cfg = {
        "pipeline": {
            "visualization": {
                "kwargs": {"skeleton_2d": 2},
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [{"plot": "skeleton_2d", "view": "rh"}],
                    }
                ],
            }
        }
    }
    with pytest.raises(ValueError, match="must be a table"):
        compose.read_video_specs(cfg)


def test_scale_from_kwargs_lifted_to_panel_and_panel_key_wins():
    cfg = {
        "pipeline": {
            "visualization": {
                "kwargs": {"skeleton_2d": {"scale": 0.5, "line_thickness": 2}},
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [
                            {"plot": "skeleton_2d", "view": "rh"},  # scale from kwargs
                            {
                                "plot": "skeleton_2d",
                                "view": "rm",
                                "scale": 0.25,
                            },  # wins
                        ],
                    }
                ],
            }
        }
    }
    panels = compose.read_video_specs(cfg)[0].panels
    assert panels[0].scale == 0.5
    assert panels[1].scale == 0.25
    # scale is structural -- it must not leak into the forwarded draw-op kwargs
    assert "scale" not in panels[0].options
    assert panels[0].options == {"line_thickness": 2}


def test_width_height_resolve_scales_and_override_scale():
    cfg = {
        "pipeline": {
            "visualization": {
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [
                            {"plot": "imshow", "view": "rh", "scale": 0.5},  # uniform
                            {"plot": "imshow", "view": "rh", "width": 64, "height": 32},
                            {
                                "plot": "imshow",
                                "view": "rh",
                                "width": 64,
                            },  # aspect kept
                            {
                                "plot": "imshow",
                                "view": "rh",
                                "height": 48,
                            },  # aspect kept
                        ],
                    }
                ]
            }
        }
    }
    panels = compose.read_video_specs(cfg)[0].panels
    # a 96-tall, 128-wide source view
    assert panels[0].scales(96, 128) == (0.5, 0.5)
    assert panels[1].scales(96, 128) == (64 / 128, 32 / 96)  # exact box, non-uniform
    assert panels[2].scales(96, 128) == (0.5, 0.5)  # width 64/128 on both axes
    assert panels[3].scales(96, 128) == (0.5, 0.5)  # height 48/96 on both axes
    # width/height win over a co-specified scale
    assert panels[1].width == 64 and panels[1].height == 32


def test_width_height_set_panel_footprint_for_canvas_size(result, fly, frames):
    # frames are 96x128 per view; pin each tile to a fixed 100x60 box regardless
    cfg = {
        "pipeline": {
            "visualization": {
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [
                            {
                                "plot": "imshow",
                                "view": "rh",
                                "width": 100,
                                "height": 60,
                            },
                            {
                                "plot": "imshow",
                                "view": "rm",
                                "x0": 100,
                                "width": 100,
                                "height": 60,
                            },
                        ],
                    }
                ]
            }
        }
    }
    spec = compose.read_video_specs(cfg)[0]
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    # two 100x60 tiles side by side -> 200 wide x 60 tall, independent of frame size
    assert compose.canvas_size(spec, src) == (60, 200)
    frame = compose.compose_frame(spec, src, t=0)
    assert frame.shape == (60, 200, 3)


def test_width_height_via_global_kwargs():
    cfg = {
        "pipeline": {
            "visualization": {
                "kwargs": {"imshow": {"width": 80, "height": 80}},
                "videos": [
                    {"video_name": "v", "panels": [{"plot": "imshow", "view": "rh"}]}
                ],
            }
        }
    }
    panel = compose.read_video_specs(cfg)[0].panels[0]
    assert panel.width == 80 and panel.height == 80
    assert "width" not in panel.options and "height" not in panel.options


def test_background_two_levels_global_and_panel():
    cfg = {
        "pipeline": {
            "visualization": {
                "background": "white",  # global canvas fill
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [
                            {"plot": "skeleton_3d", "view": "rh"},
                            {
                                "plot": "skeleton_3d",
                                "view": "rm",
                                "background": "black",
                            },
                        ],
                    }
                ],
            }
        }
    }
    spec = compose.read_video_specs(cfg)[0]
    assert spec.background == "white"
    assert spec.panels[0].background is None  # inherits the canvas fill
    assert spec.panels[1].background == "black"  # per-panel override
    # background is not forwarded to the draw op
    assert "background" not in spec.panels[1].options


def test_background_defaults_to_black():
    cfg = {
        "pipeline": {
            "visualization": {
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [{"plot": "skeleton_3d", "view": "rh"}],
                    }
                ]
            }
        }
    }
    assert compose.read_video_specs(cfg)[0].background == "black"


def test_fill_region_paints_and_clips():
    canvas = cv.new_canvas(10, 10, "black")
    cv.fill_region(canvas, x0=8, y0=8, width=5, height=5, background="white")
    assert (canvas[8:10, 8:10] == 255).all()  # clipped to the canvas
    assert (canvas[:8, :8] == 0).all()


def test_panel_background_fills_footprint_before_op(result, fly, frames):
    cfg = {
        "pipeline": {
            "visualization": {
                "background": "black",
                "videos": [
                    {
                        "video_name": "v",
                        "panels": [
                            # skeleton-only tile on its own white backdrop
                            {
                                "plot": "skeleton_3d",
                                "view": "rh",
                                "background": "white",
                            },
                        ],
                    }
                ],
            }
        }
    }
    spec = compose.read_video_specs(cfg)[0]
    src = compose.Sources(fly, result.cameras, frames, pts3d=result.pts3d)
    frame = compose.compose_frame(spec, src, t=0)
    # the panel footprint (rh view size) was painted white over the black canvas
    h, w = src.view_size("rh")
    assert (frame[:h, :w] == 255).any()


def test_canvas_size_inferred_from_panel_bbox(result, fly, frames):
    spec = compose.read_video_specs(_two_panel_config("skeleton_2d"))[0]
    src = compose.Sources(
        skeleton=fly, camera_group=result.cameras, frames=frames, pts2d=result.pts2d
    )
    # two 128-wide / 96-tall tiles side by side -> 256 x 96
    assert compose.canvas_size(spec, src) == (96, 256)


def test_scale_shrinks_panel_footprint_and_image(result, fly, frames):
    # draw_image with scale resizes the blitted image
    canvas = cv.new_canvas(50, 60, "black")
    cv.draw_image(canvas, np.full((40, 40, 3), 200, np.uint8), x0=0, y0=0, scale=0.5)
    assert (canvas[:20, :20] == 200).all()  # 40*0.5 = 20px tile
    assert (canvas[20:, :] == 0).all()

    # a 0.5-scaled panel halves its footprint in the inferred canvas size
    cfg = _two_panel_config("skeleton_2d")
    for panel in cfg["pipeline"]["visualization"]["videos"][0]["panels"]:
        panel["scale"] = 0.5
        panel["x0"] = panel.get("x0", 0) // 2  # keep the two 64-wide tiles adjacent
    spec = compose.read_video_specs(cfg)[0]
    assert spec.panels[0].scale == 0.5
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    # two 64x48 tiles (128x96 frames at 0.5) side by side -> 128 x 48
    assert compose.canvas_size(spec, src) == (48, 128)


def test_explicit_canvas_size_overrides_inference(result, fly, frames):
    spec = compose.read_video_specs(_two_panel_config("skeleton_2d"))[0]
    spec.width, spec.height = 300, 100
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    assert compose.canvas_size(spec, src) == (100, 300)


def test_compose_frame_overlays_skeleton_on_image(result, fly, frames):
    spec = compose.read_video_specs(_two_panel_config("skeleton_3d"))[0]
    src = compose.Sources(
        fly, result.cameras, frames, pts2d=result.pts2d, pts3d=result.pts3d
    )
    frame = compose.compose_frame(spec, src, t=0)
    assert frame.shape == (96, 256, 3)
    # the imshow layer filled the canvas with the (nonzero) synthetic frames
    assert frame.any()


def test_render_video_stacks_all_frames(result, fly, frames):
    spec = compose.read_video_specs(_two_panel_config("skeleton_3d"))[0]
    src = compose.Sources(
        fly, result.cameras, frames, pts2d=result.pts2d, pts3d=result.pts3d
    )
    out = compose.render_video(spec, src)
    assert out.shape == (result.pts3d.shape[0], 96, 256, 3)
    assert out.dtype == np.uint8


def test_unknown_plot_op_raises(result, fly, frames):
    cfg = _two_panel_config("skeleton_2d")
    cfg["pipeline"]["visualization"]["videos"][0]["panels"][1]["plot"] = "bogus"
    spec = compose.read_video_specs(cfg)[0]
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    with pytest.raises(ValueError, match="unknown plot op 'bogus'"):
        compose.compose_frame(spec, src, t=0)


def test_packaged_config_videos_parse():
    """The shipped default config's [[pipeline.visualization.videos]] parse into valid specs."""
    from importlib.resources import files

    cfg = files("deeperfly.data") / "default_config.toml"
    specs = compose.read_video_specs(cfg)
    assert {s.video_name for s in specs} == {"pose2d", "pose3d"}
    assert all(p.plot in compose.OPS for s in specs for p in s.panels)
    # the global [pipeline.visualization.kwargs] sets line_thickness=2 on every skeleton panel
    skel = [p for s in specs for p in s.panels if p.plot.startswith("skeleton")]
    assert skel and all(p.options.get("line_thickness") == 2 for p in skel)
    # scale moved to global [pipeline.visualization.kwargs]; every panel resolves to 0.5
    assert all(p.scale == 0.5 for s in specs for p in s.panels)
    # and the default canvas background is black
    assert all(s.background == "black" for s in specs)
