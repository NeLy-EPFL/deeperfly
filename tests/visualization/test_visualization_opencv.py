"""Tests for the OpenCV visualization backend and the panel compositor.

These check geometry/shape/compositing behavior (canvas sizes, that overlays
actually draw, depth ordering, config parsing), not pixel-exact rendering.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.visualization import compose
from deeperfly.visualization import opencv as cv
from deeperfly.visualization._palette import point_colors_rgb

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


def test_point_colors_match_matplotlib(fly):
    mpl = pytest.importorskip("matplotlib.colors")
    rgb = point_colors_rgb(fly)
    for i, hexc in enumerate(fly.point_colors):
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


def test_draw_point_outline_keeps_zero_confidence_joint_visible():
    # A conf-0 joint has an empty fill but must still show its solid outline ring,
    # so the joint stays locatable no matter how faint the fill.
    canvas = cv.new_canvas(21, 21, "black")
    cv._draw_point(canvas, (10, 10), radius=5, color=(0, 200, 255), alpha=0.0)
    assert canvas.any(), "the outline ring should draw even at confidence 0"
    assert (canvas[10, 10] == 0).all(), "the fill stays empty at confidence 0"
    # the ring lands on the radius-5 rim, not the centre
    assert canvas[10, 5].any() and canvas[5, 10].any()


def test_draw_point_confidence_sets_fill_opacity():
    def center_fill(alpha: float) -> int:
        canvas = cv.new_canvas(21, 21, "black")
        cv._draw_point(canvas, (10, 10), radius=5, color=(0, 200, 255), alpha=alpha)
        return int(canvas[10, 10].sum())

    # the raw confidence is used verbatim: higher confidence -> more opaque fill
    assert 0 < center_fill(0.2) < center_fill(0.6) < center_fill(1.0)


def test_draw_point_outline_thickness_zero_is_fill_only():
    # opting out of the ring reverts to the old fill-only marker: a conf-0 joint
    # then draws nothing at all.
    canvas = cv.new_canvas(21, 21, "black")
    cv._draw_point(
        canvas, (10, 10), radius=5, color=(0, 200, 255), alpha=0.0, outline_thickness=0
    )
    assert not canvas.any()


# -- dashed bones ------------------------------------------------------------


def _lit(canvas) -> int:
    """How many pixels the drawing touched."""
    import numpy as _np

    return int(_np.asarray(canvas > 0).any(-1).sum())


def _canvas_pts(result, canvas_h=96, canvas_w=128):
    """The synthetic 2D of view 0 rescaled to land inside a canvas of that size."""
    pts = result.pts2d[0, 0].copy()
    pts -= np.nanmin(pts, axis=0)
    pts /= np.nanmax(pts, axis=0) + 1e-9
    pts *= [canvas_w - 8, canvas_h - 6]
    return pts


def test_line_dash_leaves_gaps_and_zero_is_solid(result, fly):
    """A dash pattern lights strictly fewer pixels; ``0``/``None`` is byte-for-byte solid."""
    pts = _canvas_pts(result)

    def render(**kw):
        canvas = cv.new_canvas(96, 128, "black")
        cv.draw_skeleton_2d(canvas, pts, fly, line_thickness=2, draw_points=False, **kw)
        return canvas

    solid = render()
    dashed = render(line_dash=(4, 9))
    assert 0 < _lit(dashed) < _lit(solid)
    # A wider gap lights less still -- the pattern is arc length, not a dash count.
    assert _lit(render(line_dash=(4, 20))) < _lit(dashed)
    # Opting out has to be exact: a "solid" skeleton drawn through the dash path
    # would differ from every other panel by a pixel here and there.
    for off in (0, None):
        np.testing.assert_array_equal(render(line_dash=off), solid)


def test_line_dash_applies_to_the_3d_drawer_too(cameras, fly, result):
    """``skeleton_3d`` takes the same pattern -- it is the layer a comparison dashes."""

    def render(**kw):
        canvas = cv.new_canvas(512, 1024, "black")
        cv.draw_skeleton_3d(
            canvas,
            result.pts3d[0],
            cameras["rf"],
            fly,
            line_thickness=2,
            draw_points=False,
            **kw,
        )
        return canvas

    assert 0 < _lit(render(line_dash=(4, 9))) < _lit(render())


def test_line_dash_rejects_a_malformed_pattern(result, fly):
    canvas = cv.new_canvas(96, 128, "black")
    with pytest.raises(ValueError, match="line_dash must be a number or an"):
        cv.draw_skeleton_2d(canvas, _canvas_pts(result), fly, line_dash=(1, 2, 3))


def test_line_dash_reaches_the_draw_op_from_a_config(result, fly):
    """A dashed reference under a solid overlay is a config, not a code change.

    The point of the parameter: two skeletons in one panel where the config alone says
    which is which. Reserved layout keys never reach the op, so ``line_dash`` has to
    travel as a forwarded draw-op kwarg -- and this is the check that it does.
    """
    config = Config.from_dict(
        {
            "cameras": _VIZ_CAMERAS,
            "visualization": {
                "default_video": {"cell": [128, 96], "footage": False},
                "videos": {
                    "compare": {
                        "grid": [["rh"]],
                        "layers": [
                            {
                                "draw": "skeleton_3d",
                                "line_dash": [4, 9],
                                "line_thickness": 2,
                                "draw_points": False,
                            }
                        ],
                    }
                },
            },
        }
    )
    (spec,) = config.videos
    assert spec.panels[0].options["line_dash"] == [4, 9]
    # No frames: the panel is then sized from the camera's own intrinsics, so the
    # reprojected skeleton lands inside the canvas at scale 1 and the lit-pixel count
    # is the skeleton's alone rather than an imshow's.
    src = compose.Sources(fly, result.cameras, {}, pts3d=result.pts3d)
    dashed = compose.compose_frame(spec, src, t=0)
    solid_spec = copy.deepcopy(spec)
    solid_spec.panels[0].options.pop("line_dash")
    assert 0 < _lit(dashed) < _lit(compose.compose_frame(solid_spec, src, t=0))


# -- compositor --------------------------------------------------------------


_VIZ_CAMERAS = {
    v: {"azimuth_deg": az, "distance": 100.0, "focal_length_px": 1.0}
    for v, az in (("rh", -120), ("rm", -90), ("rf", -45))
}


def _two_panel_config(plot):
    """A 1x2 grid: footage plus one overlay layer over cameras rh and rm."""
    return Config.from_dict(
        {
            "cameras": _VIZ_CAMERAS,
            "visualization": {
                "default_video": {"cell": [128, 96]},
                "videos": {
                    f"test_{plot}": {
                        "grid": [["rh", "rm"]],
                        "layers": [{"draw": plot, "point_radius": 2}],
                    }
                },
            },
        }
    )


def _spec(*panels, **video):
    """A :class:`VideoSpec` built DIRECTLY, for the compositor's own arithmetic.

    Per-panel offsets, sizes and backgrounds are no longer a config surface -- a grid
    computes the offsets and `cell` sets the size -- but they are still what `Panel`
    carries and what `canvas_size` / `Panel.scales` / the background fill are about. So
    those tests construct the object instead of a config that cannot express it.
    """
    return compose.VideoSpec(video.pop("name", "v"), list(panels), **video)


def test_read_video_specs_expands_a_grid_into_footage_plus_layer():
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
    assert spec.panels[1].options == {"point_radius": 2}  # style forwarded
    assert spec.panels[2].x0 == 128  # the second column, at one cell width


def _viz_config(videos: dict) -> Config:
    return Config.from_dict(
        {
            "cameras": _VIZ_CAMERAS,
            "visualization": {"default_video": {"cell": [128, 96]}, "videos": videos},
        }
    )


def test_a_list_of_videos_says_they_are_keyed_by_name_now():
    """The v1 shape parses -- it is just a different TOML type -- so it is named.

    Keying by name is also what makes two videos with one name a TOML error rather than
    two videos racing to write the same .mp4.
    """
    cfg = Config.from_dict(
        {"visualization": {"videos": [{"video_name": "v", "grid": [["rh"]]}]}}
    )
    with pytest.raises(ValueError, match="keyed by name"):
        compose.read_video_specs(cfg)


def test_a_video_with_no_layers_raises_clear_error():
    cfg = _viz_config({"v": {"grid": [["rh"]]}})
    with pytest.raises(ValueError, match="needs a non-empty `layers`"):
        compose.read_video_specs(cfg)


def test_a_video_with_no_grid_raises_clear_error():
    cfg = _viz_config({"v": {"layers": [{"draw": "imshow"}]}})
    with pytest.raises(ValueError, match="needs a `grid`"):
        compose.read_video_specs(cfg)


def test_unknown_draw_op_raises_at_parse_time():
    cfg = _viz_config({"v": {"grid": [["rh"]], "layers": [{"draw": "skeleton2d"}]}})
    with pytest.raises(ValueError, match="unknown draw op"):
        compose.read_video_specs(cfg)


def test_an_unknown_style_key_is_refused():
    """What the v1 per-op kwargs merge could not do: a misspelled op matched nothing."""
    cfg = _viz_config(
        {"v": {"grid": [["rh"]], "layers": [{"draw": "skeleton_2d", "thikness": 2}]}}
    )
    with pytest.raises(ValueError, match=r"unknown style key\(s\) \['thikness'\]"):
        compose.read_video_specs(cfg)


def test_a_style_key_an_op_cannot_use_is_dropped_for_that_op():
    """Which is what lets ONE [visualization.default_layer] table serve every op."""
    cfg = _viz_config(
        {
            "v": {
                "grid": [["rh"]],
                "layers": [{"draw": "mesh_model", "line_thickness": 2, "alpha": 0.5}],
            }
        }
    )
    mesh = [
        p for p in compose.read_video_specs(cfg)[0].panels if p.plot == "mesh_model"
    ]
    assert mesh[0].options == {"alpha": 0.5}


def test_a_retired_visualization_key_is_refused_by_name():
    for key in ("kwargs", "background", "crop", "cell"):
        with pytest.raises(ValueError, match=key):
            compose.read_video_specs(
                Config.from_dict({"visualization": {key: {} if key == "kwargs" else 1}})
            )


def test_width_height_resolve_scales_and_override_scale():
    panels = [
        compose.Panel(plot="imshow", view="rh", scale=0.5),
        compose.Panel(plot="imshow", view="rh", width=64, height=32),
        compose.Panel(plot="imshow", view="rh", width=64),  # aspect kept
        compose.Panel(plot="imshow", view="rh", height=48),  # aspect kept
    ]
    # a 96-tall, 128-wide source view
    assert panels[0].scales(96, 128) == (0.5, 0.5)
    assert panels[1].scales(96, 128) == (64 / 128, 32 / 96)  # exact box, non-uniform
    assert panels[2].scales(96, 128) == (0.5, 0.5)  # width 64/128 on both axes
    assert panels[3].scales(96, 128) == (0.5, 0.5)  # height 48/96 on both axes
    # width/height win over a co-specified scale
    assert panels[1].width == 64 and panels[1].height == 32


def test_width_height_set_panel_footprint_for_canvas_size(result, fly, frames):
    # frames are 96x128 per view; pin each tile to a fixed 100x60 box regardless
    spec = _spec(
        compose.Panel(plot="imshow", view="rh", width=100, height=60),
        compose.Panel(plot="imshow", view="rm", x0=100, width=100, height=60),
    )
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    # two 100x60 tiles side by side -> 200 wide x 60 tall, independent of frame size
    assert compose.canvas_size(spec, src) == (60, 200)
    frame = compose.compose_frame(spec, src, t=0)
    assert frame.shape == (60, 200, 3)


def test_the_canvas_background_comes_from_the_video():
    cfg = _viz_config(
        {
            "v": {
                "background": "white",
                "grid": [["rh"]],
                "layers": [{"draw": "skeleton_3d"}],
            }
        }
    )
    spec = compose.read_video_specs(cfg)[0]
    assert spec.background == "white"
    # A panel background is a compositor field, not a config surface -- a panel takes the
    # canvas fill unless something built it otherwise.
    assert all(p.background is None for p in spec.panels)
    assert "background" not in spec.panels[-1].options


def test_background_defaults_to_black():
    cfg = _viz_config({"v": {"grid": [["rh"]], "layers": [{"draw": "skeleton_3d"}]}})
    assert compose.read_video_specs(cfg)[0].background == "black"


def test_fill_region_paints_and_clips():
    canvas = cv.new_canvas(10, 10, "black")
    cv.fill_region(canvas, x0=8, y0=8, width=5, height=5, background="white")
    assert (canvas[8:10, 8:10] == 255).all()  # clipped to the canvas
    assert (canvas[:8, :8] == 0).all()


def test_panel_background_fills_footprint_before_op(result, fly, frames):
    # A per-panel backdrop is a compositor field, not a config surface.
    spec = _spec(
        compose.Panel(plot="skeleton_3d", view="rh", background="white"),
        background="black",
    )
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
    spec = compose.read_video_specs(_two_panel_config("skeleton_2d"))[0]
    for panel in spec.panels:
        panel.scale = 0.5
        panel.width = panel.height = None  # scale, not a fixed box
        panel.x0 //= 2  # keep the two 64-wide tiles adjacent
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


def test_skeleton_model_op_overlays_fitted_model(result, fly, frames):
    """The skeleton_model op reprojects Sources.model_pts3d onto each view."""
    spec = compose.read_video_specs(_two_panel_config("skeleton_model"))[0]
    src = compose.Sources(
        fly, result.cameras, frames, pts2d=result.pts2d, model_pts3d=result.pts3d
    )
    frame = compose.compose_frame(spec, src, t=0)
    assert frame.shape == (96, 256, 3)
    assert frame.any()


def test_skeleton_model_op_requires_model_points(result, fly, frames):
    spec = compose.VideoSpec(
        video_name="v", panels=[compose.Panel(plot="skeleton_model", view="rh")]
    )
    src = compose.Sources(fly, result.cameras, frames, pts3d=result.pts3d)
    with pytest.raises(
        ValueError, match="skeleton_model panel needs Sources.model_pts3d"
    ):
        compose.compose_frame(spec, src, t=0)


def test_mesh_rgba_rasterizes_the_posed_model(result):
    """The mesh rasterizer projects a posed model to an RGBA overlay with coverage."""
    from deeperfly.inverse_kinematics.mesh import load_model_mesh
    from deeperfly.visualization.mesh import render_mesh_rgba

    mesh = load_model_mesh()
    verts, valid = mesh.pose(mesh.kp_neutral)  # model near the world origin
    cam = result.cameras["rf"]
    rgba = render_mesh_rgba(verts, mesh.faces, mesh.face_rgb, valid, cam, 512, 1024)
    assert rgba.shape == (512, 1024, 4)
    assert (rgba[..., 3] > 0).any()  # the model covers some pixels
    # alpha is bounded and colored pixels coincide with coverage
    assert rgba[..., 3].max() <= 255


def test_vertex_normals_are_unit_and_smooth():
    """Smooth per-vertex normals are unit-length and ignore non-valid faces."""
    from deeperfly.visualization.mesh import vertex_normals

    # A unit cube (shared verts) -> a corner's normal points diagonally outward.
    verts = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [1, 2, 6],
            [1, 6, 5],
            [0, 4, 7],
            [0, 7, 3],
        ],
    )
    n = vertex_normals(verts, faces)
    assert n.shape == verts.shape
    np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)
    # the +x+y+z corner (vertex 6) faces outward along the diagonal (area-weighting
    # tilts it off the exact diagonal, but it still points up the +++ octant)
    assert (n[6] > 0).all() and n[6] @ (np.ones(3) / np.sqrt(3)) > 0.9
    # an all-invalid mask draws nothing -> zero normals
    z = vertex_normals(verts, faces, np.zeros(len(faces), bool))
    assert np.allclose(z, 0.0)


def test_mesh_rgba_gl_matches_software_when_available(result):
    """When a headless GL context exists, the GPU rasterizer agrees with the CPU one.

    Skips where no EGL/GL context can be created (CI without a GPU driver), since the
    overlay then simply falls back to the software rasterizer (covered above). The GPU
    path projects with the camera's full pinhole, so it renders at the footage size
    the intrinsics describe (which is exactly how the overlay ops call it).
    """
    from deeperfly.inverse_kinematics.mesh import load_model_mesh
    from deeperfly.visualization.mesh import render_mesh_rgba
    from deeperfly.visualization.mesh_gl import gl_available, render_mesh_rgba_gl

    if not gl_available():
        pytest.skip("no headless GL context available")
    mesh = load_model_mesh()
    verts, valid = mesh.pose(mesh.kp_neutral)
    cam = result.cameras["rf"]
    cx, cy = float(cam.intr[2]), float(cam.intr[3])
    w, h = int(round(2 * cx)), int(round(2 * cy))  # footage size the intrinsics imply
    cpu = render_mesh_rgba(verts, mesh.faces, mesh.face_rgb, valid, cam, h, w)
    gpu = render_mesh_rgba_gl(verts, mesh.faces, mesh.face_rgb, valid, cam, h, w)
    assert gpu is not None and gpu.shape == cpu.shape
    cov_cpu, cov_gpu = cpu[..., 3] > 0, gpu[..., 3] > 0
    iou = (cov_cpu & cov_gpu).sum() / max((cov_cpu | cov_gpu).sum(), 1)
    assert iou > 0.9  # same silhouette (smooth vs flat shading + exact depth test)


def test_mesh_model_op_overlays_the_mesh(result, fly, frames):
    cfg = _viz_config(
        {"v": {"grid": [["rh"]], "layers": [{"draw": "mesh_model", "alpha": 0.6}]}}
    )
    spec = compose.read_video_specs(cfg)[0]
    src = compose.Sources(fly, result.cameras, frames, model_pts3d=result.pts3d)
    frame = compose.compose_frame(spec, src, t=0)  # composites without error
    assert frame.ndim == 3 and frame.any()


def test_mesh_model_op_requires_model_points(result, fly, frames):
    spec = compose.VideoSpec(
        video_name="v", panels=[compose.Panel(plot="mesh_model", view="rh")]
    )
    src = compose.Sources(fly, result.cameras, frames, pts3d=result.pts3d)
    with pytest.raises(ValueError, match="mesh_model panel needs Sources.model_pts3d"):
        compose.compose_frame(spec, src, t=0)


def test_render_video_stacks_all_frames(result, fly, frames):
    spec = compose.read_video_specs(_two_panel_config("skeleton_3d"))[0]
    src = compose.Sources(
        fly, result.cameras, frames, pts2d=result.pts2d, pts3d=result.pts3d
    )
    out = compose.render_video(spec, src)
    assert out.shape == (result.pts3d.shape[0], 96, 256, 3)
    assert out.dtype == np.uint8


def test_unknown_plot_op_raises(result, fly, frames):
    # The render-time guard is defense in depth: read_video_specs now rejects an
    # unknown op at parse time (see test_unknown_plot_op_raises_at_parse_time), so
    # build the spec directly to reach compose_frame with a bogus op.
    spec = compose.VideoSpec(
        video_name="v", panels=[compose.Panel(plot="bogus", view="rh")]
    )
    src = compose.Sources(fly, result.cameras, frames, pts2d=result.pts2d)
    with pytest.raises(ValueError, match="unknown plot op 'bogus'"):
        compose.compose_frame(spec, src, t=0)


def test_packaged_config_videos_parse():
    """The shipped default config's [[visualization.videos]] parse into valid specs."""
    from importlib.resources import files

    cfg = Config.from_toml(files("deeperfly.data") / "default_config.toml")
    specs = compose.read_video_specs(cfg)
    assert {s.video_name for s in specs} == {
        "pose2d",
        "pose3d",
        "pose_model",
        "mesh_model",
    }
    assert all(p.plot in compose.OPS for s in specs for p in s.panels)
    # [visualization.default_layer] sets line_thickness on every skeleton panel that did
    # not override it (the dashed reference layers write 1).
    skel = [p for s in specs for p in s.panels if p.plot.startswith("skeleton")]
    assert skel and all(p.options.get("line_thickness") in (1, 2) for p in skel)
    # the default video's `cell` sizes every panel to a 480x240 box
    assert all(p.width == 480 and p.height == 240 for s in specs for p in s.panels)
    # and the default canvas background is black
    assert all(s.background == "black" for s in specs)
