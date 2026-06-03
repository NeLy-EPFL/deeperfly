"""Smoke tests for the headless visualization, video and CLI layers.

These check that figures render and MP4s round-trip, not pixel-level output.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import cli, video, viz
from deeperfly.io import PoseResult
from deeperfly.viz import compose

# `cameras`, `fly` and `result` fixtures live in conftest.py (shared with
# test_cli_run.py).


# -- viz ---------------------------------------------------------------------


def test_plot_skeleton_2d_runs(result):
    ax = viz.plot_skeleton_2d(
        result.pts2d[0, 0], result.skeleton, conf=result.conf[0, 0]
    )
    assert ax.has_data()


def test_plot_skeleton_3d_runs(result):
    ax = viz.plot_skeleton_3d(result.pts3d[0], result.skeleton)
    assert ax.has_data()


def test_overlay_grid_runs(result):
    fig = viz.overlay_grid(
        result.pts2d[:, 0], result.skeleton, camera_names=result.cameras.names
    )
    assert len(fig.axes) >= result.n_views


def test_limb_palette_colors(fly):
    import matplotlib.colors as mc

    colors = viz.limb_colors(fly)
    for limb, hexc in fly.palette.items():
        # each of the limb's joints takes that limb's color from the skeleton palette
        idx = [i for i, lid in enumerate(fly.limb_id) if fly.limb_names[lid] == limb]
        assert idx, limb
        for i in idx:
            np.testing.assert_allclose(colors[i], mc.to_rgba(hexc))


def test_background_modes(result):
    import matplotlib.colors as mc

    for bg, face in (("white", "#ffffff"), ("black", "#000000")):
        ax = viz.plot_skeleton_3d(result.pts3d[0], result.skeleton, background=bg)
        assert mc.to_hex(ax.get_figure().get_facecolor()) == face
        ax2 = viz.plot_skeleton_2d(result.pts2d[0, 0], result.skeleton, background=bg)
        assert mc.to_hex(ax2.get_figure().get_facecolor()) == face
    with pytest.raises(ValueError, match="background must be one of"):
        viz.plot_skeleton_3d(result.pts3d[0], result.skeleton, background="navy")


# -- video -------------------------------------------------------------------


def test_write_read_mp4_roundtrip(tmp_path, rng):
    frames = rng.integers(0, 255, size=(8, 64, 48, 3), dtype=np.uint8)
    path = tmp_path / "clip.mp4"
    video.write_mp4(frames, path, fps=10)
    back = video.read_video(path)
    assert back.shape[0] == 8
    assert back.shape[1:3] == (64, 48)


def test_read_images(tmp_path, rng):
    import imageio.v2 as imageio

    for i in range(3):
        imageio.imwrite(
            tmp_path / f"frame_{i:03d}.png",
            rng.integers(0, 255, (16, 16, 3), dtype=np.uint8),
        )
    frames = video.read_images(tmp_path)
    assert frames.shape == (3, 16, 16, 3)


def test_render_pose3d_video(result, tmp_path):
    path = tmp_path / "pose3d.mp4"
    video.render_pose3d_video(result, path, fps=5)
    assert video.read_video(path).shape[0] == result.n_frames


def test_video_fps_roundtrip(tmp_path, rng):
    frames = rng.integers(0, 255, size=(8, 32, 32, 3), dtype=np.uint8)
    path = tmp_path / "clip.mp4"
    video.write_mp4(frames, path, fps=12)
    assert abs(video.video_fps(path) - 12) < 0.5
    # an image sequence / non-video path carries no intrinsic frame rate.
    assert video.video_fps(tmp_path / "frames") is None


# -- visualization output fps (output_fps / speed) ---------------------------


def test_video_spec_resolve_fps_levels():
    """output_fps / speed resolve with per-video winning over the global setting."""
    config = {
        "pipeline": {
            "visualization": {
                "speed": 0.5,  # global: half speed unless a video overrides it
                "videos": [
                    {"video_name": "a", "panels": []},  # inherits global speed
                    {"video_name": "b", "panels": [], "output_fps": 24},  # explicit fps
                    {"video_name": "c", "panels": [], "speed": 2},  # own speed
                ],
            }
        }
    }
    specs = {s.video_name: s for s in compose.read_video_specs(config)}
    assert specs["a"].resolve_fps(30) == 15.0  # 30 * global speed 0.5
    assert specs["b"].resolve_fps(30) == 24.0  # per-video output_fps beats global speed
    assert specs["c"].resolve_fps(30) == 60.0  # 30 * per-video speed 2


def test_video_spec_resolve_fps_defaults_to_input():
    (spec,) = compose.read_video_specs(
        {"pipeline": {"visualization": {"videos": [{"video_name": "a", "panels": []}]}}}
    )
    assert spec.output_fps is None and spec.speed is None
    assert spec.resolve_fps(37.5) == 37.5  # native input rate when neither is set


# -- cli ---------------------------------------------------------------------


def _seed_2d(result, outdir):
    """Pre-seed a 2D-only poses.h5 in ``outdir`` so a run resumes at pose3d."""
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )


def test_cli_run_resume_pose3d_and_info(result, tmp_path, capsys):
    outdir = tmp_path / "out"
    _seed_2d(result, outdir)
    # Resume: pose2d/bundle_adjustment/visualization off (reuse the cached 2D, no
    # recalibration, no video); triangulate the cached 2D into 3D.
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = true\ndo_visualization = false\n"
    )

    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--log-level",
            "error",
        ]
    )
    out = PoseResult.load(outdir / "poses.h5")
    assert out.pts3d is not None
    assert out.pts3d.shape == (result.n_frames, 38, 3)

    cli.main(["inspect", str(outdir / "poses.h5")])
    printed = capsys.readouterr().out
    assert "skeleton: fly38  (38 points)" in printed
    assert "has 3D:   True" in printed


def test_cli_run_visualization_only(result, tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    cfg = tmp_path / "cfg.toml"
    # every compute stage off -> visualization from the cached 3D result. A
    # skeleton_3d-only video needs no image frames (canvas sized from the camera's
    # intrinsics); fps comes from the config.
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_visualization = true\nfps = 5\n"
        "[[pipeline.visualization.videos]]\n"
        'video_name = "pose3d"\n'
        'panels = [{ plot = "skeleton_3d", view = "rh", x0 = 0, y0 = 0 }]\n'
    )
    result.save(outdir / "poses.h5")  # full result (has 3D)
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--log-level",
            "error",
        ]
    )
    assert video.read_video(outdir / "pose3d.mp4").shape[0] == result.n_frames


def _viz_only_cfg(tmp_path):
    """A compute-off config rendering one skeleton_3d video (needs no input frames)."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_visualization = true\nfps = 5\n"
        "[[pipeline.visualization.videos]]\n"
        'video_name = "pose3d"\n'
        'panels = [{ plot = "skeleton_3d", view = "rh", x0 = 0, y0 = 0 }]\n'
    )
    return cfg


def test_visualization_reused_when_mp4_exists(result, tmp_path, monkeypatch):
    """An existing output MP4 is reused (not re-rendered) without --overwrite."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")
    cfg = _viz_only_cfg(tmp_path)
    (outdir / "pose3d.mp4").write_bytes(b"already rendered")  # pretend a prior render
    monkeypatch.setattr(
        cli, "_stage_visualization", lambda *a, **k: pytest.fail("should reuse the MP4")
    )
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--log-level",
            "error",
        ]
    )
    assert (outdir / "pose3d.mp4").read_bytes() == b"already rendered"  # untouched


def test_overwrite_visualization_rerenders(result, tmp_path):
    """--overwrite visualization re-renders even when the output MP4 exists."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")
    cfg = _viz_only_cfg(tmp_path)
    (outdir / "pose3d.mp4").write_bytes(b"stale")  # not a real video
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--overwrite",
            "visualization",
            "--log-level",
            "error",
        ]
    )
    assert video.read_video(outdir / "pose3d.mp4").shape[0] == result.n_frames
