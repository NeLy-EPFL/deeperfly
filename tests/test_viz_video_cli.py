"""Smoke tests for the headless visualization, video and CLI layers.

These check that figures render and MP4s round-trip, not pixel-level output.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import cli, video, viz
from deeperfly.io import PoseResult

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
    # pose3d options live in the config; disable calibration here.
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[pipeline]\ncalibrate = false\n")

    # `run` resumes from the cached 2D (start = pose3d); --until pose3d skips video.
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--until",
            "pose3d",
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


def test_cli_run_visualize_only(result, tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("")
    result.save(outdir / "poses.h5")  # full result (has 3D) -> auto-resume to visualize
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--fps",
            "5",
            "--log-level",
            "error",
        ]
    )
    assert video.read_video(outdir / "pose3d.mp4").shape[0] == result.n_frames
