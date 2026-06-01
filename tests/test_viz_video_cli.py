"""Smoke tests for the headless visualization, video and CLI layers.

These check that figures render and MP4s round-trip, not pixel-level output.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import cli, video, viz
from deeperfly.cameras import CameraGroup
from deeperfly.io import PoseResult
from deeperfly.skeleton import Skeleton


@pytest.fixture
def cameras(rig) -> CameraGroup:
    return CameraGroup.from_arrays(
        rig["names"], rig["rvecs"], rig["tvecs"], rig["intrs"], rig["dists"]
    )


@pytest.fixture
def fly() -> Skeleton:
    return Skeleton.fly()


@pytest.fixture
def result(cameras, fly, rng):
    pts3d = rng.uniform(-1.5, 1.5, size=(6, 38, 3))
    pts2d = np.array(cameras.project(pts3d))
    return PoseResult(
        cameras=cameras,
        skeleton=fly,
        pts2d=pts2d,
        conf=rng.uniform(0, 1, size=pts2d.shape[:3]),
        pts3d=pts3d,
        reproj_error=np.zeros(pts2d.shape[:3]),
    )


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


def test_leg_palette_colors(fly):
    import matplotlib.colors as mc

    colors = viz.limb_colors(fly)
    for joint, hexc in viz.LEG_PALETTE.items():
        # each leg's joints take that leg's color
        idx = [i for i, lid in enumerate(fly.limb_id) if fly.limb_names[lid] == joint]
        assert idx, joint
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


def test_cli_pose3d_and_info(result, tmp_path, capsys):
    in_path = tmp_path / "in.h5"
    out_path = tmp_path / "out.h5"
    # store only 2D so pose3d has to triangulate.
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        in_path
    )

    cli.main(["pose3d", "--in", str(in_path), "--out", str(out_path), "--no-calibrate"])
    out = PoseResult.load(out_path)
    assert out.pts3d is not None
    assert out.pts3d.shape == (result.n_frames, 38, 3)

    cli.main(["info", "--in", str(out_path)])
    printed = capsys.readouterr().out
    assert "skeleton: drosophila" in printed
    assert "has 3D:   True" in printed


def test_cli_visualize_3d(result, tmp_path):
    in_path = tmp_path / "res.h5"
    mp4 = tmp_path / "vid.mp4"
    result.save(in_path)
    cli.main(
        [
            "visualize",
            "--in",
            str(in_path),
            "--out",
            str(mp4),
            "--mode",
            "3d",
            "--fps",
            "5",
        ]
    )
    assert video.read_video(mp4).shape[0] == result.n_frames
