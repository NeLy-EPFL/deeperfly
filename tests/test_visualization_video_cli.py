"""Smoke tests for the video and CLI layers (OpenCV overlays live in
test_viz_opencv.py).

These check that MP4s round-trip and the CLI renders, not pixel-level output.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import cli, io, pipeline
from deeperfly.results import PoseResult
from deeperfly.visualization import compose

# `cameras`, `fly` and `result` fixtures live in conftest.py (shared with
# test_cli_run.py).


# -- video -------------------------------------------------------------------


def test_write_read_mp4_roundtrip(tmp_path, rng):
    frames = rng.integers(0, 255, size=(8, 64, 48, 3), dtype=np.uint8)
    path = tmp_path / "clip.mp4"
    with io.VideoWriter(path, fps=10) as writer:
        writer.write_frames(frames)
    back = io.VideoReader(path).read()
    assert back.shape[0] == 8
    assert back.shape[1:3] == (64, 48)


def test_read_images(tmp_path, rng):
    import cv2

    for i in range(3):
        img = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / f"frame_{i:03d}.png"), img)
    frames = io.open_reader(tmp_path).read()
    assert frames.shape == (3, 16, 16, 3)


def test_video_fps_roundtrip(tmp_path, rng):
    import cv2

    frames = rng.integers(0, 255, size=(8, 32, 32, 3), dtype=np.uint8)
    path = tmp_path / "clip.mp4"
    with io.VideoWriter(path, fps=12) as writer:
        writer.write_frames(frames)
    assert abs(io.open_reader(path).fps() - 12) < 0.5
    # an image sequence carries no intrinsic frame rate.
    img_dir = tmp_path / "frames"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "frame_000.png"), frames[0])
    assert io.open_reader(img_dir).fps() is None


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
    assert io.VideoReader(outdir / "pose3d.mp4").read().shape[0] == result.n_frames


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
        pipeline.stages,
        "render_videos",
        lambda *a, **k: pytest.fail("should reuse the MP4"),
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
    assert io.VideoReader(outdir / "pose3d.mp4").read().shape[0] == result.n_frames
