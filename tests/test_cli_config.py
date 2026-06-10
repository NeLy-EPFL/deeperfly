"""Tests for the merged config + the slim, config-driven CLI.

Covers ``deeperfly init`` (writing the packaged template), the drift guard that
keeps the template's inlined skeleton in step with ``Skeleton.fly()``, and the
``[inputs]`` filename->camera resolution used by ``deeperfly run``.
"""

from __future__ import annotations

import tomllib

import numpy as np
import pytest

from deeperfly import cli
from deeperfly.cameras import CameraGroup
from deeperfly.config import Config, DEFAULT_CONFIG_PATH
from deeperfly.recordings import camera_files, source_patterns
from deeperfly.skeleton import Skeleton


# -- deeperfly init ----------------------------------------------------------


def test_init_writes_parseable_config(tmp_path):
    dst = tmp_path / "config.toml"
    cli.main(["init", str(dst)])
    assert dst.exists()
    config = tomllib.load(dst.open("rb"))
    # The written file is the packaged template, verbatim.
    assert dst.read_text() == DEFAULT_CONFIG_PATH.read_text()
    # Every section round-trips through its loader.
    cameras = {n: spec for n, spec in config["cameras"].items() if n != "defaults"}
    sizes = {n: (512, 1024) for n in cameras}
    cfg = Config.from_dict(config)
    assert CameraGroup.from_config(cfg, image_sizes=sizes).names == [
        "rh",
        "rm",
        "rf",
        "f",
        "lf",
        "lm",
        "lh",
    ]
    assert Skeleton.from_config(cfg).n_points == 38
    # Footage globs live on the sources; the detection plan parses end to end.
    assert all("filename" in s for s in config["sources"])
    plan = cfg.detection_plan()
    assert len(plan.sources) == 7 and len(plan.pathways) == 8


def test_init_refuses_to_clobber(tmp_path):
    dst = tmp_path / "config.toml"
    dst.write_text("keep me\n")
    with pytest.raises(SystemExit):
        cli.main(["init", str(dst)])
    assert dst.read_text() == "keep me\n"  # untouched
    cli.main(["init", str(dst), "--force"])  # --force overwrites
    assert dst.read_text() == DEFAULT_CONFIG_PATH.read_text()


# -- template drift guard ----------------------------------------------------


def test_template_skeleton_matches_fly():
    config = Config.from_toml(DEFAULT_CONFIG_PATH)
    sk = Skeleton.from_config(config)
    fly = Skeleton.fly()
    assert sk.point_names == fly.point_names
    assert sk.limb_names == fly.limb_names
    np.testing.assert_array_equal(sk.limb_id, fly.limb_id)
    np.testing.assert_array_equal(sk.bones, fly.bones)
    assert sk.palette == fly.palette


# -- [inputs] filename -> camera resolution ----------------------------------


def test_camera_files_finds_video(tmp_path):
    # A bare name is a prefix, so "camera_0" globs "camera_0*" and matches the video.
    (tmp_path / "camera_0.mp4").write_bytes(b"x")
    assert camera_files(tmp_path, "camera_0") == [tmp_path / "camera_0.mp4"]


def test_camera_files_explicit_filename(tmp_path):
    # A value naming a file is matched verbatim (not used as a prefix).
    (tmp_path / "camera_0.mp4").write_bytes(b"x")
    (tmp_path / "camera_0_extra.mp4").write_bytes(b"x")
    assert camera_files(tmp_path, "camera_0.mp4") == [tmp_path / "camera_0.mp4"]


def test_camera_files_finds_image_sequence_natsorted(tmp_path):
    # An image sequence is the whole set of files, sorted naturally (2 before 10).
    for i in (0, 2, 10):
        (tmp_path / f"camera_0_img_{i}.jpg").write_bytes(b"x")
    assert camera_files(tmp_path, "camera_0") == [
        tmp_path / "camera_0_img_0.jpg",
        tmp_path / "camera_0_img_2.jpg",
        tmp_path / "camera_0_img_10.jpg",
    ]


def test_camera_files_subdir_via_explicit_glob(tmp_path):
    # A subdirectory of images is addressed with an explicit "<name>/*" glob.
    sub = tmp_path / "camera_0"
    sub.mkdir()
    (sub / "f0.png").write_bytes(b"x")
    (sub / "f1.png").write_bytes(b"x")
    assert camera_files(tmp_path, "camera_0/*") == [sub / "f0.png", sub / "f1.png"]


def test_camera_files_multiple_videos_keeps_first_and_warns(tmp_path, caplog):
    # Video footage is one file per camera: only the first matching video is used.
    for i in range(3):
        (tmp_path / f"camera_0_{i}.mp4").write_bytes(b"x")
    with caplog.at_level("WARNING", logger="deeperfly"):
        files = camera_files(tmp_path, "camera_0")
    assert files == [tmp_path / "camera_0_0.mp4"]
    assert any("using only the first" in r.message for r in caplog.records)


def test_camera_files_mixed_extensions_keeps_priority(tmp_path):
    # Video outranks image when both match the prefix.
    (tmp_path / "camera_0.mp4").write_bytes(b"x")
    (tmp_path / "camera_0.png").write_bytes(b"x")
    assert camera_files(tmp_path, "camera_0") == [tmp_path / "camera_0.mp4"]


def test_camera_files_missing_returns_empty(tmp_path):
    # No raise: the caller decides what an absent camera means.
    assert camera_files(tmp_path, "camera_9") == []


def test_source_patterns_defaults_to_source_name():
    # A source with no `filename` uses its own name as the pattern; [[sources]] order.
    config = Config.from_dict(
        {"sources": [{"name": "rh", "filename": "cam0.mp4"}, {"name": "lf"}]}
    )
    assert source_patterns(config) == {"rh": "cam0.mp4", "lf": "lf"}
