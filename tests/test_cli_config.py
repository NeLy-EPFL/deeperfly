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
from deeperfly.cli import DEFAULT_CONFIG_PATH, _camera_source
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
    sizes = {n: (512, 1024) for n in config["cameras"]}
    assert CameraGroup.from_config(config, image_sizes=sizes).names == [
        "rh",
        "rm",
        "rf",
        "f",
        "lf",
        "lm",
        "lh",
    ]
    assert Skeleton.from_config(config).n_points == 38
    assert set(config["inputs"]) == set(config["cameras"])


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
    config = tomllib.load(DEFAULT_CONFIG_PATH.open("rb"))
    sk = Skeleton.from_config(config)
    fly = Skeleton.fly()
    assert sk.joint_names == fly.joint_names
    assert sk.limb_names == fly.limb_names
    np.testing.assert_array_equal(sk.limb_id, fly.limb_id)
    np.testing.assert_array_equal(sk.bones, fly.bones)
    np.testing.assert_array_equal(sk.bones3d, fly.bones3d)
    np.testing.assert_array_equal(sk.left_idx, fly.left_idx)
    np.testing.assert_array_equal(sk.right_idx, fly.right_idx)
    assert set(sk.visibility) == set(fly.visibility)
    for cam in fly.visibility:
        np.testing.assert_array_equal(sk.visibility[cam], fly.visibility[cam])


# -- [inputs] filename -> camera resolution ----------------------------------


def test_camera_source_finds_video(tmp_path):
    (tmp_path / "camera_0.mp4").write_bytes(b"x")
    assert _camera_source(tmp_path, "camera_0") == tmp_path / "camera_0.mp4"


def test_camera_source_finds_image_sequence(tmp_path):
    for i in range(3):
        (tmp_path / f"camera_0_img_{i:06d}.jpg").write_bytes(b"x")
    src = _camera_source(tmp_path, "camera_0")
    assert str(src) == str(tmp_path / "camera_0*")


def test_camera_source_finds_image_subdir(tmp_path):
    (tmp_path / "camera_0").mkdir()
    assert _camera_source(tmp_path, "camera_0") == tmp_path / "camera_0"


def test_camera_source_falls_back_to_name(tmp_path):
    # An unmapped camera uses its own name as the prefix.
    (tmp_path / "rh.mp4").write_bytes(b"x")
    assert _camera_source(tmp_path, "rh") == tmp_path / "rh.mp4"


def test_camera_source_missing_raises(tmp_path):
    with pytest.raises(SystemExit, match="no video or images for camera 'camera_9'"):
        _camera_source(tmp_path, "camera_9")
