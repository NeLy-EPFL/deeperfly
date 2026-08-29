"""Tests for the merged config + the slim, config-driven CLI.

Covers ``deeperfly init`` (writing the packaged template) and the ``[inputs]``
filename->camera resolution used by ``deeperfly run``.
"""

from __future__ import annotations

import tomllib

from deeperfly import cli
from deeperfly.config import DEFAULT_CONFIG_PATH, Config
from deeperfly.recordings import camera_files, source_patterns
from deeperfly.rig.cameras import CameraGroup
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
    # The six side cameras, the front one, and the axial hind view -- which is the rig's
    # only left/right bridge and what the shipped detector was trained on. Asserted in
    # ORDER, because that order is the V axis of every (V, T, P, ...) array a run writes.
    assert CameraGroup.from_config(cfg, image_sizes=sizes).names == [
        "rh",
        "rm",
        "rf",
        "f",
        "lf",
        "lm",
        "lh",
        "h",
    ]
    assert Skeleton.from_config(cfg).n_points == 38
    # Footage patterns live on the cameras; the detection plan builds end to end.
    assert all("video" in spec for spec in config["cameras"].values())
    plan = cfg.detection_plan()
    # Dense: one pathway per camera, and one source per pathway.
    assert len(plan.sources) == len(cameras)
    assert len(plan.pathways) == len(cameras)


def test_init_refuses_to_clobber(tmp_path, capsys):
    dst = tmp_path / "config.toml"
    dst.write_text("keep me\n")
    cli.main(["init", str(dst)])  # without --overwrite: warns, leaves the file alone
    assert dst.read_text() == "keep me\n"  # untouched
    # rich may hard-wrap the message; normalize whitespace before matching.
    assert "already exists" in " ".join(capsys.readouterr().out.split())
    cli.main(["init", str(dst), "--overwrite"])  # --overwrite replaces it
    assert dst.read_text() == DEFAULT_CONFIG_PATH.read_text()


# -- [inputs] filename -> camera resolution ----------------------------------


def test_camera_files_matches_the_pattern_in_full(tmp_path):
    """A full match on the filename, extension included -- nothing is inferred.

    A plain filename needs no escaping: glob is the default, and `.` is literal.
    """
    (tmp_path / "camera_0.mp4").write_bytes(b"x")
    (tmp_path / "camera_0_extra.mp4").write_bytes(b"x")
    assert camera_files(tmp_path, "camera_0.mp4") == [tmp_path / "camera_0.mp4"]


def test_camera_files_finds_an_image_sequence_natsorted(tmp_path):
    # An image sequence is the whole set of files, sorted naturally (2 before 10) --
    # the same rule a SPLIT recording follows.
    for i in (0, 2, 10):
        (tmp_path / f"camera_0_img_{i}.jpg").write_bytes(b"x")
    assert camera_files(tmp_path, "camera_0_img_*.jpg") == [
        tmp_path / "camera_0_img_0.jpg",
        tmp_path / "camera_0_img_2.jpg",
        tmp_path / "camera_0_img_10.jpg",
    ]


def test_several_videos_concatenate_rather_than_keeping_the_first(tmp_path):
    """The v1 rule was "keep the first, warn"; a split recording is one stream now."""
    for i in range(3):
        (tmp_path / f"camera_0_{i}.mp4").write_bytes(b"x")
    files = camera_files(tmp_path, "camera_0_*.mp4")
    assert [p.name for p in files] == [
        "camera_0_0.mp4",
        "camera_0_1.mp4",
        "camera_0_2.mp4",
    ]


def test_mixed_extensions_are_an_error_not_a_silent_pick(tmp_path):
    """v1 ranked video over images; a pattern names its extension now."""
    import pytest

    (tmp_path / "camera_0.mp4").write_bytes(b"x")
    (tmp_path / "camera_0.jpg").write_bytes(b"x")
    with pytest.raises(ValueError, match="not parts of one series"):
        camera_files(tmp_path, r"/camera_0\..+/")


def test_camera_files_missing_returns_empty(tmp_path):
    # No raise: the caller decides what an absent camera means.
    assert camera_files(tmp_path, "camera_9") == []


def test_source_patterns_defaults_to_the_camera_name():
    # A camera with no `video` uses its own name as the pattern; camera order.
    config = Config.from_dict({"cameras": {"rh": {"video": "cam0.mp4"}, "lf": {}}})
    assert source_patterns(config) == {"rh": "cam0.mp4", "lf": "lf"}
