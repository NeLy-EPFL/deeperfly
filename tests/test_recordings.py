"""Tests for recording discovery: globbing, validation and output planning.

``recordings.py`` turns ``deeperfly run`` inputs into the ``(directory, camera ->
footage files)`` units of work. These cover the filesystem-globbing and path logic
directly with ``tmp_path`` fakes (image *sequences* and empty video files), so no
real decoding is needed -- ``find_recording`` only reads frame counts for video
footage, which these recordings avoid.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from deeperfly import recordings as rec
from deeperfly.config import Config


def _cfg(*names: str, filenames: list[str] | None = None) -> Config:
    """A minimal config whose ``[[sources]]`` name each camera (filename = glob)."""
    sources = []
    for i, name in enumerate(names):
        entry = {"name": name}
        if filenames is not None:
            entry["filename"] = filenames[i]
        sources.append(entry)
    return Config.from_dict({"sources": sources})


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def _seq(root: Path, prefix: str, n: int, ext: str = ".jpg") -> list[Path]:
    """An image sequence ``<prefix>_0000<ext> ...`` under ``root``."""
    return [_touch(root / f"{prefix}_{i:04d}{ext}") for i in range(n)]


# -- glob shaping ------------------------------------------------------------


def test_camera_glob_prefixes_bare_names():
    assert rec._camera_glob("camera_0") == "camera_0*"


def test_camera_glob_passes_through_files_and_wildcards():
    assert rec._camera_glob("camera_0.mp4") == "camera_0.mp4"  # known footage suffix
    assert rec._camera_glob("cam*") == "cam*"
    assert rec._camera_glob("camera_0/*") == "camera_0/*"


def test_has_glob():
    assert rec._has_glob("a*")
    assert rec._has_glob("a?b")
    assert rec._has_glob("a[0-9]")
    assert not rec._has_glob("plain_name")


def test_is_video_ext():
    assert rec._is_video_ext(".mp4")
    assert rec._is_video_ext(".MP4")  # case-insensitive
    assert not rec._is_video_ext(".jpg")


# -- default_outdir ----------------------------------------------------------


def test_default_outdir_for_directory(tmp_path):
    assert rec.default_outdir(tmp_path) == tmp_path / "deeperfly_outputs"


def test_default_outdir_for_file_uses_parent(tmp_path):
    f = _touch(tmp_path / "camera_0.mp4")
    assert rec.default_outdir(f) == tmp_path / "deeperfly_outputs"


# -- camera_files ------------------------------------------------------------


def test_camera_files_returns_image_sequence_natsorted(tmp_path):
    # natural (not lexicographic) order: _2 before _10
    _touch(tmp_path / "cam_10.jpg")
    _touch(tmp_path / "cam_2.jpg")
    _touch(tmp_path / "cam_1.jpg")
    files = rec.camera_files(tmp_path, "cam")
    assert [p.name for p in files] == ["cam_1.jpg", "cam_2.jpg", "cam_10.jpg"]


def test_camera_files_single_video(tmp_path):
    _touch(tmp_path / "camera_0.mp4")
    assert rec.camera_files(tmp_path, "camera_0") == [tmp_path / "camera_0.mp4"]


def test_camera_files_prefers_video_when_extensions_mix(tmp_path):
    _touch(tmp_path / "cam_0.mp4")
    _seq(tmp_path, "cam_0", 3)  # also cam_0_0000.jpg ...
    files = rec.camera_files(tmp_path, "cam_0")
    assert [p.name for p in files] == ["cam_0.mp4"]  # video wins over images


def test_camera_files_keeps_only_first_of_several_videos(tmp_path):
    _touch(tmp_path / "cam_a.mp4")
    _touch(tmp_path / "cam_b.mp4")
    files = rec.camera_files(tmp_path, "cam")  # prefix matches both videos
    assert len(files) == 1
    assert files[0].name == "cam_a.mp4"


def test_camera_files_empty_when_no_footage(tmp_path):
    _touch(tmp_path / "notes.txt")  # not a footage extension
    assert rec.camera_files(tmp_path, "cam") == []


# -- find_recording ----------------------------------------------------------


def test_find_recording_valid_image_sequences(tmp_path):
    _seq(tmp_path, "cam_a", 4)
    _seq(tmp_path, "cam_b", 4)
    sources = rec.find_recording(tmp_path, _cfg("cam_a", "cam_b"))
    assert sources is not None
    assert set(sources) == {"cam_a", "cam_b"}
    assert len(sources["cam_a"]) == 4


def test_find_recording_none_when_not_a_directory(tmp_path):
    f = _touch(tmp_path / "cam_a_0000.jpg")
    assert rec.find_recording(f, _cfg("cam_a")) is None


def test_find_recording_none_when_no_camera_matches(tmp_path):
    _touch(tmp_path / "unrelated.jpg")
    # nothing matches the cam_a/cam_b globs -> not a recording (silently)
    assert rec.find_recording(tmp_path, _cfg("cam_a", "cam_b")) is None


def test_find_recording_none_when_some_cameras_missing(tmp_path):
    _seq(tmp_path, "cam_a", 3)  # cam_b has no footage
    assert rec.find_recording(tmp_path, _cfg("cam_a", "cam_b")) is None


def test_find_recording_none_on_uneven_file_count(tmp_path):
    _seq(tmp_path, "cam_a", 4)
    _seq(tmp_path, "cam_b", 3)  # mismatched sequence length
    assert rec.find_recording(tmp_path, _cfg("cam_a", "cam_b")) is None


# -- _frame_counts_match (image path: file counts settle it) -----------------


def test_frame_counts_match_equal_image_sequences(tmp_path):
    sources = {"a": _seq(tmp_path, "a", 3), "b": _seq(tmp_path, "b", 3)}
    assert rec._frame_counts_match(tmp_path, sources) is True


def test_frame_counts_match_unequal_file_counts(tmp_path):
    sources = {"a": _seq(tmp_path, "a", 3), "b": _seq(tmp_path, "b", 2)}
    assert rec._frame_counts_match(tmp_path, sources) is False


# -- plan_outdirs ------------------------------------------------------------


def test_plan_outdirs_single_default(tmp_path):
    rec_dir = tmp_path / "rec"
    rec_dir.mkdir()  # default_outdir nests inside an existing directory
    plan = rec.plan_outdirs([rec_dir], None)
    assert plan.outdirs == [rec_dir / "deeperfly_outputs"]
    assert plan.mirror_confirm is None


def test_plan_outdirs_single_explicit_output():
    plan = rec.plan_outdirs([Path("rec")], "somewhere/out")
    assert plan.outdirs == [Path("somewhere/out")]


def test_plan_outdirs_batch_default(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    plan = rec.plan_outdirs([a, b], None)
    assert plan.outdirs == [a / "deeperfly_outputs", b / "deeperfly_outputs"]


def test_plan_outdirs_batch_relative_nests_inside_each():
    plan = rec.plan_outdirs([Path("a"), Path("b")], "results")
    assert plan.outdirs == [Path("a/results"), Path("b/results")]


def test_plan_outdirs_batch_collect_trailing_slash():
    plan = rec.plan_outdirs([Path("x/rec1"), Path("y/rec2")], "out/")
    assert plan.outdirs == [Path("out/rec1"), Path("out/rec2")]
    assert plan.mirror_confirm is None


def test_plan_outdirs_batch_absolute_treated_as_collect():
    plan = rec.plan_outdirs([Path("a/x"), Path("b/y")], "/abs/out")
    assert plan.outdirs == [Path("/abs/out/x"), Path("/abs/out/y")]


def test_plan_outdirs_batch_name_collision_mirrors_with_confirm(tmp_path):
    a = tmp_path / "a" / "rec"
    b = tmp_path / "b" / "rec"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    plan = rec.plan_outdirs([a, b], "out/")
    assert plan.mirror_confirm is not None
    assert "collide" in plan.mirror_confirm
    # mirrored from the common ancestor (tmp_path) so the two can't share a dir
    assert plan.outdirs == [Path("out/a/rec"), Path("out/b/rec")]


# -- resolve_recordings ------------------------------------------------------


def test_resolve_single_valid_recording(tmp_path):
    _seq(tmp_path, "cam_a", 3)
    _seq(tmp_path, "cam_b", 3)
    found = rec.resolve_recordings(
        [tmp_path], recursive=False, config=_cfg("cam_a", "cam_b")
    )
    assert len(found) == 1
    path, sources = found[0]
    assert path == tmp_path
    assert set(sources) == {"cam_a", "cam_b"}


def test_resolve_single_invalid_kept_with_empty_sources(tmp_path):
    # A single literal path is resume-friendly: kept even without valid footage.
    rec_dir = tmp_path / "out_only"
    rec_dir.mkdir()
    found = rec.resolve_recordings(
        [rec_dir], recursive=False, config=_cfg("cam_a", "cam_b")
    )
    assert found == [(rec_dir, {})]


def test_resolve_batch_wildcard_keeps_only_valid(tmp_path):
    good = tmp_path / "rec_good"
    _seq(good, "cam_a", 2)
    _seq(good, "cam_b", 2)
    bad = tmp_path / "rec_bad"
    _seq(bad, "cam_a", 2)  # missing cam_b -> dropped silently for a wildcard
    found = rec.resolve_recordings(
        [tmp_path / "rec_*"], recursive=False, config=_cfg("cam_a", "cam_b")
    )
    assert [p for p, _ in found] == [good]


def test_resolve_recursive_walks_subtree(tmp_path):
    for name in ("rec1", "rec2"):
        d = tmp_path / name
        _seq(d, "cam_a", 2)
        _seq(d, "cam_b", 2)
    found = rec.resolve_recordings(
        [tmp_path], recursive=True, config=_cfg("cam_a", "cam_b")
    )
    assert sorted(p.name for p, _ in found) == ["rec1", "rec2"]


def test_resolve_no_valid_recording_raises(tmp_path):
    with pytest.raises(SystemExit):
        rec.resolve_recordings(
            [tmp_path / "missing_*"], recursive=False, config=_cfg("cam_a")
        )


# -- require_input_footage ---------------------------------------------------


def test_require_input_footage_missing_dir(tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        rec.require_input_footage(_cfg("cam_a"), input=tmp_path / "nope")


def test_require_input_footage_not_a_directory(tmp_path):
    f = _touch(tmp_path / "camera_0.mp4")
    with pytest.raises(SystemExit, match="not a directory"):
        rec.require_input_footage(_cfg("cam_a"), input=f)


def test_require_input_footage_camera_missing_files(tmp_path):
    _seq(tmp_path, "cam_a", 2)  # cam_b absent
    with pytest.raises(SystemExit, match="no video or images for camera 'cam_b'"):
        rec.require_input_footage(_cfg("cam_a", "cam_b"), input=tmp_path)


def test_require_input_footage_valid_input_passes(tmp_path):
    _seq(tmp_path, "cam_a", 2)
    _seq(tmp_path, "cam_b", 2)
    assert rec.require_input_footage(_cfg("cam_a", "cam_b"), input=tmp_path) is None


def test_require_input_footage_sources_missing_camera():
    sources = {"cam_a": [Path("cam_a_0000.jpg")], "cam_b": []}
    with pytest.raises(SystemExit, match="needs footage"):
        rec.require_input_footage(_cfg("cam_a", "cam_b"), sources=sources)


def test_require_input_footage_sources_complete_passes():
    sources = {"cam_a": [Path("cam_a.jpg")], "cam_b": [Path("cam_b.jpg")]}
    assert rec.require_input_footage(_cfg("cam_a", "cam_b"), sources=sources) is None
