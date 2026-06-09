"""Tests for the unified ``deeperfly run`` command and its plumbing.

Covers the per-stage flags (:meth:`Config.stage_flags`), config resolution
(:meth:`Config.read_for_run`: the output-dir snapshot wins over ``-c``, else
``-c``, else the packaged default), the
stage execution -- an enabled stage runs and recomputes, a disabled one is reused
from the cached ``poses.h5``, and an enabled stage whose input is missing is
skipped with a reason -- the default ``<input>/deeperfly_outputs``, automatic
weight provisioning, the pictorial-skipped notice on resume, the detector progress
hook, and the overlay frame-recovery order. The detector and frame I/O are stubbed
so these need neither real weights nor video files.
"""

from __future__ import annotations

import argparse
import re

import numpy as np
import pytest

from deeperfly import Config, cli
from deeperfly.cameras import CameraGroup
from deeperfly.io import PoseResult

FLY_CAMERAS = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]


def _default_cfg(tmp_path, *, name="config.toml", **flags):
    """Write the packaged default config with the given ``do_<stage>`` flags flipped.

    Pass any subset of stage names (``pose2d``, ``bundle_adjustment``,
    ``pictorial_structures``, ``triangulation``, ``smoothing``, ``visualization``) as
    keyword booleans. The full default is needed by tests that run the ``pose2d``
    stage (it builds the camera rig from ``[cameras]``); tests that resume from a
    cached result use small hand-written configs instead.
    """
    text = cli.DEFAULT_CONFIG_PATH.read_text()
    for stage, on in flags.items():
        # Anchor to the start of a line so a `do_<stage> = ...` example inside a
        # comment is not matched ahead of the real [pipeline] flag.
        text, n = re.subn(
            rf"(?m)^(do_{stage}\s*=\s*)(?:true|false)",
            rf"\g<1>{str(on).lower()}",
            text,
            count=1,
        )
        assert n == 1, f"do_{stage} not found in the default config"
    cfg = tmp_path / name
    cfg.write_text(text)
    return cfg


# -- stage flags -------------------------------------------------------------


def test_stage_flags_defaults():
    """An empty config uses STAGE_DEFAULTS; the packaged default matches it."""
    assert Config.from_dict({}).stage_flags() == cli.STAGE_DEFAULTS
    assert Config.default().stage_flags() == {
        "pose2d": True,
        "bundle_adjustment": True,
        "pictorial_structures": False,
        "triangulation": True,
        "smoothing": False,
        "visualization": True,
    }


def test_stage_flags_toggles_and_validates():
    flags = Config.from_dict(
        {"pipeline": {"do_pose2d": False, "do_smoothing": True}}
    ).stage_flags()
    assert flags["pose2d"] is False and flags["smoothing"] is True
    assert flags["triangulation"] is True  # untouched default
    with pytest.raises(SystemExit, match="unknown stage toggle"):
        Config.from_dict({"pipeline": {"do_detekt": True}})  # validated at construction


def test_stage_flags_rejects_removed_keys():
    """Old [stages] / [pipeline] keys fail with a pointer to the new location."""
    with pytest.raises(SystemExit, match="do_bundle_adjustment"):
        Config.from_dict({"pipeline": {"calibrate": True}})
    with pytest.raises(SystemExit, match="do_visualization"):
        Config.from_dict({"pipeline": {"do_visualize": True}})
    with pytest.raises(SystemExit, match=r"triangulation"):
        Config.from_dict({"pipeline": {"triangulation": "ransac"}})
    with pytest.raises(SystemExit, match="stages"):
        Config.from_dict({"stages": {"detect": True}})
    # the new [pipeline.triangulation] sub-table is *not* mistaken for the removed
    # scalar key.
    assert Config.from_dict(
        {"pipeline": {"triangulation": {"method": "dlt"}}}
    ).stage_flags()


# -- config resolution -------------------------------------------------------


def test_resolve_config_default_when_no_cli_and_no_snapshot(tmp_path):
    config = Config.read_for_run(None, tmp_path)
    assert config.source == cli.DEFAULT_CONFIG_PATH
    assert config.data == Config.default().data


def test_resolve_config_uses_cli_when_no_snapshot(tmp_path):
    cfg = tmp_path / "mine.toml"
    cfg.write_text("[pipeline]\ndo_pose2d = false\n")
    config = Config.read_for_run(str(cfg), tmp_path)
    assert config.source == cfg and config.data["pipeline"] == {"do_pose2d": False}


def test_resolve_config_snapshot_wins_over_cli_and_notifies(tmp_path, caplog):
    snapshot = tmp_path / "config.toml"
    snapshot.write_text("[pipeline]\ndo_triangulation = false\n")
    other = tmp_path / "other.toml"
    other.write_text("[pipeline]\ndo_triangulation = true\n")
    with caplog.at_level("WARNING"):
        config = Config.read_for_run(str(other), tmp_path)
    assert config.source == snapshot and config.data["pipeline"] == {
        "do_triangulation": False
    }
    assert any("ignoring -c" in r.message for r in caplog.records)


def test_resolve_config_snapshot_used_silently_without_cli(tmp_path, caplog):
    snapshot = tmp_path / "config.toml"
    snapshot.write_text("[pipeline]\ndo_visualization = false\n")
    with caplog.at_level("WARNING"):
        config = Config.read_for_run(None, tmp_path)
    assert config.source == snapshot and config.data["pipeline"] == {
        "do_visualization": False
    }
    assert not [r for r in caplog.records if "ignoring" in r.message]


# -- run: stage toggles, caching, skips -------------------------------------


def _stub_detect(monkeypatch, tmp_path):
    """Stub frame sizing + detection so detect needs no files, weights or recording."""
    T, H, W = 3, 16, 16
    sizes = {n: (H, W) for n in FLY_CAMERAS}
    monkeypatch.setattr(cli, "_camera_image_sizes", lambda args, config: sizes)
    monkeypatch.setattr(cli, "_load_detector", lambda checkpoint: object())
    # The recording footage is stubbed away, so skip the pre-run footage validation a
    # real fresh pose2d run does (covered separately by the input-resolution tests).
    monkeypatch.setattr(cli, "_require_input_footage", lambda args, config: None)

    def fake_detect_2d(args, config, model, sides, flips, **kw):
        v = len(FLY_CAMERAS)
        return np.zeros((v, T, 38, 2)), np.ones((v, T, 38)), None

    monkeypatch.setattr(cli, "_detect_2d", fake_detect_2d)
    return T


def _make_fly_recording(d):
    """A directory holding (empty) per-camera videos for the packaged 7-camera rig."""
    d.mkdir(parents=True, exist_ok=True)
    for i in range(len(FLY_CAMERAS)):
        (d / f"camera_{i}.mp4").write_bytes(b"")
    return d


def test_pose2d_only_writes_2d_result(tmp_path, monkeypatch):
    """pose2d on, every later stage off -> a 2D-only result + snapshot."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    T = _stub_detect(monkeypatch, tmp_path)

    outdir = tmp_path / "out"
    rec = tmp_path / "rec"
    cli.main(
        ["run", str(rec), "-c", str(cfg), "-o", str(outdir), "--log-level", "error"]
    )

    res = PoseResult.load(outdir / "poses.h5")
    assert res.pts3d is None  # 2D only (no triangulation/pictorial)
    assert res.pts2d.shape == (7, T, 38, 2)
    # the config used is snapshotted next to the results for reproducibility.
    assert (outdir / "config.toml").read_text() == cfg.read_text()


def test_run_without_config_uses_default(tmp_path, monkeypatch):
    """Omitting -c falls back to the packaged default config (snapshotted as-is)."""
    _stub_detect(monkeypatch, tmp_path)
    # The default runs pose2d + bundle_adjustment + triangulation + visualization; stub
    # the compute/render stages so the run needs no real BA, triangulation or video
    # frames -- this test is about config resolution.
    monkeypatch.setattr(cli, "_stage_bundle_adjustment", lambda config, result: result)
    monkeypatch.setattr(cli, "_stage_triangulation", lambda config, result: result)
    monkeypatch.setattr(cli, "_stage_visualization", lambda *a, **k: None)
    outdir = tmp_path / "out"
    rec = tmp_path / "rec"
    cli.main(["run", str(rec), "-o", str(outdir), "--log-level", "error"])
    assert (outdir / "poses.h5").exists()
    assert (outdir / "config.toml").read_text() == cli.DEFAULT_CONFIG_PATH.read_text()


def test_default_outdir_inside_input(tmp_path, monkeypatch):
    """With no -o, results land in <input>/deeperfly_outputs/."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _stub_detect(monkeypatch, tmp_path)
    rec = tmp_path / "rec"
    rec.mkdir()
    cli.main(["run", str(rec), "-c", str(cfg), "--log-level", "error"])
    assert (rec / "deeperfly_outputs" / "poses.h5").exists()


# -- input footage validated before the output dir is created ----------------


def _footage_cfg(tmp_path):
    """A minimal two-camera config whose pose2d run reads real footage."""
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[cameras.cam0]\n[cameras.cam1]\n"
        "[pipeline]\ndo_pose2d = true\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_visualization = false\n"
    )
    return cfg


def test_run_errors_on_missing_recording_before_creating_outdir(tmp_path):
    """A fresh pose2d run whose recording does not exist fails up front, leaving no
    empty output dir behind."""
    outdir = tmp_path / "out"
    with pytest.raises(SystemExit, match="needs footage for pose2d"):
        cli.main(
            [
                "run",
                str(tmp_path / "ghost"),
                "-c",
                str(_footage_cfg(tmp_path)),
                "-o",
                str(outdir),
                "--log-level",
                "error",
            ]
        )
    assert not outdir.exists()  # validated before any deeperfly_outputs was made


def test_run_errors_on_missing_camera_footage_before_creating_outdir(tmp_path, caplog):
    """A recording that exists but lacks a configured camera's footage is warned
    (naming the camera) at discovery and fails before the output dir is created."""
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "cam0.mp4").write_bytes(b"")  # cam0 present, cam1 missing
    outdir = tmp_path / "out"
    with caplog.at_level("WARNING", logger="deeperfly"):
        with pytest.raises(SystemExit, match="needs footage for pose2d"):
            cli.main(
                [
                    "run",
                    str(rec),
                    "-c",
                    str(_footage_cfg(tmp_path)),
                    "-o",
                    str(outdir),
                ]
            )
    assert any("missing ['cam1']" in r.message for r in caplog.records)
    assert not outdir.exists()


def test_resume_skips_footage_validation_when_pose2d_cached(result, tmp_path):
    """A resume that reuses the cached 2D (pose2d off) does not require the
    recording, so a missing input is fine."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = true\ndo_visualization = false\n"
    )
    cli.main(
        [
            "run",
            str(tmp_path / "ghost"),  # no recording on disk
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--log-level",
            "error",
        ]
    )
    assert PoseResult.load(outdir / "poses.h5").pts3d is not None


# -- input resolution: multiple inputs, wildcards, --recursive ----------------

_RES_CFG = Config.from_dict({"cameras": {"cam0": {}, "cam1": {}}})


def _make_recording(d, *, ext="mp4", count=1):
    """A directory holding (empty) footage files for the two configured cameras.

    ``count`` files per camera (an image sequence when > 1); ``ext`` sets the
    extension. Empty files, so discovery validates filenames without decoding.
    """
    d.mkdir(parents=True, exist_ok=True)
    for cam in ("cam0", "cam1"):
        for i in range(count):
            suffix = f"_{i}" if count > 1 else ""
            (d / f"{cam}{suffix}.{ext}").write_bytes(b"")
    return d


def _resolve(patterns, tmp_path, *, recursive=False, output=None):
    # globs are written relative to tmp_path, so resolve from there
    return cli._resolve_recordings(
        [str(tmp_path / p) for p in patterns],
        recursive=recursive,
        config=_RES_CFG,
        output=output,
    )


def _rec_dir(rec):
    """The recording directory, recovered from its resolved footage (flat layout)."""
    files = next(iter(rec.sources.values()), [])
    return files[0].parent if files else None


def _rec_dirs(recordings):
    return [_rec_dir(rec) for rec in recordings]


def test_resolve_single_literal_recording(tmp_path):
    rec = _make_recording(tmp_path / "rec")
    out = _resolve(["rec"], tmp_path)
    # the per-camera footage is resolved up front and threaded to the run, alongside
    # the resolved output directory (the input dir is not retained).
    assert out[0].sources == {
        "cam0": [rec / "cam0.mp4"],
        "cam1": [rec / "cam1.mp4"],
    }
    assert out[0].outdir == rec / "deeperfly_outputs"


def test_resolve_single_literal_invalid_warns_but_keeps(tmp_path, caplog):
    """A single explicit path is kept even when it is not valid footage (so a resume
    from its cache still works), with a warning naming the resolved path."""
    bad = tmp_path / "nope"  # does not exist -> no footage
    with caplog.at_level("WARNING", logger="deeperfly"):
        out = _resolve(["nope"], tmp_path)
    # kept with empty footage; its output dir still resolves so a cached resume works.
    assert out[0].sources == {}
    assert out[0].outdir == cli._default_outdir(bad)
    assert any(
        str(bad.resolve()) in r.message and "not a valid recording" in r.message
        for r in caplog.records
    )


def test_resolve_image_sequence_recording(tmp_path):
    """A camera's footage may be a multi-file image sequence (natsorted)."""
    rec = _make_recording(tmp_path / "rec", ext="png", count=3)
    out = _resolve(["rec"], tmp_path)
    assert out[0].sources["cam0"] == [rec / f"cam0_{i}.png" for i in range(3)]


def test_resolve_partial_cameras_warns_and_skips(tmp_path, caplog):
    """A directory with footage for only some cameras is warned and skipped."""
    rec = tmp_path / "rec"
    rec.mkdir()
    (rec / "cam0.mp4").write_bytes(b"")  # cam1 missing
    with caplog.at_level("WARNING", logger="deeperfly"):
        out = _resolve(["rec"], tmp_path)
    assert out[0].sources == {}  # not a valid recording -> kept empty for resume
    assert any("missing ['cam1']" in r.message for r in caplog.records)


def test_resolve_uneven_file_count_warns_and_skips(tmp_path, caplog):
    """Image sequences with different file counts across cameras are skipped."""
    rec = tmp_path / "rec"
    rec.mkdir()
    for i in range(3):
        (rec / f"cam0_{i}.png").write_bytes(b"")
    (rec / "cam1_0.png").write_bytes(b"")  # only one frame for cam1
    with caplog.at_level("WARNING", logger="deeperfly"):
        out = _resolve(["rec"], tmp_path)
    assert out[0].sources == {}
    assert any("uneven file count" in r.message for r in caplog.records)


def test_resolve_wildcard_keeps_valid_skips_nonrecordings_quietly(tmp_path, caplog):
    """A wildcard keeps the valid recordings and silently drops its incidental
    non-recording matches (no warning while at least one is valid)."""
    _make_recording(tmp_path / "fly1")
    _make_recording(tmp_path / "fly2")
    (tmp_path / "deeperfly_outputs").mkdir()  # a non-recording the glob also matches
    with caplog.at_level("WARNING", logger="deeperfly"):
        out = _resolve(["*"], tmp_path)
    assert sorted(p.name for p in _rec_dirs(out)) == ["fly1", "fly2"]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_resolve_multiple_inputs_batch_filters_quietly(tmp_path, caplog):
    """Several inputs run as a batch: an invalid one is skipped without warning as
    long as another is valid (matches a shell-expanded wildcard)."""
    rec = _make_recording(tmp_path / "fly1")
    (tmp_path / "fly2").mkdir()  # exists but no footage
    with caplog.at_level("WARNING", logger="deeperfly"):
        out = _resolve(["fly1", "fly2"], tmp_path)
    assert _rec_dirs(out) == [rec]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_resolve_batch_no_valid_warns_and_errors(tmp_path, caplog):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()  # neither holds footage
    with caplog.at_level("WARNING", logger="deeperfly"):
        with pytest.raises(SystemExit, match="no valid recording"):
            _resolve(["*"], tmp_path)
    assert any("none of the inputs" in r.message for r in caplog.records)


def test_resolve_wildcard_no_match_warns_and_errors(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="deeperfly"):
        with pytest.raises(SystemExit):
            _resolve(["none*"], tmp_path)
    assert any("matched no paths" in r.message for r in caplog.records)


def test_resolve_recursive_multiple_and_glob_roots(tmp_path):
    """--recursive searches each (possibly wildcard) parent for nested recordings."""
    _make_recording(tmp_path / "expA" / "fly1")
    _make_recording(tmp_path / "expB" / "sub" / "fly2")
    out = _resolve(["exp*"], tmp_path, recursive=True)
    assert sorted(p.name for p in _rec_dirs(out)) == ["fly1", "fly2"]


def test_resolve_output_dir_handling(tmp_path):
    """Each recording's output directory is resolved up front into the Recording.

    No -o: each recording writes into its own <recording>/deeperfly_outputs. With -o
    and a single recording: that directory directly. With -o and a batch: a
    per-recording subdirectory so the runs don't overwrite each other.
    """
    _make_recording(tmp_path / "fly1")
    _make_recording(tmp_path / "fly2")
    (out,) = _resolve(["fly1"], tmp_path)
    assert out.outdir == tmp_path / "fly1" / "deeperfly_outputs"
    (out,) = _resolve(["fly1"], tmp_path, output=str(tmp_path / "results"))
    assert out.outdir == tmp_path / "results"
    batch = _resolve(["*"], tmp_path, output=str(tmp_path / "results"))
    assert {r.outdir for r in batch} == {
        tmp_path / "results" / "fly1",
        tmp_path / "results" / "fly2",
    }


def test_resolve_recursive_nondir_literal_warns_and_errors(tmp_path, caplog):
    with caplog.at_level("WARNING", logger="deeperfly"):
        with pytest.raises(SystemExit, match="no recordings"):
            _resolve(["ghost"], tmp_path, recursive=True)
    assert any("is not a directory" in r.message for r in caplog.records)


def test_resolve_recursive_no_recordings_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no recordings"):
        _resolve(["empty"], tmp_path, recursive=True)


def test_run_batch_multiple_inputs(tmp_path, monkeypatch):
    """Two input recordings run as a batch into per-recording output subdirs."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _stub_detect(
        monkeypatch, tmp_path
    )  # detection stubbed; footage only needs to exist
    _make_fly_recording(tmp_path / "r1")
    _make_fly_recording(tmp_path / "r2")
    out = tmp_path / "out"
    cli.main(
        [
            "run",
            str(tmp_path / "r1"),
            str(tmp_path / "r2"),
            "-c",
            str(cfg),
            "-o",
            str(out),
            "--log-level",
            "error",
        ]
    )
    assert (out / "r1" / "poses.h5").exists()
    assert (out / "r2" / "poses.h5").exists()


def test_disabled_pose2d_reuses_cached_2d(result, tmp_path, monkeypatch):
    """pose2d off reuses the cached 2D; triangulation on reconstructs 3D from it."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = true\ndo_visualization = false\n"
    )
    monkeypatch.setattr(
        cli, "_stage_pose2d", lambda *a, **k: pytest.fail("pose2d is off")
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
    assert PoseResult.load(outdir / "poses.h5").pts3d is not None


def test_triangulation_skipped_without_2d(tmp_path, caplog):
    """triangulation on but no 2D (pose2d off, no cached result) -> skip + reason."""
    outdir = tmp_path / "out"
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = true\ndo_visualization = false\n"
    )
    with caplog.at_level("WARNING"):
        cli.main(["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)])
    assert not (outdir / "poses.h5").exists()
    assert any(
        "skipping triangulation" in r.message and "no 2D pose" in r.message
        for r in caplog.records
    )


def test_smoothing_skipped_without_3d(result, tmp_path, caplog):
    """smoothing on but the cached result has no 3D -> skip + reason."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )  # 2D only
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_smoothing = true\ndo_visualization = false\n"
    )
    with caplog.at_level("WARNING"):
        cli.main(["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)])
    assert any(
        "skipping smoothing" in r.message and "no 3D pose" in r.message
        for r in caplog.records
    )


def test_visualization_only_from_cached_result(result, tmp_path, monkeypatch):
    """Every compute stage off, visualization on -> render from the cached 3D result."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")  # full result with 3D
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_visualization = true\n"
    )
    monkeypatch.setattr(cli, "_stage_pose2d", lambda *a, **k: pytest.fail("no pose2d"))
    monkeypatch.setattr(
        cli, "_stage_triangulation", lambda *a, **k: pytest.fail("no triangulation")
    )
    called: list[bool] = []
    monkeypatch.setattr(
        cli, "_stage_visualization", lambda *a, **k: called.append(True)
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
    assert called == [True]


def test_visualization_skipped_without_result(tmp_path, caplog):
    """visualization on but nothing to draw (no cached result) -> skip + reason."""
    outdir = tmp_path / "out"
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_visualization = true\n"
    )
    with caplog.at_level("WARNING"):
        cli.main(["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)])
    assert any(
        "skipping visualization" in r.message and "no pose result" in r.message
        for r in caplog.records
    )


def test_normalize_overwrite_argv():
    """`--overwrite` accepts bare (all) or space-separated stage names."""
    norm = cli._normalize_overwrite_argv
    # bare -> the "all" sentinel; only `run` is rewritten.
    assert norm(["run", "rec", "--overwrite"]) == [
        "run",
        "rec",
        "--overwrite",
        "__all__",
    ]
    assert norm(["inspect", "--overwrite"]) == ["inspect", "--overwrite"]
    # space-separated stage names -> the repeated form click understands; the
    # following positional / option (a non-stage token) ends the run of names.
    assert norm(["run", "rec", "--overwrite", "pose2d", "visualization"]) == [
        "run",
        "rec",
        "--overwrite",
        "pose2d",
        "--overwrite",
        "visualization",
    ]
    assert norm(["run", "--overwrite", "triangulation", "rec"]) == [
        "run",
        "--overwrite",
        "triangulation",
        "rec",
    ]


def test_overwrite_stages_parsing():
    assert cli._overwrite_stages(None) == set()
    assert cli._overwrite_stages([cli._OVERWRITE_ALL]) == set(cli.STAGES)
    assert cli._overwrite_stages(["pose2d", "smoothing"]) == {"pose2d", "smoothing"}
    with pytest.raises(SystemExit, match="unknown stage"):
        cli._overwrite_stages(["pose2d", "nope"])


def test_enabled_pose2d_reused_when_cached(result, tmp_path, monkeypatch):
    """An enabled stage whose output is already cached is reused, not recomputed."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")  # cached full result, n_frames=6
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _stub_detect(monkeypatch, tmp_path)
    monkeypatch.setattr(
        cli, "_stage_pose2d", lambda *a, **k: pytest.fail("pose2d should be reused")
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
    # the cache is untouched: nothing recomputed, so the n_frames=6 result stands.
    assert PoseResult.load(outdir / "poses.h5").n_frames == 6


def test_overwrite_pose2d_recomputes_over_cache(result, tmp_path, monkeypatch):
    """--overwrite forces an enabled stage to recompute over its cache."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")  # cached full result, n_frames=6
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    T = _stub_detect(monkeypatch, tmp_path)  # stub detects 3 frames
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--overwrite",
            "pose2d",
            "--log-level",
            "error",
        ]
    )
    res = PoseResult.load(outdir / "poses.h5")
    assert res.pts3d is None and res.pts2d.shape == (7, T, 38, 2)  # fresh 2D, not cache


def test_overwrite_cascades_to_downstream_stages(result, tmp_path, monkeypatch):
    """Overwriting an upstream stage also refreshes the enabled stages after it."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")  # cached full result with 3D, n_frames=6
    # pose2d + bundle_adjustment + triangulation on (visualization off); all cached.
    cfg = _default_cfg(tmp_path, visualization=False)
    T = _stub_detect(monkeypatch, tmp_path)  # fresh 2D has 3 frames
    triangulated: list[bool] = []
    real_triangulate = cli._stage_triangulation

    def spy_triangulate(config, res):
        triangulated.append(True)
        return real_triangulate(config, res)

    monkeypatch.setattr(cli, "_stage_bundle_adjustment", lambda config, res: res)
    monkeypatch.setattr(cli, "_stage_triangulation", spy_triangulate)
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--overwrite",
            "pose2d",
            "--log-level",
            "error",
        ]
    )
    # triangulation reran (cascade) even though it was not named, producing 3D over
    # the freshly detected 3-frame 2D rather than reusing the cached 6-frame 3D.
    assert triangulated == [True]
    res = PoseResult.load(outdir / "poses.h5")
    assert res.pts3d is not None and res.pts3d.shape[0] == T


def test_overwrite_bundle_adjustment_rebuilds_config_rig(result, tmp_path, monkeypatch):
    """--overwrite bundle_adjustment on a BA-refined cache re-runs BA from the
    *config* rig, not the already-refined cached cameras.

    The cached poses.h5 stores the previous BA *output*; feeding it back to BA as
    the starting point would begin at the prior optimum, so edited
    [pipeline.bundle_adjustment] params barely move it. The stage must instead
    rebuild the un-refined config rig (the regression behind "--overwrite does
    nothing" after editing the BA config).
    """
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(
        result.cameras,
        result.skeleton,
        result.pts2d,
        conf=result.conf,
        meta={"bundle_adjustment": True},  # cached cameras are a previous BA output
    ).save(outdir / "poses.h5")
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = true\n"
        "do_triangulation = false\ndo_visualization = false\n"
    )

    # a recognizable rig the config rebuild yields (distinct from the cached one).
    fresh = CameraGroup.from_arrays(
        result.cameras.names,
        result.cameras.rvecs,
        result.cameras.tvecs * 2.0,
        result.cameras.intrs,
        result.cameras.dists,
    )
    monkeypatch.setattr(cli, "_config_camera_rig", lambda args, config: fresh)
    monkeypatch.setattr(cli, "_stage_pose2d", lambda *a, **k: pytest.fail("pose2d off"))
    seen: dict = {}

    def spy_ba(config, res):
        seen["is_fresh"] = res.cameras is fresh
        return res

    monkeypatch.setattr(cli, "_stage_bundle_adjustment", spy_ba)
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--overwrite",
            "bundle_adjustment",
            "--log-level",
            "error",
        ]
    )
    assert seen["is_fresh"]  # BA started from the rebuilt config rig, not the cache


def test_recompute_bundle_adjustment_keeps_unrefined_cached_rig(
    result, tmp_path, monkeypatch
):
    """Cached cameras that were never BA-refined are legitimate input: a BA
    recompute keeps them and does not rebuild the config rig."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )  # no bundle_adjustment marker
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = true\n"
        "do_triangulation = false\ndo_visualization = false\n"
    )
    monkeypatch.setattr(
        cli,
        "_config_camera_rig",
        lambda args, config: pytest.fail("must not rebuild an un-refined rig"),
    )
    monkeypatch.setattr(cli, "_stage_pose2d", lambda *a, **k: pytest.fail("pose2d off"))
    seen: dict = {}

    def spy_ba(config, res):
        seen["is_cached"] = res.cameras.names == result.cameras.names
        return res

    monkeypatch.setattr(cli, "_stage_bundle_adjustment", spy_ba)
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--overwrite",
            "bundle_adjustment",
            "--log-level",
            "error",
        ]
    )
    assert seen["is_cached"]  # BA refined the stored cameras, no config rebuild


def test_overwrite_unknown_stage_errors(tmp_path):
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[pipeline]\ndo_pose2d = false\n")
    with pytest.raises(SystemExit, match="unknown stage"):
        cli.main(
            [
                "run",
                str(tmp_path / "rec"),
                "-c",
                str(cfg),
                "-o",
                str(tmp_path / "out"),
                "--overwrite=nope",
                "--log-level",
                "error",
            ]
        )


def test_existing_config_in_outdir_wins(result, tmp_path, caplog):
    """A config.toml in the output dir is used over -c, with a notification."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")  # full 3D result already present
    # the snapshot in outdir disables every stage, so the run is a quiet no-op ...
    (outdir / "config.toml").write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_visualization = false\n"
    )
    cfg = (
        tmp_path / "cfg.toml"
    )  # ... while -c would enable pose2d (and fail without it)
    cfg.write_text("[pipeline]\ndo_pose2d = true\n")
    with caplog.at_level("WARNING"):
        cli.main(["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)])
    assert any("ignoring -c" in r.message for r in caplog.records)


def test_verbose_logs_image_sizes_and_batch(tmp_path, monkeypatch, caplog):
    """The default (info) surfaces input image sizes and the detector forward
    batch / input size."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _stub_detect(monkeypatch, tmp_path)
    outdir = tmp_path / "out"
    with caplog.at_level("INFO"):
        cli.main(["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)])
    msgs = "\n".join(r.message for r in caplog.records)
    assert "input image sizes" in msgs
    assert "forward passes/frame" in msgs
    assert "network input 256x512" in msgs


# -- automatic weight provisioning -------------------------------------------


def test_load_detector_downloads_cached(tmp_path, monkeypatch):
    # With no explicit checkpoint, _load_detector downloads the cached torch
    # weights and loads the detector from them.
    from deeperfly.pose2d import backends, download

    sentinel = tmp_path / "sh8_deepfly.pth"
    monkeypatch.setattr(download, "download_torch_weights", lambda: sentinel)
    monkeypatch.setattr(backends, "load_detector", lambda path: ("loaded", path))
    assert cli._load_detector(None) == ("loaded", sentinel)


def test_load_detector_explicit_checkpoint(tmp_path, monkeypatch):
    from deeperfly.pose2d import backends, download

    monkeypatch.setattr(
        download,
        "download_torch_weights",
        lambda: pytest.fail("should not download when a checkpoint is given"),
    )
    monkeypatch.setattr(backends, "load_detector", lambda path: ("loaded", path))
    ckpt = tmp_path / "custom.pth"
    ckpt.write_bytes(b"weights")
    assert cli._load_detector(str(ckpt)) == ("loaded", str(ckpt))


def test_load_detector_missing_checkpoint_raises(tmp_path):
    with pytest.raises(SystemExit, match="no detector checkpoint"):
        cli._load_detector(str(tmp_path / "nope.pth"))


# -- doctor (installation / runtime report) ----------------------------------


def test_fmt_bytes_units():
    assert cli._fmt_bytes(512) == "512 B"
    assert cli._fmt_bytes(1536) == "1.5 KiB"
    assert cli._fmt_bytes(2 * 1024**3) == "2.0 GiB"


def test_doctor_reports_install_details(tmp_path, monkeypatch, capsys):
    """`deeperfly doctor` prints each section and reflects the weights cache.

    The weights cache is redirected to a temp dir with the detector checkpoint
    present, so the report shows it downloaded. COLUMNS is widened so rich does
    not wrap the lines we assert on.
    """
    from deeperfly.pose2d import download

    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(download, "cache_dir", lambda: tmp_path)
    (tmp_path / download.TORCH_WEIGHTS_NAME).write_bytes(b"x" * 2048)

    cli.main(["doctor"])
    out = capsys.readouterr().out

    for section in (
        "deeperfly",
        "system",
        "inference",
        "frame I/O backends",
        "weights",
        "config",
    ):
        assert section in out
    assert "video read" in out and "image read" in out  # the new I/O rows
    assert "GPU inference" in out and "detector" in out
    assert "downloaded" in out  # the detector checkpoint we created
    assert download.TORCH_WEIGHTS_NAME in out
    assert str(cli.DEFAULT_CONFIG_PATH) in out


# -- pictorial structures skipped (no candidates) on resume ------------------


@pytest.mark.parametrize("method", ["ransac", "greedy"])
def test_resume_pictorial_skipped_without_candidates(result, tmp_path, caplog, method):
    """Resuming with pictorial_structures on but pose2d off skips it (candidates
    are not cached); the triangulation stage still produces 3D."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_pictorial_structures = true\ndo_triangulation = true\ndo_visualization = false\n"
        f'[pipeline.triangulation]\nmethod = "{method}"\n'
    )
    with caplog.at_level("WARNING"):
        cli.main(["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir)])
    assert any(
        "skipping pictorial_structures" in r.message and "candidates" in r.message
        for r in caplog.records
    )
    assert PoseResult.load(outdir / "poses.h5").pts3d is not None


def test_resume_uses_stored_cameras_with_full_config(result, tmp_path):
    """Resuming the 3D stages uses the stored rig even when the config carries a
    [cameras] table (the packaged template's cameras have no explicit principal
    point, so rebuilding them without frame sizes would fail)."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    # full default config (with [cameras]), pose2d/visualization off so the rig is
    # never rebuilt -- bundle_adjustment + triangulation run on the stored cameras.
    cfg = _default_cfg(tmp_path, pose2d=False, visualization=False)
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
    res = PoseResult.load(outdir / "poses.h5")
    assert res.pts3d is not None
    assert res.cameras.names == result.cameras.names  # stored rig, not rebuilt


# -- detector progress hook --------------------------------------------------


def test_detect_sequence_progress_called_per_frame(monkeypatch):
    from deeperfly.pose2d import inference

    T = 4
    frames = [np.zeros((T, 8, 8, 3), np.uint8) for _ in range(2)]
    monkeypatch.setattr(
        inference, "detect", lambda *a, **k: (np.zeros((2, 38, 2)), np.zeros((2, 38)))
    )
    seen: list[int] = []

    def progress(it):
        for x in it:
            seen.append(x)
            yield x

    pts, conf = inference.detect_sequence(
        object(), frames, ["right", "left"], [False, True], progress=progress
    )
    assert seen == list(range(T))
    assert pts.shape == (2, T, 38, 2)


# -- view frame recovery on resume -------------------------------------------


def test_source_view_frames_source_priority(result, tmp_path, monkeypatch):
    from deeperfly import video

    # read_frames echoes its source so we can see which footage each view used.
    monkeypatch.setattr(video, "read_frames", lambda src, **kw: ("frames", src))
    cfg = Config.from_dict({"pipeline": {"pose2d": {}}})
    names = result.cameras.names
    v0, v1 = names[0], names[1]
    res = PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf)

    # 1) this run's already-resolved footage (args.sources) is used directly.
    got = cli._source_view_frames(
        argparse.Namespace(sources={v0: ["f0"], v1: ["f1"]}), cfg, res, [v0, v1]
    )
    assert got == {v0: ("frames", ["f0"]), v1: ("frames", ["f1"])}

    # 2) no run footage -> error telling the user to re-pass the recording.
    with pytest.raises(SystemExit, match="overwrite visualization"):
        cli._source_view_frames(argparse.Namespace(sources={}), cfg, res, [v0])

    # 3) in-memory frames (indexed by camera order) bypass the recording entirely.
    mem = [f"mem{i}" for i in range(len(names))]
    got = cli._source_view_frames(
        argparse.Namespace(sources=None), cfg, res, [names[2], v0], in_memory=mem
    )
    assert got == {names[2]: "mem2", v0: "mem0"}

    # 4) no imshow views -> nothing sourced.
    assert cli._source_view_frames(argparse.Namespace(sources=None), cfg, res, []) == {}


# -- per-camera [preprocess.*] frame transform wiring ------------------------


def test_source_view_frames_applies_preprocess_transform(result, tmp_path, monkeypatch):
    # Overlay footage must get the same per-camera transform the detector saw, so
    # 2D/3D overlays (now in transformed-frame coords) land on matching frames.
    from deeperfly import video

    rng = np.random.default_rng(0)
    names = result.cameras.names
    raw = {n: rng.integers(0, 256, (2, 4, 6, 3), np.uint8) for n in names}
    # the run's resolved footage maps each view to one "file" (its name); read_frames
    # returns that view's raw footage.
    monkeypatch.setattr(video, "read_frames", lambda src, **kw: raw[src[0]])

    v0, v1 = names[0], names[1]
    cameras = {n: {"input": n} for n in names}
    cameras[v0] = {"input": v0, "preprocess": {"rot90": 1, "fliplr": True}}
    cfg = Config.from_dict({"cameras": cameras, "pipeline": {"pose2d": {}}})
    args = argparse.Namespace(sources={n: [n] for n in names})
    got = cli._source_view_frames(args, cfg, result, [v0, v1])
    np.testing.assert_array_equal(
        got[v0], video.FrameTransform(rot90=1, fliplr=True).apply(raw[v0])
    )
    np.testing.assert_array_equal(got[v1], raw[v1])  # no table -> untouched


def test_prefetch_windows_applies_per_source_transform(monkeypatch):
    from deeperfly import video

    rng = np.random.default_rng(1)
    win = rng.integers(0, 256, (2, 4, 6, 3), np.uint8)  # one short block of 2 frames

    def fake_stream_frames(src, *, backend, image_backend, workers, block):
        yield win.copy()  # a single < block block -> last (and only) window

    monkeypatch.setattr(video, "stream_frames", fake_stream_frames)
    t = video.FrameTransform(fliplr=True, rot90=1)
    windows = list(
        cli._prefetch_windows(["camA"], backend="auto", block=8, transforms=[t])
    )
    assert len(windows) == 1
    window, n = windows[0]
    assert n == 2
    np.testing.assert_array_equal(window[0], t.apply(win))


def test_prefetch_windows_streams_multiple_blocks_then_stops(monkeypatch):
    from deeperfly import video

    # Two full blocks (block=2) then a short one ends the stream; two synced cameras
    # must stay aligned and concatenate in order.
    a = np.arange(5 * 2 * 2 * 3, dtype=np.uint8).reshape(5, 2, 2, 3)
    b = a + 100

    def fake_stream_frames(src, *, backend, image_backend, workers, block):
        full = a if src == "A" else b
        for pos in range(0, len(full), block):
            yield full[pos : pos + block]

    monkeypatch.setattr(video, "stream_frames", fake_stream_frames)
    windows = list(cli._prefetch_windows(["A", "B"], backend="auto", block=2))
    # 5 frames at block=2 -> blocks of 2, 2, 1; the short last block stops the stream.
    assert [n for _, n in windows] == [2, 2, 1]
    cam_a = np.concatenate([w[0] for w, _ in windows])
    cam_b = np.concatenate([w[1] for w, _ in windows])
    np.testing.assert_array_equal(cam_a, a)
    np.testing.assert_array_equal(cam_b, b)


def test_camera_image_sizes_uses_transformed_dims(monkeypatch):
    # A rot90 swaps H/W, so the inferred principal point must use the transformed
    # size; an untransformed camera keeps its raw (H, W).
    from deeperfly import video

    monkeypatch.setattr(
        cli, "_camera_sources", lambda root, config: [("rh", "A"), ("lf", "B")]
    )
    head = np.zeros((1, 4, 6, 3), np.uint8)
    monkeypatch.setattr(
        video,
        "read_frames",
        lambda src, indices=None, **kw: head,
    )
    cfg = Config.from_dict(
        {
            "cameras": {"rh": {"preprocess": {"rot90": 1}}, "lf": {}},
            "pipeline": {"pose2d": {}},
        }
    )
    sizes = cli._camera_image_sizes(argparse.Namespace(input="x"), cfg)
    assert sizes["rh"] == (6, 4)  # (H, W) swapped by the quarter-turn
    assert sizes["lf"] == (4, 6)  # identity
