"""Tests for the unified ``deeperfly run`` command and its plumbing.

Covers the per-stage flags (:meth:`Config.stage_flags`), config resolution
(:meth:`Config.read_for_run`: ``-c`` wins, else the output-dir snapshot, else
the packaged default), the fingerprint-driven stage execution -- an enabled
stage is reused while its config is unchanged, recomputes when it changed, and
is skipped with a reason when its input is missing -- the output-dir layout
(:func:`recordings.plan_outdirs`), automatic weight provisioning, the
pictorial-skipped notice on resume, the detector progress hook, and the overlay
frame-recovery order. The detector and frame I/O are stubbed so these need
neither real weights nor video files.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from deeperfly import Config, cli, pipeline, recordings
from deeperfly.cli.app import _normalize_overwrite_argv
from deeperfly.cli.report import _fmt_bytes
from deeperfly.config import DEFAULT_CONFIG_PATH, STAGE_DEFAULTS, STAGES
from deeperfly.pose2d import stream as pose2d_stream
from deeperfly.results import PoseResult

FLY_CAMERAS = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]


def _default_cfg(tmp_path, *, name="config.toml", **flags):
    """Write the packaged default config with the given ``do_<stage>`` flags flipped.

    Pass any subset of stage names (``pose2d``, ``bundle_adjustment``,
    ``pictorial_structures``, ``triangulation``, ``visualization``) as keyword
    booleans. The full default is needed by tests that run the ``pose2d``
    stage (it builds the camera rig from ``[cameras]``); tests that resume from a
    cached result use small hand-written configs instead.
    """
    text = DEFAULT_CONFIG_PATH.read_text()
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
    assert Config.from_dict({}).stage_flags() == STAGE_DEFAULTS
    assert Config.default().stage_flags() == {
        "pose2d": True,
        "bundle_adjustment": True,
        "pictorial_structures": False,
        "triangulation": True,
        "visualization": True,
    }


def test_stage_flags_toggles():
    flags = Config.from_dict(
        {"pipeline": {"do_pose2d": False, "do_pictorial_structures": True}}
    ).stage_flags()
    assert flags["pose2d"] is False and flags["pictorial_structures"] is True
    assert flags["triangulation"] is True  # untouched default


# -- config resolution -------------------------------------------------------


def test_resolve_config_default_when_no_cli_and_no_snapshot(tmp_path):
    config = Config.read_for_run(None, tmp_path)
    assert config.source == DEFAULT_CONFIG_PATH
    assert config.data == Config.default().data


def test_resolve_config_uses_cli_when_no_snapshot(tmp_path):
    cfg = tmp_path / "mine.toml"
    cfg.write_text("[pipeline]\ndo_pose2d = false\n")
    config = Config.read_for_run(str(cfg), tmp_path)
    assert config.source == cfg and config.data["pipeline"] == {"do_pose2d": False}


def test_resolve_config_cli_wins_over_snapshot(tmp_path):
    snapshot = tmp_path / "config.toml"
    snapshot.write_text("[pipeline]\ndo_triangulation = false\n")
    other = tmp_path / "other.toml"
    other.write_text("[pipeline]\ndo_triangulation = true\n")
    config = Config.read_for_run(str(other), tmp_path)
    assert config.source == other and config.data["pipeline"] == {
        "do_triangulation": True
    }


def test_resolve_config_snapshot_used_without_cli(tmp_path):
    snapshot = tmp_path / "config.toml"
    snapshot.write_text("[pipeline]\ndo_visualization = false\n")
    config = Config.read_for_run(None, tmp_path)
    assert config.source == snapshot and config.data["pipeline"] == {
        "do_visualization": False
    }


# -- run: stage toggles, caching, skips -------------------------------------


def _stub_detect(monkeypatch, tmp_path):
    """Stub frame sizing + detection so detect needs no files, weights or recording.

    Returns ``(T, calls)``: the stubbed frame count and a list appended to on
    every detection, so a test can assert whether pose2d recomputed or was
    reused from cache.
    """
    from types import SimpleNamespace

    T, H, W = 3, 16, 16
    calls: list[bool] = []
    # Per-source raw sizes (the default plan's 7 sources are vid_<view>).
    sizes = {f"vid_{view}": (H, W) for view in FLY_CAMERAS}
    monkeypatch.setattr(
        pipeline.stages, "source_image_sizes", lambda config, **kw: sizes
    )
    monkeypatch.setattr(
        pipeline.stages,
        "load_models",
        lambda plan: {
            "deepfly2d": SimpleNamespace(
                set_precision=lambda p: None, device=lambda: "cpu"
            )
        },
    )
    # The recording footage is stubbed away, so skip the pre-run footage validation a
    # real fresh pose2d run does (covered separately by the input-resolution tests).
    monkeypatch.setattr(
        pipeline.run, "require_input_footage", lambda config, **kw: None
    )

    def fake_detect_2d(config, plan, models, *, want_candidates=False, k=5, **kw):
        from deeperfly.pictorial import Candidates

        calls.append(True)
        v = len(FLY_CAMERAS)
        cand = None
        if want_candidates:
            cand = Candidates(
                xy=np.zeros((v, T, 38, k, 2)), score=np.ones((v, T, 38, k))
            )
        return np.zeros((v, T, 38, 2)), np.ones((v, T, 38)), cand

    monkeypatch.setattr(pipeline.stages, "detect_2d", fake_detect_2d)
    return T, calls


def _stub_compute_stages(monkeypatch):
    """Stub BA / triangulation / render with shape-correct no-ops.

    Returns ``(ba_calls, tri_calls)``: lists appended to (with the input rig /
    the triangulation params) on every call, so tests can assert which stages
    recomputed and with what.
    """
    ba_calls: list = []
    tri_calls: list = []

    def stub_ba(config, cameras, pts2d, conf, skeleton):
        ba_calls.append(cameras)
        return cameras

    def stub_tri(config, cameras, pts2d, conf=None):
        tri_calls.append(config.triangulation)
        return pts2d, np.zeros((pts2d.shape[1], 38, 3)), None

    monkeypatch.setattr(pipeline.stages, "stage_bundle_adjustment", stub_ba)
    monkeypatch.setattr(pipeline.stages, "stage_triangulation", stub_tri)
    monkeypatch.setattr(pipeline.stages, "render_videos", lambda *a, **k: None)
    return ba_calls, tri_calls


def _edit_snapshot(outdir, pattern, repl):
    """Regex-edit the config snapshot in ``outdir`` (asserting one substitution)."""
    snapshot = outdir / "config.toml"
    text, n = re.subn(pattern, repl, snapshot.read_text(), count=1, flags=re.M)
    assert n == 1, f"{pattern!r} not found in the snapshot"
    snapshot.write_text(text)


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
    T, _ = _stub_detect(monkeypatch, tmp_path)

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
    _stub_compute_stages(monkeypatch)
    outdir = tmp_path / "out"
    rec = tmp_path / "rec"
    cli.main(["run", str(rec), "-o", str(outdir), "--log-level", "error"])
    assert (outdir / "poses.h5").exists()
    assert (outdir / "config.toml").read_text() == DEFAULT_CONFIG_PATH.read_text()


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
    """A minimal but complete two-source/two-view plan that reads real footage."""
    from deeperfly.skeleton import Skeleton

    names = Skeleton.fly().point_names

    def point_sources(view, pathway):
        lines = [f"[point_sources.{view}]"]
        lines += [
            f'{names[i]} = {{ pathway = "{pathway}", out_channel = {i} }}'
            for i in range(19)
        ]
        return "\n".join(lines) + "\n"

    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        '[[sources]]\nname = "cam0"\n[[sources]]\nname = "cam1"\n'
        '[[preprocessors]]\nname = "plain"\nops = []\n'
        '[[models]]\nname = "m"\nclass = "hourglass"\n'
        "input_size = [256, 512]\nn_out_channels = 19\n"
        '[[pathways]]\nname = "p0"\nsource = "cam0"\npreprocessor = "plain"\nmodel = "m"\n'
        '[[pathways]]\nname = "p1"\nsource = "cam1"\npreprocessor = "plain"\nmodel = "m"\n'
        "[cameras.cam0]\nazimuth_deg = 0\ndistance = 10\nfocal_length_px = 100\n"
        "[cameras.cam1]\nazimuth_deg = 90\ndistance = 10\nfocal_length_px = 100\n"
        + point_sources("cam0", "p0")
        + point_sources("cam1", "p1")
        + "[pipeline]\ndo_pose2d = true\ndo_bundle_adjustment = false\n"
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

_RES_CFG = Config.from_dict(
    {
        "sources": [{"name": "cam0"}, {"name": "cam1"}],
        "cameras": {"cam0": {}, "cam1": {}},
    }
)


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
    # globs are written relative to tmp_path, so resolve from there. Discovery and
    # output planning are separate steps; zip them back into Recordings as the CLI
    # does (deeperfly.cli.run._cmd_run).
    found = recordings.resolve_recordings(
        [str(tmp_path / p) for p in patterns],
        recursive=recursive,
        config=_RES_CFG,
    )
    plan = recordings.plan_outdirs([d for d, _ in found], output)
    return [
        recordings.Recording(src, out) for (_, src), out in zip(found, plan.outdirs)
    ]


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
    assert out[0].outdir == recordings.default_outdir(bad)
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
    """Each recording's output directory is planned up front (plan_outdirs).

    No -o: each recording writes into its own <recording>/deeperfly_outputs. With -o
    and a single recording: that directory directly. With an absolute -o and a
    batch: a per-recording subdirectory under it ("collect") so the runs don't
    overwrite each other.
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


def test_plan_outdirs_batch_no_output_uses_defaults(tmp_path):
    fly1 = _make_recording(tmp_path / "fly1")
    fly2 = _make_recording(tmp_path / "fly2")
    batch = _resolve(["*"], tmp_path)
    assert {r.outdir for r in batch} == {
        fly1 / "deeperfly_outputs",
        fly2 / "deeperfly_outputs",
    }


def test_plan_outdirs_trailing_slash_collects(tmp_path):
    """A batch -o ending in '/' collects one subdirectory per recording name."""
    _make_recording(tmp_path / "a" / "rec1")
    _make_recording(tmp_path / "b" / "rec2")
    batch = _resolve(["a/rec1", "b/rec2"], tmp_path, output=str(tmp_path / "out") + "/")
    assert [r.outdir for r in batch] == [
        tmp_path / "out" / "rec1",
        tmp_path / "out" / "rec2",
    ]


def test_plan_outdirs_relative_no_slash_nests_inside_each_recording(tmp_path):
    """A batch with a relative -o (no trailing slash) writes <recording>/<o>,
    generalizing the default (which is effectively -o deeperfly_outputs)."""
    fly1 = _make_recording(tmp_path / "fly1")
    fly2 = _make_recording(tmp_path / "fly2")
    batch = _resolve(["fly1", "fly2"], tmp_path, output="myrun")
    assert [r.outdir for r in batch] == [fly1 / "myrun", fly2 / "myrun"]


def test_plan_outdirs_collision_mirrors_from_common_ancestor(tmp_path):
    """Colliding recording names under a collect -o fall back to mirroring the
    input paths from their common ancestor, pending confirmation."""
    a = _make_recording(tmp_path / "a" / "rec")
    b = _make_recording(tmp_path / "b" / "rec")
    plan = recordings.plan_outdirs([a, b], str(tmp_path / "out") + "/")
    assert plan.outdirs == [
        tmp_path / "out" / "a" / "rec",
        tmp_path / "out" / "b" / "rec",
    ]
    assert plan.mirror_confirm is not None
    assert "rec" in plan.mirror_confirm and str(tmp_path) in plan.mirror_confirm
    # without a collision there is nothing to confirm
    assert recordings.plan_outdirs([a], str(tmp_path / "out")).mirror_confirm is None


def test_run_batch_collision_errors_noninteractively(tmp_path, monkeypatch):
    """A collision fallback cannot be confirmed without a TTY: the run aborts up
    front, before any output dir is created."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _stub_detect(monkeypatch, tmp_path)
    _make_fly_recording(tmp_path / "a" / "rec")
    _make_fly_recording(tmp_path / "b" / "rec")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="collide"):
        cli.main(
            [
                "run",
                str(tmp_path / "a" / "rec"),
                str(tmp_path / "b" / "rec"),
                "-c",
                str(cfg),
                "-o",
                str(out),  # absolute, no trailing slash -> collect -> collision
                "--log-level",
                "error",
            ]
        )
    assert not out.exists()


def test_run_batch_collision_confirmed_mirrors(tmp_path, monkeypatch):
    """Confirming the collision fallback runs the batch into mirrored outdirs."""
    import sys

    import typer

    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _stub_detect(monkeypatch, tmp_path)
    _make_fly_recording(tmp_path / "a" / "rec")
    _make_fly_recording(tmp_path / "b" / "rec")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(typer, "confirm", lambda *a, **k: True)
    cli.main(
        [
            "run",
            str(tmp_path / "a" / "rec"),
            str(tmp_path / "b" / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(tmp_path / "out"),
            "--log-level",
            "error",
        ]
    )
    assert (tmp_path / "out" / "a" / "rec" / "poses.h5").exists()
    assert (tmp_path / "out" / "b" / "rec" / "poses.h5").exists()


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
        pipeline.stages, "stage_pose2d", lambda *a, **k: pytest.fail("pose2d is off")
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


def test_visualization_only_from_cached_result(result, tmp_path, monkeypatch):
    """Every compute stage off, visualization on -> render from the cached result.

    With the 3D stages disabled their cached output is not drawn (a derived
    stage feeds downstream only while enabled), so this renders the 2D pose."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")  # full result with 3D
    cfg = tmp_path / "cfg.toml"
    cfg.write_text(
        "[pipeline]\ndo_pose2d = false\ndo_bundle_adjustment = false\n"
        "do_triangulation = false\ndo_visualization = true\n"
    )
    monkeypatch.setattr(
        pipeline.stages, "stage_pose2d", lambda *a, **k: pytest.fail("no pose2d")
    )
    monkeypatch.setattr(
        pipeline.stages,
        "stage_triangulation",
        lambda *a, **k: pytest.fail("no triangulation"),
    )
    called: list[bool] = []
    monkeypatch.setattr(
        pipeline.stages, "render_videos", lambda *a, **k: called.append(True)
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
    norm = _normalize_overwrite_argv
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
    assert pipeline.overwrite_stages(None) == set()
    assert pipeline.overwrite_stages([pipeline._OVERWRITE_ALL]) == set(STAGES)
    assert pipeline.overwrite_stages(["pose2d", "visualization"]) == {
        "pose2d",
        "visualization",
    }
    with pytest.raises(SystemExit, match="unknown stage"):
        pipeline.overwrite_stages(["pose2d", "nope"])


def _run_args(tmp_path, cfg=None, *, extra=(), log_level="error"):
    """The CLI argv for one stubbed run against <tmp_path>/out.

    ``cfg=None`` omits ``-c`` (the run picks up the snapshot in the output dir).
    Tests asserting on log records must pass a permissive ``log_level`` (the CLI
    reconfigures the logger level on every invocation).
    """
    args = ["run", str(tmp_path / "rec")]
    if cfg is not None:
        args += ["-c", str(cfg)]
    args += ["-o", str(tmp_path / "out"), *extra, "--log-level", log_level]
    return args


def test_fresh_run_logs_running_not_recomputing(tmp_path, monkeypatch, caplog):
    """A first run over an empty output dir announces plain "running <stage>" --
    there is no cached result, so nothing is being "recomputed"."""
    cfg = _default_cfg(tmp_path, bundle_adjustment=False, visualization=False)
    _stub_detect(monkeypatch, tmp_path)
    _stub_compute_stages(monkeypatch)
    with caplog.at_level("INFO", logger="deeperfly"):
        cli.main(_run_args(tmp_path, cfg, log_level="info"))
    assert not any("recomputing" in r.message for r in caplog.records)
    for stage in ("pose2d", "triangulation"):
        assert any(f"running {stage}" == r.message for r in caplog.records)


def test_enabled_pose2d_reused_when_cached(tmp_path, monkeypatch):
    """An enabled stage whose config is unchanged reuses its cache on re-run."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _, calls = _stub_detect(monkeypatch, tmp_path)
    cli.main(_run_args(tmp_path, cfg))
    assert len(calls) == 1
    cli.main(_run_args(tmp_path, cfg))  # identical re-run -> nothing recomputes
    assert len(calls) == 1


def test_overwrite_pose2d_recomputes_over_cache(tmp_path, monkeypatch):
    """--overwrite forces an enabled stage to recompute over a valid cache."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _, calls = _stub_detect(monkeypatch, tmp_path)
    cli.main(_run_args(tmp_path, cfg))
    cli.main(_run_args(tmp_path, cfg, extra=("--overwrite", "pose2d")))
    assert len(calls) == 2


def test_overwrite_cascades_to_downstream_stages(tmp_path, monkeypatch):
    """Overwriting an upstream stage also refreshes the enabled stages after it."""
    cfg = _default_cfg(tmp_path, visualization=False)
    _, calls = _stub_detect(monkeypatch, tmp_path)
    ba_calls, tri_calls = _stub_compute_stages(monkeypatch)
    cli.main(_run_args(tmp_path, cfg))
    assert (len(calls), len(ba_calls), len(tri_calls)) == (1, 1, 1)
    cli.main(_run_args(tmp_path, cfg, extra=("--overwrite", "pose2d")))
    # bundle_adjustment and triangulation reran (cascade) though not named.
    assert (len(calls), len(ba_calls), len(tri_calls)) == (2, 2, 2)


def test_config_change_recomputes_only_affected_stages(tmp_path, monkeypatch, caplog):
    """Editing a stage's params re-runs it (and downstream) while the slow pose2d
    cache is reused -- no --overwrite needed."""
    cfg = _default_cfg(tmp_path, bundle_adjustment=False, visualization=False)
    _, calls = _stub_detect(monkeypatch, tmp_path)
    _, tri_calls = _stub_compute_stages(monkeypatch)
    cli.main(_run_args(tmp_path, cfg))
    assert (len(calls), len(tri_calls)) == (1, 1)
    assert tri_calls[0].ransac_threshold == 15.0

    _edit_snapshot(
        tmp_path / "out", r"^ransac_threshold = 15\.0", "ransac_threshold = 9.0"
    )
    with caplog.at_level("INFO", logger="deeperfly"):
        # no -c -> the edited snapshot drives the run
        cli.main(_run_args(tmp_path, log_level="info"))
    assert len(calls) == 1  # pose2d reused
    assert len(tri_calls) == 2 and tri_calls[1].ransac_threshold == 9.0
    assert any("reusing cached pose2d" in r.message for r in caplog.records)
    assert any(
        "recomputing triangulation" in r.message and "ransac_threshold" in r.message
        for r in caplog.records
    )


def test_perf_only_knobs_do_not_invalidate(tmp_path, monkeypatch):
    """batch_size / decode_buffer are performance-only: editing them reuses every
    cache."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _, calls = _stub_detect(monkeypatch, tmp_path)
    cli.main(_run_args(tmp_path, cfg))
    _edit_snapshot(tmp_path / "out", r"^batch_size = 16", "batch_size = 2")
    _edit_snapshot(tmp_path / "out", r"^decode_buffer = 4", "decode_buffer = 8")
    cli.main(_run_args(tmp_path))  # no -c -> the edited snapshot drives
    assert len(calls) == 1  # nothing recomputed


def test_pose2d_param_change_recomputes_with_loud_warning(
    tmp_path, monkeypatch, caplog
):
    """Editing a result-affecting pose2d param recomputes the slow stage with a
    WARNING naming exactly what changed."""
    cfg = _default_cfg(
        tmp_path, bundle_adjustment=False, triangulation=False, visualization=False
    )
    _, calls = _stub_detect(monkeypatch, tmp_path)
    cli.main(_run_args(tmp_path, cfg))
    _edit_snapshot(
        tmp_path / "out", r'^precision = "bfloat16"', 'precision = "float32"'
    )
    with caplog.at_level("WARNING", logger="deeperfly"):
        cli.main(_run_args(tmp_path, log_level="warning"))
    assert len(calls) == 2
    assert any(
        "recomputing pose2d" in r.message
        and "precision" in r.message
        and "float32" in r.message
        and r.levelname == "WARNING"
        for r in caplog.records
    )


def test_enable_pictorial_later_redetects_candidates(tmp_path, monkeypatch, caplog):
    """Enabling pictorial_structures after a run re-detects pose2d (candidates are
    extracted during detection); a pictorial param tweak afterwards reuses the
    cached candidates without re-detecting."""
    cfg = _default_cfg(tmp_path, bundle_adjustment=False, visualization=False)
    _, calls = _stub_detect(monkeypatch, tmp_path)
    _, tri_calls = _stub_compute_stages(monkeypatch)
    ps_calls: list[float] = []

    def stub_ps(config, cameras, skeleton, candidates, pts2d):
        assert candidates is not None
        ps_calls.append(config.pictorial.lam)
        return pts2d, np.zeros((pts2d.shape[1], 38, 3)), None

    monkeypatch.setattr(pipeline.stages, "stage_pictorial_structures", stub_ps)

    cli.main(_run_args(tmp_path, cfg))
    assert (len(calls), len(ps_calls)) == (1, 0)  # pictorial off by default

    _edit_snapshot(
        tmp_path / "out",
        r"^do_pictorial_structures = false",
        "do_pictorial_structures = true",
    )
    with caplog.at_level("WARNING", logger="deeperfly"):
        cli.main(_run_args(tmp_path, log_level="warning"))
    # pose2d re-detected to extract candidates, loudly; pictorial then ran.
    assert (len(calls), len(ps_calls)) == (2, 1)
    assert any(
        "recomputing pose2d" in r.message and "candidates" in r.message
        for r in caplog.records
    )

    _edit_snapshot(tmp_path / "out", r"^lam = 1\.0", "lam = 2.0")
    cli.main(_run_args(tmp_path))
    # only pictorial (and downstream) reran, from the cached candidates.
    assert (len(calls), len(ps_calls)) == (2, 2)
    assert ps_calls[1] == 2.0


def test_disable_pictorial_later_keeps_pose2d(tmp_path, monkeypatch):
    """Disabling pictorial_structures again does NOT re-detect pose2d (subset
    fingerprint rule), but triangulation re-runs on the raw pose2d points."""
    cfg = _default_cfg(
        tmp_path,
        pictorial_structures=True,
        bundle_adjustment=False,
        visualization=False,
    )
    _, calls = _stub_detect(monkeypatch, tmp_path)
    _, tri_calls = _stub_compute_stages(monkeypatch)
    monkeypatch.setattr(
        pipeline.stages,
        "stage_pictorial_structures",
        lambda config, cameras, skeleton, candidates, pts2d: (
            pts2d,
            np.zeros((pts2d.shape[1], 38, 3)),
            None,
        ),
    )
    cli.main(_run_args(tmp_path, cfg))
    assert (len(calls), len(tri_calls)) == (1, 1)

    _edit_snapshot(
        tmp_path / "out",
        r"^do_pictorial_structures = true",
        "do_pictorial_structures = false",
    )
    cli.main(_run_args(tmp_path))
    assert len(calls) == 1  # pose2d cache (with candidates) still valid
    assert len(tri_calls) == 2  # its 2D source flipped pictorial -> pose2d


def test_bundle_adjustment_always_starts_from_config_rig(tmp_path, monkeypatch):
    """A BA recompute starts from the un-refined config rig, never the prior BA
    output -- so edited BA params actually move the solution."""
    cfg = _default_cfg(tmp_path, triangulation=False, visualization=False)
    _stub_detect(monkeypatch, tmp_path)
    seen: list = []

    def spy_ba(config, cameras, pts2d, conf, skeleton):
        seen.append(cameras)
        # return a recognizably different rig (the "refined" output)
        from deeperfly.cameras import CameraGroup

        return CameraGroup.from_arrays(
            cameras.names,
            cameras.rvecs,
            cameras.tvecs * 2.0,
            cameras.intrs,
            cameras.dists,
        )

    monkeypatch.setattr(pipeline.stages, "stage_bundle_adjustment", spy_ba)
    cli.main(_run_args(tmp_path, cfg))
    cli.main(_run_args(tmp_path, cfg, extra=("--overwrite", "bundle_adjustment")))
    assert len(seen) == 2
    # the second run's input rig equals the first's (the config rig), not the
    # doubled-tvec output the first run cached.
    np.testing.assert_allclose(seen[1].tvecs, seen[0].tvecs)


def test_camera_geometry_edit_invalidates_bundle_adjustment(tmp_path, monkeypatch):
    """Editing [cameras] geometry re-runs BA (its fingerprint embeds the rig) while
    the pose2d cache -- which only depends on footage patterns/preprocess -- is
    reused."""
    cfg = _default_cfg(tmp_path, triangulation=False, visualization=False)
    _, calls = _stub_detect(monkeypatch, tmp_path)
    ba_calls, _ = _stub_compute_stages(monkeypatch)
    cli.main(_run_args(tmp_path, cfg))
    assert (len(calls), len(ba_calls)) == (1, 1)
    _edit_snapshot(tmp_path / "out", r"^distance = 107\.463", "distance = 99.0")
    cli.main(_run_args(tmp_path))
    assert len(calls) == 1  # pose2d reused
    assert len(ba_calls) == 2  # BA recomputed from the edited rig


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


def test_cli_config_wins_and_refreshes_snapshot(tmp_path, monkeypatch):
    """-c drives the run even when a snapshot exists, and refreshes the snapshot."""
    cfg1 = _default_cfg(
        tmp_path,
        name="cfg1.toml",
        bundle_adjustment=False,
        triangulation=False,
        visualization=False,
    )
    cfg2 = _default_cfg(
        tmp_path,
        name="cfg2.toml",
        bundle_adjustment=False,
        triangulation=True,
        visualization=False,
    )
    _stub_detect(monkeypatch, tmp_path)
    _stub_compute_stages(monkeypatch)
    cli.main(_run_args(tmp_path, cfg1))
    assert (tmp_path / "out" / "config.toml").read_text() == cfg1.read_text()
    cli.main(_run_args(tmp_path, cfg2))
    assert (tmp_path / "out" / "config.toml").read_text() == cfg2.read_text()
    # ... and the -c'd triangulation toggle actually drove the run
    assert PoseResult.load(tmp_path / "out" / "poses.h5").pts3d is not None


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
    assert "source image sizes" in msgs
    assert "pathways" in msgs
    assert "forward batch" in msgs


# -- automatic weight provisioning -------------------------------------------


def test_model_load_downloads_cached(tmp_path, monkeypatch):
    # With no explicit weights, an hourglass model downloads the cached torch
    # weights and loads the detector from them.
    from deeperfly.pose2d import detector, download, models
    from deeperfly.pose2d.models import ModelSpec

    sentinel = tmp_path / "sh8_deepfly.pth"
    monkeypatch.setattr(download, "download_torch_weights", lambda: sentinel)
    monkeypatch.setattr(detector, "load_detector", lambda path: ("loaded", path))
    lm = models.load_model(ModelSpec(name="m", cls="hourglass", weights=None))
    assert lm.module == ("loaded", sentinel)


def test_model_load_explicit_weights(tmp_path, monkeypatch):
    from deeperfly.pose2d import detector, download, models
    from deeperfly.pose2d.models import ModelSpec

    monkeypatch.setattr(
        download,
        "download_torch_weights",
        lambda: pytest.fail("should not download when weights are given"),
    )
    monkeypatch.setattr(detector, "load_detector", lambda path: ("loaded", path))
    ckpt = tmp_path / "custom.pth"
    ckpt.write_bytes(b"weights")
    lm = models.load_model(ModelSpec(name="m", cls="hourglass", weights=str(ckpt)))
    assert lm.module == ("loaded", str(ckpt))


def test_model_load_missing_weights_raises(tmp_path):
    from deeperfly.pose2d import models
    from deeperfly.pose2d.models import ModelSpec

    with pytest.raises(SystemExit, match="no detector checkpoint"):
        models.load_model(
            ModelSpec(name="m", cls="hourglass", weights=str(tmp_path / "nope.pth"))
        )


# -- doctor (installation / runtime report) ----------------------------------


def test_fmt_bytes_units():
    assert _fmt_bytes(512) == "512 B"
    assert _fmt_bytes(1536) == "1.5 KiB"
    assert _fmt_bytes(2 * 1024**3) == "2.0 GiB"


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
        "frame I/O",
        "weights",
        "config",
    ):
        assert section in out
    assert "video read" in out and "image read" in out  # the new I/O rows
    assert "GPU inference" in out and "detector" in out
    assert "downloaded" in out  # the detector checkpoint we created
    assert download.TORCH_WEIGHTS_NAME in out
    assert str(DEFAULT_CONFIG_PATH) in out


# -- pictorial structures skipped (no candidates) on resume ------------------


@pytest.mark.parametrize("method", ["ransac", "greedy"])
def test_resume_pictorial_skipped_without_candidates(result, tmp_path, caplog, method):
    """Resuming with pictorial_structures on but pose2d off (and no candidates in
    the cached 2D result) skips it with a reason; the triangulation stage still
    produces 3D. With pose2d *on*, the candidates clause of its fingerprint would
    have re-detected instead."""
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
    from types import SimpleNamespace

    import torch

    from deeperfly.pose2d import inference
    from deeperfly.preprocessing import FrameTransform

    T = 4

    # A fake model that resizes to a tiny input and emits a flat peak per channel,
    # so the run needs no real weights -- we only check the progress ticks.
    class _FakeModel:
        input_size = (4, 4)

        def device(self):
            return "cpu"

        def prepare(self, frames):
            t = torch.as_tensor(np.asarray(frames)).float()
            return torch.zeros((*t.shape[:-3], 3, 4, 4))

        def predict_points(self, inputs, *, method="weighted", radius=2):
            lead = tuple(inputs.shape[:-3])  # (B, V) of the standard (B, V, 3, H, W)
            return np.zeros((*lead, 19, 2), np.float32), np.ones(
                (*lead, 19), np.float32
            )

    def _pw(view):
        return SimpleNamespace(
            source="s",
            model="m",
            transform=FrameTransform(()),
            mapping=np.array([[i, view, i] for i in range(19)]),
        )

    plan = SimpleNamespace(n_views=2, n_points=38, pathways=[_pw(0), _pw(1)])
    models = {"m": _FakeModel()}
    windows = {"s": np.zeros((T, 8, 8, 3), np.uint8)}
    seen: list[int] = []

    def progress(it):
        for x in it:
            seen.append(x)
            yield x

    pts, conf = inference.detect_sequence(plan, models, windows, progress=progress)
    assert seen == list(range(T))
    assert pts.shape == (2, T, 38, 2)


# -- view frame recovery on resume -------------------------------------------


def test_source_view_frames_source_priority(result, tmp_path, monkeypatch):
    from deeperfly import io

    # open_reader()[key] echoes its source so we see which footage each view used.
    class _FakeReader:
        def __init__(self, src):
            self._src = src

        def __getitem__(self, key):
            return ("frames", self._src)

    monkeypatch.setattr(io, "open_reader", lambda src, **kw: _FakeReader(src))
    cfg = Config.from_dict({"pipeline": {"pose2d": {}}})
    names = result.cameras.names
    v0, v1 = names[0], names[1]
    res = PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf)

    # 1) this run's already-resolved footage (sources) is used directly.
    got = pipeline.source_view_frames(
        cfg, res, [v0, v1], sources={v0: ["f0"], v1: ["f1"]}
    )
    assert got == {v0: ("frames", ["f0"]), v1: ("frames", ["f1"])}

    # 2) no run footage -> error telling the user to re-pass the recording.
    with pytest.raises(SystemExit, match="original frames"):
        pipeline.source_view_frames(cfg, res, [v0], sources={})

    # 3) in-memory frames (indexed by camera order) bypass the recording entirely.
    mem = [f"mem{i}" for i in range(len(names))]
    got = pipeline.source_view_frames(
        cfg, res, [names[2], v0], sources=None, in_memory=mem
    )
    assert got == {names[2]: "mem2", v0: "mem0"}

    # 4) no imshow views -> nothing sourced.
    assert pipeline.source_view_frames(cfg, res, [], sources=None) == {}


def test_prefetch_windows_applies_per_source_transform(monkeypatch):
    from types import SimpleNamespace

    from deeperfly import io, preprocessing

    rng = np.random.default_rng(1)
    win = rng.integers(0, 256, (2, 4, 6, 3), np.uint8)  # one short block of 2 frames

    def fake_open_reader(src, **kw):
        def stream_blocks(*, block_size):
            yield win.copy()  # a single < block block -> last (and only) window

        return SimpleNamespace(stream_blocks=stream_blocks)

    monkeypatch.setattr(io, "open_reader", fake_open_reader)
    t = preprocessing.FrameTransform((preprocessing.Fliplr(), preprocessing.Rot90(k=1)))
    windows = list(pose2d_stream.prefetch_windows(["camA"], block=8, transforms=[t]))
    assert len(windows) == 1
    window, n = windows[0]
    assert n == 2
    np.testing.assert_array_equal(window[0], t.apply(win))


def test_prefetch_windows_streams_multiple_blocks_then_stops(monkeypatch):
    from types import SimpleNamespace

    from deeperfly import io

    # Two full blocks (block=2) then a short one ends the stream; two synced cameras
    # must stay aligned and concatenate in order.
    a = np.arange(5 * 2 * 2 * 3, dtype=np.uint8).reshape(5, 2, 2, 3)
    b = a + 100

    def fake_open_reader(src, **kw):
        full = a if src == "A" else b

        def stream_blocks(*, block_size):
            for pos in range(0, len(full), block_size):
                yield full[pos : pos + block_size]

        return SimpleNamespace(stream_blocks=stream_blocks)

    monkeypatch.setattr(io, "open_reader", fake_open_reader)
    windows = list(pose2d_stream.prefetch_windows(["A", "B"], block=2))
    # 5 frames at block=2 -> blocks of 2, 2, 1; the short last block stops the stream.
    assert [n for _, n in windows] == [2, 2, 1]
    cam_a = np.concatenate([w[0] for w, _ in windows])
    cam_b = np.concatenate([w[1] for w, _ in windows])
    np.testing.assert_array_equal(cam_a, a)
    np.testing.assert_array_equal(cam_b, b)


def test_source_image_sizes_returns_raw_dims(monkeypatch):
    # Sizes are the *raw* footage dims per source -- a view's intrinsics describe
    # its source's raw frame.
    from deeperfly import io

    monkeypatch.setattr(
        recordings,
        "source_sources",
        lambda config, **kw: [("cam0", "A"), ("cam1", "B")],
    )
    head = np.zeros((1, 4, 6, 3), np.uint8)

    class _FakeReader:
        def __getitem__(self, key):
            return head

    monkeypatch.setattr(io, "open_reader", lambda src, **kw: _FakeReader())
    cfg = Config.from_dict(
        {"sources": [{"name": "cam0"}, {"name": "cam1"}], "pipeline": {"pose2d": {}}}
    )
    sizes = recordings.source_image_sizes(cfg, input="x")
    assert sizes["cam0"] == (4, 6)  # raw (H, W)
    assert sizes["cam1"] == (4, 6)
