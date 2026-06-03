"""Tests for the unified ``deeperfly run`` command and its plumbing.

Covers the per-stage flags (:func:`cli._stage_flags`), config resolution (the
output-dir snapshot wins over ``-c``, else ``-c``, else the packaged default), the
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

from deeperfly import cli
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
    assert cli._stage_flags({}) == cli.STAGE_DEFAULTS
    assert cli._stage_flags(cli._load_config(cli.DEFAULT_CONFIG_PATH)) == {
        "pose2d": True,
        "bundle_adjustment": True,
        "pictorial_structures": False,
        "triangulation": True,
        "smoothing": False,
        "visualization": True,
    }


def test_stage_flags_toggles_and_validates():
    flags = cli._stage_flags({"pipeline": {"do_pose2d": False, "do_smoothing": True}})
    assert flags["pose2d"] is False and flags["smoothing"] is True
    assert flags["triangulation"] is True  # untouched default
    with pytest.raises(SystemExit, match="unknown stage toggle"):
        cli._stage_flags({"pipeline": {"do_detekt": True}})


def test_stage_flags_rejects_removed_keys():
    """Old [stages] / [pipeline] keys fail with a pointer to the new location."""
    with pytest.raises(SystemExit, match="do_bundle_adjustment"):
        cli._stage_flags({"pipeline": {"calibrate": True}})
    with pytest.raises(SystemExit, match="do_visualization"):
        cli._stage_flags({"pipeline": {"do_visualize": True}})
    with pytest.raises(SystemExit, match=r"triangulation"):
        cli._stage_flags({"pipeline": {"triangulation": "ransac"}})
    with pytest.raises(SystemExit, match="stages"):
        cli._stage_flags({"stages": {"detect": True}})
    # the new [pipeline.triangulation] sub-table is *not* mistaken for the removed
    # scalar key.
    assert cli._stage_flags({"pipeline": {"triangulation": {"method": "dlt"}}})


# -- config resolution -------------------------------------------------------


def test_resolve_config_default_when_no_cli_and_no_snapshot(tmp_path):
    config, path = cli._resolve_config(None, tmp_path)
    assert path == cli.DEFAULT_CONFIG_PATH
    assert config == cli._load_config(cli.DEFAULT_CONFIG_PATH)


def test_resolve_config_uses_cli_when_no_snapshot(tmp_path):
    cfg = tmp_path / "mine.toml"
    cfg.write_text("[pipeline]\ndo_pose2d = false\n")
    config, path = cli._resolve_config(str(cfg), tmp_path)
    assert path == cfg and config["pipeline"] == {"do_pose2d": False}


def test_resolve_config_snapshot_wins_over_cli_and_notifies(tmp_path, caplog):
    snapshot = tmp_path / "config.toml"
    snapshot.write_text("[pipeline]\ndo_triangulation = false\n")
    other = tmp_path / "other.toml"
    other.write_text("[pipeline]\ndo_triangulation = true\n")
    with caplog.at_level("WARNING"):
        config, path = cli._resolve_config(str(other), tmp_path)
    assert path == snapshot and config["pipeline"] == {"do_triangulation": False}
    assert any("ignoring -c" in r.message for r in caplog.records)


def test_resolve_config_snapshot_used_silently_without_cli(tmp_path, caplog):
    snapshot = tmp_path / "config.toml"
    snapshot.write_text("[pipeline]\ndo_visualization = false\n")
    with caplog.at_level("WARNING"):
        config, path = cli._resolve_config(None, tmp_path)
    assert path == snapshot and config["pipeline"] == {"do_visualization": False}
    assert not [r for r in caplog.records if "ignoring" in r.message]


# -- run: stage toggles, caching, skips -------------------------------------


def _stub_detect(monkeypatch, tmp_path):
    """Stub frame sizing + detection so detect needs no files or weights."""
    T, H, W = 3, 16, 16
    sizes = {n: (H, W) for n in FLY_CAMERAS}
    monkeypatch.setattr(cli, "_camera_image_sizes", lambda args, config: sizes)
    monkeypatch.setattr(cli, "_load_detector", lambda checkpoint, backend: object())

    def fake_detect_2d(args, config, model, sides, flips, **kw):
        v = len(FLY_CAMERAS)
        return np.zeros((v, T, 38, 2)), np.ones((v, T, 38)), None

    monkeypatch.setattr(cli, "_detect_2d", fake_detect_2d)
    return T


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
    # the recording path is recorded so a later resume can recover the frames.
    assert res.meta["input"] == str(rec.resolve())
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


def test_enabled_pose2d_recomputes_over_cache(result, tmp_path, monkeypatch):
    """An enabled stage recomputes, overwriting a cached result of another shape."""
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
            "--log-level",
            "error",
        ]
    )
    res = PoseResult.load(outdir / "poses.h5")
    assert res.pts3d is None and res.pts2d.shape == (7, T, 38, 2)  # fresh 2D, not cache


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


def test_ensure_jax_weights_cache_hit(tmp_path, monkeypatch):
    from deeperfly.pose2d import download

    dst = tmp_path / "weights.eqx"
    dst.write_bytes(b"cached")
    monkeypatch.setattr(
        download,
        "download_torch_weights",
        lambda **k: pytest.fail("should not download on a cache hit"),
    )
    assert download.ensure_jax_weights(dst) == dst


def test_ensure_jax_weights_cache_miss_converts(tmp_path, monkeypatch):
    from deeperfly.pose2d import download
    from deeperfly.pose2d.backends import jax as jaxb
    from deeperfly.pose2d.backends import torch as torchb
    import deeperfly.pose2d.backends as backends

    dst = tmp_path / "weights.eqx"
    order = []
    monkeypatch.setattr(
        download,
        "download_torch_weights",
        lambda **k: (order.append("download"), tmp_path / "src.pth")[1],
    )
    monkeypatch.setattr(
        torchb,
        "state_dict_from_torch_checkpoint",
        lambda src: (order.append("read"), {})[1],
    )
    monkeypatch.setattr(backends, "infer_num_stacks", lambda sd: 8)

    class _FakeNet:
        @staticmethod
        def deepfly2d(*, key, num_stacks):
            return "model"

    monkeypatch.setattr(jaxb, "HourglassNet", _FakeNet)
    monkeypatch.setattr(
        jaxb, "convert_state_dict", lambda sd, m: (order.append("convert"), m)[1]
    )
    monkeypatch.setattr(
        jaxb,
        "save_checkpoint",
        lambda m, p: (order.append("save"), open(p, "wb").write(b"x"))[0],
    )

    out = download.ensure_jax_weights(dst)
    assert out == dst and dst.exists()
    assert order == ["download", "read", "convert", "save"]


def test_load_detector_jax_provisions(tmp_path, monkeypatch):
    from deeperfly.pose2d import backends, download

    sentinel = tmp_path / "w.eqx"
    monkeypatch.setattr(download, "ensure_jax_weights", lambda: sentinel)
    monkeypatch.setattr(
        backends, "load_detector", lambda backend, path: (backend, path)
    )
    assert cli._load_detector(None, "jax") == ("jax", sentinel)


# -- doctor (installation / runtime report) ----------------------------------


def test_fmt_bytes_units():
    assert cli._fmt_bytes(512) == "512 B"
    assert cli._fmt_bytes(1536) == "1.5 KiB"
    assert cli._fmt_bytes(2 * 1024**3) == "2.0 GiB"


def test_doctor_reports_install_details(tmp_path, monkeypatch, capsys):
    """`deeperfly doctor` prints each section and reflects the weights cache.

    The weights cache is redirected to a temp dir with only the PyTorch
    checkpoint present, so the report shows one weight downloaded and the other
    not. COLUMNS is widened so rich does not wrap the lines we assert on.
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
        "video backends",
        "weights",
        "config",
    ):
        assert section in out
    assert "GPU inference" in out and "detectors" in out
    assert "downloaded" in out  # the PyTorch checkpoint we created
    assert f"not downloaded -- would cache as {download.JAX_WEIGHTS_NAME}" in out
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


def test_source_view_frames_root_order(result, tmp_path, monkeypatch):
    from deeperfly import video

    monkeypatch.setattr(cli, "_camera_source", lambda root, prefix: (root, prefix))
    monkeypatch.setattr(
        video, "read_frames", lambda src, backend="auto": ("frames", src)
    )
    cfg = {"inputs": {}, "pipeline": {"pose2d": {}}}
    names = result.cameras.names

    rec_a, rec_b = tmp_path / "recA", tmp_path / "recB"
    rec_a.mkdir()
    rec_b.mkdir()
    res = PoseResult(
        result.cameras,
        result.skeleton,
        result.pts2d,
        conf=result.conf,
        meta={"input": str(rec_a)},
    )

    # the run's own input (the positional recording) wins over the metadata path.
    got = cli._source_view_frames(
        argparse.Namespace(input=str(rec_b)), cfg, res, [names[0]]
    )
    assert got == {names[0]: ("frames", (str(rec_b), names[0]))}

    # else fall back to the recording stored in the result; every view is sourced.
    got = cli._source_view_frames(
        argparse.Namespace(input=None), cfg, res, [names[1], names[2]]
    )
    assert got == {
        names[1]: ("frames", (str(rec_a), names[1])),
        names[2]: ("frames", (str(rec_a), names[2])),
    }

    # a stored path that no longer exists is skipped in favor of the input.
    rec_c = tmp_path / "recC"
    rec_c.mkdir()
    stale_meta = PoseResult(
        result.cameras,
        result.skeleton,
        result.pts2d,
        meta={"input": str(tmp_path / "gone")},
    )
    got = cli._source_view_frames(
        argparse.Namespace(input=str(rec_c)), cfg, stale_meta, [names[2]]
    )
    assert got == {names[2]: ("frames", (str(rec_c), names[2]))}

    # in-memory frames (indexed by camera order) bypass the recording entirely.
    bare_meta = PoseResult(result.cameras, result.skeleton, result.pts2d, meta={})
    mem = [f"mem{i}" for i in range(len(names))]
    got = cli._source_view_frames(
        argparse.Namespace(input=None),
        cfg,
        bare_meta,
        [names[2], names[0]],
        in_memory=mem,
    )
    assert got == {names[2]: "mem2", names[0]: "mem0"}

    # no imshow views -> nothing sourced (no recording needed).
    ns = argparse.Namespace(input=None)
    assert cli._source_view_frames(ns, cfg, bare_meta, []) == {}

    # else error telling the user to re-run with the recording as the input.
    with pytest.raises(SystemExit, match="recording"):
        cli._source_view_frames(ns, cfg, bare_meta, [names[0]])
