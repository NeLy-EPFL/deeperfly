"""Tests for the unified ``deeperfly run`` command and its plumbing.

Covers the stage-range resolver, the output-directory cache (start inferred from
what is cached, ``--overwrite``, the default ``<input>/deeperfly_outputs``), the
``--until detect`` 2D-only write, automatic weight provisioning, the
pictorial->reproject fallback on resume, the detector progress hook, and the
overlay frame-recovery order. The detector and frame I/O are stubbed so these need
neither real weights nor video files.
"""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from deeperfly import cli
from deeperfly.io import PoseResult

FLY_CAMERAS = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]


# -- stage resolution --------------------------------------------------------


def _args(**kw):
    kw.setdefault("until", None)
    kw.setdefault("overwrite", False)
    kw.setdefault("input", "recording/")
    return argparse.Namespace(**kw)


@pytest.mark.parametrize(
    "have_2d,have_3d,until,overwrite,expected",
    [
        (False, False, None, False, ("detect", "visualize")),
        (False, False, "detect", False, ("detect", "detect")),
        (False, False, "pose3d", False, ("detect", "pose3d")),
        (True, False, None, False, ("pose3d", "visualize")),
        (True, False, "pose3d", False, ("pose3d", "pose3d")),
        (True, True, None, False, ("visualize", "visualize")),
        (True, True, None, True, ("detect", "visualize")),  # --overwrite ignores cache
    ],
)
def test_resolve_stages(have_2d, have_3d, until, overwrite, expected):
    start, stop = cli._resolve_stages(
        _args(until=until, overwrite=overwrite), have_2d=have_2d, have_3d=have_3d
    )
    assert (cli.STAGES[start], cli.STAGES[stop]) == expected


def test_resolve_stages_all_cached_is_noop():
    # a 3D result resumes at visualize, so --until pose3d has nothing to do.
    start, stop = cli._resolve_stages(_args(until="pose3d"), have_2d=True, have_3d=True)
    assert stop < start
    # a 2D result resumes at pose3d, so --until detect has nothing to do.
    start, stop = cli._resolve_stages(
        _args(until="detect"), have_2d=True, have_3d=False
    )
    assert stop < start


# -- run --until detect (2D-only write, no weights) --------------------------


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


def test_run_to_detect_writes_2d_only(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cli.main(["init", str(cfg)])
    T = _stub_detect(monkeypatch, tmp_path)

    outdir = tmp_path / "out"
    rec = tmp_path / "rec"
    cli.main(
        ["run", str(rec), "-c", str(cfg), "-o", str(outdir), "--until", "detect", "-q"]
    )

    res = PoseResult.load(outdir / "poses.h5")
    assert res.pts3d is None  # 2D only
    assert res.pts2d.shape == (7, T, 38, 2)
    # the recording path is recorded so a later resume can recover the frames.
    assert res.meta["input"] == str(rec.resolve())
    # the config used is snapshotted next to the results for reproducibility.
    assert (outdir / "config.toml").read_text() == cfg.read_text()


def test_run_without_config_uses_default(tmp_path, monkeypatch):
    """Omitting -c falls back to the packaged default config (snapshotted as-is)."""
    _stub_detect(monkeypatch, tmp_path)
    outdir = tmp_path / "out"
    rec = tmp_path / "rec"
    cli.main(["run", str(rec), "-o", str(outdir), "--until", "detect", "-q"])
    assert (outdir / "poses.h5").exists()
    assert (outdir / "config.toml").read_text() == cli.DEFAULT_CONFIG_PATH.read_text()


def test_default_outdir_inside_input(tmp_path, monkeypatch):
    """With no -o, results land in <input>/deeperfly_outputs/."""
    cfg = tmp_path / "config.toml"
    cli.main(["init", str(cfg)])
    _stub_detect(monkeypatch, tmp_path)
    rec = tmp_path / "rec"
    rec.mkdir()
    cli.main(["run", str(rec), "-c", str(cfg), "--until", "detect", "-q"])
    assert (rec / "deeperfly_outputs" / "poses.h5").exists()


def test_run_skips_cached_2d(result, tmp_path, monkeypatch):
    """A cached 2D poses.h5 skips detect and resumes at pose3d."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[pipeline]\ncalibrate = false\n")
    monkeypatch.setattr(
        cli, "_stage_detect", lambda *a, **k: pytest.fail("detect should be skipped")
    )
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
            "-q",
        ]
    )
    assert PoseResult.load(outdir / "poses.h5").pts3d is not None


def test_run_skips_cached_3d_only_renders(result, tmp_path, monkeypatch):
    """A cached 3D poses.h5 skips detect+pose3d and only visualizes."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    result.save(outdir / "poses.h5")  # full result with 3D
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("")
    monkeypatch.setattr(cli, "_stage_detect", lambda *a, **k: pytest.fail("no detect"))
    monkeypatch.setattr(cli, "_stage_pose3d", lambda *a, **k: pytest.fail("no pose3d"))
    called: list[bool] = []
    monkeypatch.setattr(cli, "_stage_visualize", lambda *a, **k: called.append(True))
    cli.main(["run", str(tmp_path / "rec"), "-c", str(cfg), "-o", str(outdir), "-q"])
    assert called == [True]


def test_overwrite_forces_full_recompute(tmp_path, monkeypatch):
    """--overwrite ignores any cached poses.h5 and recomputes from detect."""
    cfg = tmp_path / "config.toml"
    cli.main(["init", str(cfg)])
    T = _stub_detect(monkeypatch, tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "poses.h5").write_bytes(b"stale")  # never read under --overwrite
    cli.main(
        [
            "run",
            str(tmp_path / "rec"),
            "-c",
            str(cfg),
            "-o",
            str(outdir),
            "--overwrite",
            "--until",
            "detect",
            "-q",
        ]
    )
    res = PoseResult.load(outdir / "poses.h5")
    assert res.pts3d is None and res.pts2d.shape == (7, T, 38, 2)


def test_stale_config_warns(result, tmp_path, caplog):
    """Reusing a cache whose saved config differs warns (but still reuses it)."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    (outdir / "config.toml").write_text(
        "[pipeline]\ncalibrate = false\n# previous run\n"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text("[pipeline]\ncalibrate = false\n")  # differs from the snapshot
    with caplog.at_level("WARNING"):
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
            ]
        )
    assert any(
        "differs from the one that produced" in r.message for r in caplog.records
    )


def test_verbose_logs_image_sizes_and_batch(tmp_path, monkeypatch, caplog):
    """-v surfaces input image sizes and the detector forward batch / input size."""
    cfg = tmp_path / "config.toml"
    cli.main(["init", str(cfg)])
    _stub_detect(monkeypatch, tmp_path)
    outdir = tmp_path / "out"
    with caplog.at_level("INFO"):
        cli.main(
            [
                "run",
                str(tmp_path / "rec"),
                "-c",
                str(cfg),
                "-o",
                str(outdir),
                "--until",
                "detect",
                "-v",
            ]
        )
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


# -- pictorial -> reproject fallback on resume -------------------------------


def test_resume_pictorial_falls_back_to_reproject(result, tmp_path, caplog):
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    cfg = tmp_path / "cfg.toml"
    cfg.write_text('[pipeline]\ncalibrate = false\ncorrect = "pictorial"\n')
    with caplog.at_level("WARNING"):
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
            ]
        )
    assert any(
        "pictorial" in r.message and "reproject" in r.message for r in caplog.records
    )
    assert PoseResult.load(outdir / "poses.h5").pts3d is not None


def test_resume_uses_stored_cameras_with_full_config(result, tmp_path):
    """Resuming pose3d uses the stored rig even when the config carries a
    [cameras] table (the packaged template's cameras have no explicit principal
    point, so rebuilding them without frame sizes would fail)."""
    outdir = tmp_path / "out"
    outdir.mkdir()
    PoseResult(result.cameras, result.skeleton, result.pts2d, conf=result.conf).save(
        outdir / "poses.h5"
    )
    cfg = tmp_path / "config.toml"
    cli.main(["init", str(cfg)])  # full default config, with [cameras]
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
            "-q",
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


# -- overlay frame recovery on resume ----------------------------------------


def test_overlay_frames_source_order(result, tmp_path, monkeypatch):
    from deeperfly import video

    monkeypatch.setattr(cli, "_camera_source", lambda root, prefix: (root, prefix))
    monkeypatch.setattr(
        video, "read_frames", lambda src, backend="auto": ("frames", src)
    )
    cfg = {"inputs": {}, "detector": {}}

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

    # --recording wins over the metadata path.
    frames = cli._overlay_frames(argparse.Namespace(recording=str(rec_b)), cfg, res, 0)
    assert frames == ("frames", (str(rec_b), res.cameras.names[0]))

    # else fall back to the recording recorded in meta (if it still exists).
    frames = cli._overlay_frames(
        argparse.Namespace(recording=None, input=None), cfg, res, 1
    )
    assert frames == ("frames", (str(rec_a), res.cameras.names[1]))

    # else fall back to the run's own input (recording) if it exists.
    rec_c = tmp_path / "recC"
    rec_c.mkdir()
    bare_meta = PoseResult(result.cameras, result.skeleton, result.pts2d, meta={})
    frames = cli._overlay_frames(
        argparse.Namespace(recording=None, input=str(rec_c)), cfg, bare_meta, 2
    )
    assert frames == ("frames", (str(rec_c), result.cameras.names[2]))

    # else error pointing at --recording.
    bare = PoseResult(result.cameras, result.skeleton, result.pts2d, meta={})
    with pytest.raises(SystemExit, match="--recording"):
        cli._overlay_frames(
            argparse.Namespace(recording=None, input=None), cfg, bare, 0
        )
