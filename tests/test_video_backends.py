"""Tests for the pluggable video read/write backends.

CPU backends that are installed (opencv, pyav, video_reader_rs) are exercised for
real; the GPU backend (torchcodec) is only checked for registration/availability
since it needs CUDA. Encoded video is lossy, so round-trips assert on frame count /
shape / dtype and coarse color, not pixel values.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import video
from deeperfly.video import base

# Backends we can actually run on CPU here, restricted to what's installed.
_CPU_CANDIDATES = (
    "opencv",
    "pyav",
    "video_reader_rs",
    "torchcodec",
)
CPU_READERS = [b for b in _CPU_CANDIDATES if b in video.available_read_backends()]
CPU_WRITERS = [b for b in ("opencv", "pyav") if b in video.available_write_backends()]
ROUND_TRIP = [b for b in CPU_READERS if b in CPU_WRITERS]


def _gradient_clip(n=8, h=64, w=48):
    """Deterministic, smooth (codec-friendly) RGB clip."""
    t = np.linspace(0, 1, n)[:, None, None, None]
    yy = np.linspace(0, 1, h)[None, :, None, None]
    xx = np.linspace(0, 1, w)[None, None, :, None]
    c = np.array([1.0, 0.5, 0.2])[None, None, None, :]
    frames = (255 * (0.4 * t + 0.3 * yy + 0.3 * xx) * c).clip(0, 255)
    return frames.astype(np.uint8)


def _indexed_clip(n=12, h=32, w=32):
    """Each frame a distinct solid gray so its identity survives compression."""
    vals = (np.arange(n) * 20 + 10).clip(0, 255)
    frames = np.broadcast_to(vals[:, None, None, None], (n, h, w, 3))
    return frames.astype(np.uint8)


def _write_clip(tmp_path, frames, *, backend="pyav", name="clip.mp4"):
    path = tmp_path / name
    video.write_mp4(frames, path, fps=10, backend=backend)
    return path


@pytest.fixture(autouse=True)
def _reset_gpu_cache():
    """Clear the process-wide GPU-decode probe cache around every test.

    Tests that exercise the real auto path on a GPU box set ``_gpu_auto_failed`` /
    ``_gpu_auto_reader``; reset both so order can't leak between tests.
    """
    base._gpu_auto_failed, base._gpu_auto_reader = False, None
    yield
    base._gpu_auto_failed, base._gpu_auto_reader = False, None


def _boom(*_a, **_k):
    raise RuntimeError("Unsupported device: cuda")


def _installed_gpu_readers():
    return [base._READERS[n] for n in base.GPU_READ_ORDER if n in base._READERS]


def _break_gpu_decode(monkeypatch, cls):
    """Make ``cls`` fail on a GPU device but decode normally on the CPU.

    Mirrors the real failure mode (a torchcodec build whose CUDA path is missing)
    for backends that also serve CPU reads, so a CPU fallback still works.
    """
    monkeypatch.setattr(cls, "is_available", lambda: True)
    orig = cls._read_sequential

    def maybe_boom(path, device, start, stop, step):
        if base.is_gpu_device(device):
            raise RuntimeError("Unsupported device: cuda")
        return orig(path, device, start, stop, step)

    monkeypatch.setattr(cls, "_read_sequential", staticmethod(maybe_boom))


# -- registry / selection ----------------------------------------------------


def test_builtin_backends_registered():
    assert set(video.list_read_backends()) >= {
        "opencv",
        "pyav",
        "video_reader_rs",
        "torchcodec",
    }
    assert set(video.list_write_backends()) >= {"opencv", "pyav"}


def test_backend_capability_flags():
    assert base._READERS["torchcodec"].supports_gpu
    for name in ("opencv", "pyav", "video_reader_rs", "torchcodec"):
        assert base._READERS[name].supports_seek, name


@pytest.mark.skipif(
    "torchcodec" not in video.available_read_backends() or not base.cuda_available(),
    reason="needs torchcodec and a CUDA GPU",
)
def test_torchcodec_windowed_decode_is_frame_accurate(tmp_path):
    # The GPU decoder must decode *only* the requested window (bounded memory) and
    # match a reference FFmpeg decoder (pyav) frame-for-frame, bar a YUV->RGB
    # rounding bit. Random access and chunk boundaries must agree too.
    frames = _indexed_clip(12, 64, 64)  # NVDEC needs a minimum frame size
    path = _write_clip(tmp_path, frames)
    ref = video.read_video(path, backend="pyav", device="cpu").astype(np.int16)
    win = video.to_numpy(
        video.read_video(path, backend="torchcodec", device="cuda", start=4, stop=9)
    )
    assert win.shape[0] == 5  # decoded only the window, not all 12 frames
    assert np.abs(win.astype(np.int16) - ref[4:9]).max() <= 4  # frame-accurate
    idx = [0, 7, 3]
    pick = video.to_numpy(
        video.read_video(path, backend="torchcodec", device="cuda", indices=idx)
    )
    assert np.abs(pick.astype(np.int16) - ref[idx]).max() <= 4  # seek-accurate


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        video.read_video("nope.mp4", backend="does-not-exist")
    with pytest.raises(ValueError):
        video.write_mp4(np.zeros((1, 4, 4, 3), np.uint8), "x.mp4", backend="nope")


def test_cpu_backend_rejects_gpu_device(tmp_path):
    path = _write_clip(tmp_path, _gradient_clip(4))
    with pytest.raises(ValueError):
        video.read_video(path, backend="pyav", device="cuda")


def test_auto_select_returns_installed_backend():
    assert video.select_reader("auto", device="cpu").name in CPU_READERS
    assert video.select_writer("auto").name in CPU_WRITERS


def test_cpu_order_leads_with_in_process_pyav():
    # pyav links FFmpeg directly (no subprocess), so it is the in-process CPU
    # default, ahead of the other decoders in the auto preference order.
    assert base.CPU_READ_ORDER[0] == "pyav"
    for other in ("opencv", "torchcodec", "video_reader_rs"):
        assert base.CPU_READ_ORDER.index("pyav") < base.CPU_READ_ORDER.index(other)
    # GPU order leads with torchcodec and lists only frame-accurate decoders.
    assert base.GPU_READ_ORDER[0] == "torchcodec"


def test_resolve_device_passthrough_and_no_gpu(monkeypatch):
    # Concrete devices are returned unchanged, regardless of hardware.
    monkeypatch.setattr(base, "cuda_available", lambda: True)
    assert base.resolve_device("cpu") == "cpu"
    assert base.resolve_device("cuda:1") == "cuda:1"
    # No GPU -> "auto" falls back to CPU.
    monkeypatch.setattr(base, "cuda_available", lambda: False)
    assert base.resolve_device("auto") == "cpu"
    assert base.resolve_device("auto", "torchcodec") == "cpu"


def test_gpu_alias_normalizes_to_cuda(monkeypatch):
    # Users (and the example config) write "gpu", but torch/torchcodec only know
    # "cuda" -- the alias must be rewritten before any device string reaches them.
    assert base.canonical_device("gpu") == "cuda"
    assert base.canonical_device("gpu:1") == "cuda:1"
    assert base.canonical_device("cuda") == "cuda"  # passthrough
    assert base.canonical_device("cpu") == "cpu"
    monkeypatch.setattr(base, "cuda_available", lambda: True)
    assert base.resolve_device("gpu", "torchcodec") == "cuda"
    # A "gpu" request still routes to a GPU-capable backend (CPU-only ones reject it).
    with pytest.raises(ValueError):
        base.select_reader("pyav", device="gpu")


def test_resolve_device_prefers_gpu_when_available(monkeypatch):
    # Pretend a GPU and the torchcodec backend are both present.
    monkeypatch.setattr(base, "_gpu_auto_failed", False)  # ignore prior probes
    monkeypatch.setattr(base, "cuda_available", lambda: True)
    monkeypatch.setattr(base._READERS["torchcodec"], "is_available", lambda: True)
    # auto backend + GPU hardware + a GPU backend installed -> cuda.
    assert base.resolve_device("auto") == "cuda"
    assert video.select_reader("auto", device="auto").name == "torchcodec"
    # A forced GPU-capable backend also resolves to cuda...
    assert base.resolve_device("auto", "torchcodec") == "cuda"
    # ...but a CPU-only forced backend stays on the CPU even with a GPU present.
    assert base.resolve_device("auto", "pyav") == "cpu"


def test_auto_device_returns_host_numpy_even_for_gpu_decode(tmp_path, monkeypatch):
    # device="auto" is the portable path: it may decode via a GPU backend but
    # must hand back host NumPy. Explicit device="cuda" keeps the device tensor.
    torch = pytest.importorskip("torch")
    path = _write_clip(tmp_path, _gradient_clip(4))
    fake_gpu = torch.zeros((4, 8, 8, 3), dtype=torch.uint8)

    monkeypatch.setattr(base, "_gpu_auto_failed", False)  # ignore prior probes
    monkeypatch.setattr(base, "cuda_available", lambda: True)
    monkeypatch.setattr(base._READERS["torchcodec"], "is_available", lambda: True)
    monkeypatch.setattr(
        base._READERS["torchcodec"],
        "_read_sequential",
        staticmethod(lambda *a, **k: fake_gpu),
    )

    auto = video.read_video(path, device="auto")
    assert isinstance(auto, np.ndarray)  # brought to host
    explicit = video.read_video(path, backend="torchcodec", device="cuda")
    assert explicit is fake_gpu  # left on device for zero-copy


def test_gpu_reader_candidates_order_forced_and_cache(monkeypatch):
    monkeypatch.setattr(base._READERS["torchcodec"], "is_available", lambda: True)
    # torchcodec is the sole GPU backend.
    assert [c.name for c in base.gpu_reader_candidates("auto")] == ["torchcodec"]
    # A forced backend yields only itself.
    assert [c.name for c in base.gpu_reader_candidates("torchcodec")] == ["torchcodec"]
    base.remember_gpu_reader("torchcodec")  # cached winner short-circuits "auto"
    assert [c.name for c in base.gpu_reader_candidates("auto")] == ["torchcodec"]


def test_gpu_auto_decode_caches_winner(tmp_path, monkeypatch):
    # A successful auto GPU decode returns the on-device tensor and remembers the
    # winning backend, so later reads go straight to it without re-probing.
    torch = pytest.importorskip("torch")
    path = _write_clip(tmp_path, _gradient_clip(5, 32, 32))
    fake_gpu = torch.zeros((5, 32, 32, 3), dtype=torch.uint8)

    monkeypatch.setattr(base, "cuda_available", lambda: True)
    monkeypatch.setattr(base._READERS["torchcodec"], "is_available", lambda: True)
    monkeypatch.setattr(
        base._READERS["torchcodec"],
        "_read_sequential",
        staticmethod(lambda *a, **k: fake_gpu),
    )

    out = video.read_video(path, backend="auto", device="cuda", stop=5)
    assert out is fake_gpu  # device tensor handed back (no host round-trip)
    assert base._gpu_auto_reader == "torchcodec"  # winner cached


def test_auto_falls_back_to_cpu_when_all_gpu_backends_fail(
    tmp_path, monkeypatch, caplog
):
    # Every installed GPU backend fails to decode (a GPU is present but unusable).
    # The auto path warns once, disables the GPU process-wide, and returns host
    # NumPy via a CPU decoder.
    path = _write_clip(tmp_path, _gradient_clip(6, 64, 48))
    monkeypatch.setattr(base, "cuda_available", lambda: True)
    for cls in _installed_gpu_readers():
        _break_gpu_decode(monkeypatch, cls)  # GPU fails, CPU still decodes

    with caplog.at_level("WARNING", logger="deeperfly.video"):
        out = video.read_video(path, device="auto")
    assert isinstance(out, np.ndarray) and out.shape[0] == 6  # decoded on CPU
    assert sum("using CPU decode" in r.message for r in caplog.records) == 1
    assert base._gpu_auto_failed  # GPU disabled for the rest of the process
    assert base.resolve_device("auto") == "cpu"  # no second GPU probe


def test_resolve_device_downgrades_explicit_gpu_after_failure(monkeypatch):
    # Once GPU decode is known dead, an auto-backend cuda request is downgraded to
    # CPU (stop retrying), but a forced GPU backend is still honored strictly.
    monkeypatch.setattr(base, "_gpu_auto_failed", True)
    assert base.resolve_device("cuda", "auto") == "cpu"
    assert base.resolve_device("cuda", "torchcodec") == "cuda"


def test_explicit_gpu_failure_is_not_swallowed(tmp_path, monkeypatch):
    # A forced GPU backend is strict: a decode failure must propagate, not silently
    # fall back to the CPU.
    path = _write_clip(tmp_path, _gradient_clip(4))
    monkeypatch.setattr(base._READERS["torchcodec"], "is_available", lambda: True)
    monkeypatch.setattr(
        base._READERS["torchcodec"], "_read_sequential", staticmethod(_boom)
    )
    with pytest.raises(RuntimeError):
        video.read_video(path, backend="torchcodec", device="cuda")


def test_is_gpu_device_and_device_id():
    assert base.is_gpu_device("cuda")
    assert base.is_gpu_device("cuda:1")
    assert not base.is_gpu_device("cpu")
    assert not base.is_gpu_device(None)
    assert base.device_id("cuda:2") == 2
    assert base.device_id("cuda") == 0


# -- to_numpy / to_torch -----------------------------------------------------


def test_to_numpy_passthrough():
    a = np.zeros((2, 2, 3), np.uint8)
    assert video.to_numpy(a) is a


def test_to_torch_from_numpy():
    a = _gradient_clip(3, 8, 8)
    x = video.to_torch(a)
    assert tuple(x.shape) == a.shape
    np.testing.assert_array_equal(np.asarray(x), a)


def test_to_torch_passthrough_from_torch():
    # An already-on-device torch tensor passes through untouched (zero-copy).
    torch = pytest.importorskip("torch")
    t = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    x = video.to_torch(t)
    assert x is t


# -- readers -----------------------------------------------------------------


def test_video_reader_rs_installed_means_available():
    # video_reader-rs's wheel bundles FFmpeg libs that lack an inter-lib RUNPATH,
    # so `import video_reader` dies on a transitive dep unless we pre-dlopen them.
    # When the package is on disk, the backend must therefore advertise itself --
    # otherwise the preload regressed and the install is silently unusable.
    import importlib.util

    if importlib.util.find_spec("video_reader") is None:
        pytest.skip("video_reader-rs not installed")
    assert "video_reader_rs" in video.available_read_backends()


@pytest.mark.parametrize("backend", CPU_READERS)
def test_reader_roundtrip(tmp_path, backend):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames)  # written by pyav (the core default)
    out = video.read_video(path, backend=backend, device="cpu")
    assert out.shape[0] == frames.shape[0]
    assert out.shape[1:] == (64, 48, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("backend", CPU_READERS)
def test_reader_sequential_slice(tmp_path, backend):
    frames = _gradient_clip(10, 32, 32)
    path = _write_clip(tmp_path, frames)
    out = video.read_video(path, backend=backend, device="cpu", start=2, stop=8, step=2)
    assert out.shape[0] == len(range(2, 8, 2))  # 3 frames


@pytest.mark.parametrize("backend", CPU_READERS)
def test_random_access_matches_sequential(tmp_path, backend):
    # Random access must return the *same frames* (in order) as a full read,
    # whether the backend seeks or falls back to decode-and-gather.
    frames = _indexed_clip(12, 32, 32)
    path = _write_clip(tmp_path, frames)
    idx = [0, 5, 3, 9, 5]
    full = video.read_video(path, backend=backend, device="cpu")
    picked = video.read_video(path, backend=backend, device="cpu", indices=idx)
    assert picked.shape[0] == len(idx)
    np.testing.assert_allclose(
        picked.reshape(len(idx), -1).mean(1),
        full[idx].reshape(len(idx), -1).mean(1),
        atol=3,
    )


# -- writers -----------------------------------------------------------------


@pytest.mark.parametrize("backend", CPU_WRITERS)
def test_writer_roundtrip(tmp_path, backend):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames, backend=backend, name=f"{backend}.mp4")
    back = video.read_video(path, backend="pyav")
    assert back.shape[0] >= frames.shape[0] - 1  # codecs may drop/add a frame
    assert back.shape[1:] == (64, 48, 3)


@pytest.mark.parametrize("backend", ROUND_TRIP)
def test_color_channel_order_preserved(tmp_path, backend):
    # Solid red clip: a BGR/RGB mixup in a backend would surface here.
    red = np.zeros((6, 32, 32, 3), np.uint8)
    red[..., 0] = 220
    path = _write_clip(tmp_path, red, backend=backend, name=f"{backend}_red.mp4")
    out = video.read_video(path, backend=backend, device="cpu")
    mean = out.reshape(-1, 3).mean(0)
    assert mean[0] > mean[1] and mean[0] > mean[2]


def test_non_uint8_frames_are_clipped(tmp_path):
    frames = _gradient_clip(5, 32, 32).astype(np.float32)
    path = tmp_path / "float.mp4"
    video.write_mp4(frames, path, fps=10)  # must not raise on float input
    assert video.read_video(path).dtype == np.uint8


# -- image-sequence reading (read_images / read_frames) ----------------------


def _write_images(tmp_path, frames, *, ext="png", name="f"):
    import imageio.v3 as iio

    for i, fr in enumerate(frames):
        iio.imwrite(tmp_path / f"{name}_{i:03d}.{ext}", fr)
    return tmp_path


def test_count_frames_video_and_images(tmp_path):
    clip = _write_clip(tmp_path, _gradient_clip(12, 32, 32))
    assert video.count_frames(clip) == 12  # container metadata, no full decode
    img_dir = _write_images(tmp_path, _gradient_clip(5, 16, 16), ext="png")
    assert video.count_frames(img_dir) == 5  # image sequences count files
    assert video.count_frames(tmp_path / "missing.mp4") is None  # unknown -> None


def test_read_images_parallel_rgb(tmp_path):
    frames = _gradient_clip(6, 40, 50)
    _write_images(tmp_path, frames, ext="png")
    out = video.read_images(tmp_path)
    assert out.shape == (6, 40, 50, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, frames)  # PNG is lossless
    # worker count must not change the result
    np.testing.assert_array_equal(video.read_images(tmp_path, workers=1), out)


def test_read_images_grayscale_broadcasts_to_rgb(tmp_path):
    # A grayscale (H, W) PNG must broadcast to 3 equal channels, NOT slice width.
    gray = (np.arange(20 * 30).reshape(20, 30) % 255).astype(np.uint8)
    _write_images(tmp_path, gray[None], ext="png", name="g")
    out = video.read_images(tmp_path)
    assert out.shape == (1, 20, 30, 3)
    np.testing.assert_array_equal(out[0, ..., 0], out[0, ..., 2])
    np.testing.assert_array_equal(out[0, ..., 0], gray)


def test_read_images_indices_and_slice(tmp_path):
    frames = _indexed_clip(10, 16, 16)
    _write_images(tmp_path, frames, ext="png")
    np.testing.assert_array_equal(
        video.read_images(tmp_path, indices=[0, 3, 7]), frames[[0, 3, 7]]
    )
    np.testing.assert_array_equal(
        video.read_images(tmp_path, start=1, stop=9, step=2), frames[1:9:2]
    )


def test_read_frames_dispatches_dir_vs_video(tmp_path):
    frames = _indexed_clip(6, 32, 32)
    _write_images(tmp_path, frames, ext="png")
    from_dir = video.read_frames(tmp_path)
    assert from_dir.shape == (6, 32, 32, 3)
    np.testing.assert_array_equal(from_dir, frames)
    mp4 = _write_clip(tmp_path, frames, name="clip.mp4")
    assert video.read_frames(mp4).shape[0] == 6  # routed to read_video


def test_read_images_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        video.read_images(tmp_path / "empty")


def _gpu_decode_available() -> bool:
    """GPU decode needs a torch CUDA build.

    The test decodes JPEGs to a CUDA ``torch.Tensor`` and hands it to ``to_torch``
    (zero-copy), so it skips without a torch CUDA device.
    """
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu_decode_available(), reason="needs torch CUDA")
def test_read_images_gpu_decode(tmp_path):
    import torch

    frames = _indexed_clip(5, 32, 48)  # solid grays: nvJPEG == libjpeg exactly
    _write_images(tmp_path, frames, ext="jpg")
    out = video.read_images(tmp_path, device="cuda")
    assert isinstance(out, torch.Tensor) and out.is_cuda
    assert tuple(out.shape) == (5, 32, 48, 3) and out.dtype == torch.uint8
    # solid frames survive JPEG, so identity is preserved per frame
    means = out.float().mean((1, 2, 3)).cpu().numpy()
    assert np.all(np.diff(means) > 0)
    x = video.to_torch(out)  # GPU tensor stays on device (zero-copy passthrough)
    assert x.is_cuda and tuple(x.shape) == (5, 32, 48, 3)
