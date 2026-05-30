"""Tests for the pluggable video read/write backends.

CPU backends that are installed (imageio, opencv, pyav, decord,
video_reader_rs) are exercised for real; GPU backends (torchcodec,
pynvvideocodec, dali) are only checked for registration/availability since they
need CUDA. Encoded video is lossy, so round-trips assert on frame count / shape /
dtype and coarse colour, not pixel values.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import video
from deeperfly.video import base

# Backends we can actually run on CPU here, restricted to what's installed.
_CPU_CANDIDATES = ("imageio", "opencv", "pyav", "decord", "video_reader_rs")
CPU_READERS = [b for b in _CPU_CANDIDATES if b in video.available_read_backends()]
CPU_WRITERS = [
    b for b in ("imageio", "opencv", "pyav") if b in video.available_write_backends()
]
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
    """Each frame a distinct solid grey so its identity survives compression."""
    vals = (np.arange(n) * 20 + 10).clip(0, 255)
    frames = np.broadcast_to(vals[:, None, None, None], (n, h, w, 3))
    return frames.astype(np.uint8)


def _write_clip(tmp_path, frames, *, backend="imageio", name="clip.mp4"):
    path = tmp_path / name
    video.write_mp4(frames, path, fps=10, backend=backend)
    return path


# -- registry / selection ----------------------------------------------------


def test_builtin_backends_registered():
    assert set(video.list_read_backends()) >= {
        "imageio",
        "opencv",
        "pyav",
        "decord",
        "video_reader_rs",
        "torchcodec",
        "pynvvideocodec",
        "dali",
    }
    assert set(video.list_write_backends()) >= {"imageio", "opencv", "pyav"}


def test_imageio_available_in_test_env():
    assert "imageio" in video.available_read_backends()
    assert "imageio" in video.available_write_backends()


def test_backend_capability_flags():
    assert base._READERS["torchcodec"].supports_gpu
    assert base._READERS["dali"].supports_gpu
    assert base._READERS["decord"].supports_gpu
    assert not base._READERS["imageio"].supports_gpu
    # seek-capable backends
    for name in ("opencv", "pyav", "decord", "video_reader_rs", "torchcodec"):
        assert base._READERS[name].supports_seek, name
    assert not base._READERS["imageio"].supports_seek


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        video.read_video("nope.mp4", backend="does-not-exist")
    with pytest.raises(ValueError):
        video.write_mp4(np.zeros((1, 4, 4, 3), np.uint8), "x.mp4", backend="nope")


def test_cpu_backend_rejects_gpu_device(tmp_path):
    path = _write_clip(tmp_path, _gradient_clip(4))
    with pytest.raises(ValueError):
        video.read_video(path, backend="imageio", device="cuda")


def test_auto_select_returns_installed_backend():
    assert video.select_reader("auto", device="cpu").name in CPU_READERS
    assert video.select_writer("auto").name in CPU_WRITERS


def test_is_gpu_device_and_device_id():
    assert base.is_gpu_device("cuda")
    assert base.is_gpu_device("cuda:1")
    assert not base.is_gpu_device("cpu")
    assert not base.is_gpu_device(None)
    assert base.device_id("cuda:2") == 2
    assert base.device_id("cuda") == 0


# -- to_numpy / to_jax -------------------------------------------------------


def test_to_numpy_passthrough():
    a = np.zeros((2, 2, 3), np.uint8)
    assert video.to_numpy(a) is a


def test_to_jax_from_numpy():
    a = _gradient_clip(3, 8, 8)
    x = video.to_jax(a)
    assert tuple(x.shape) == a.shape
    np.testing.assert_array_equal(np.asarray(x), a)


def test_to_jax_dlpack_from_torch():
    # DLPack handoff works host-side too, exercising the zero-copy path.
    torch = pytest.importorskip("torch")
    t = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    x = video.to_jax(t)
    np.testing.assert_array_equal(np.asarray(x), t.numpy())


# -- readers -----------------------------------------------------------------


@pytest.mark.parametrize("backend", CPU_READERS)
def test_reader_roundtrip(tmp_path, backend):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames)  # written by imageio (reliable)
    out = video.read_video(path, backend=backend)
    assert out.shape[0] == frames.shape[0]
    assert out.shape[1:] == (64, 48, 3)
    assert out.dtype == np.uint8


@pytest.mark.parametrize("backend", CPU_READERS)
def test_reader_sequential_slice(tmp_path, backend):
    frames = _gradient_clip(10, 32, 32)
    path = _write_clip(tmp_path, frames)
    out = video.read_video(path, backend=backend, start=2, stop=8, step=2)
    assert out.shape[0] == len(range(2, 8, 2))  # 3 frames


@pytest.mark.parametrize("backend", CPU_READERS)
def test_random_access_matches_sequential(tmp_path, backend):
    # Random access must return the *same frames* (in order) as a full read,
    # whether the backend seeks or falls back to decode-and-gather.
    frames = _indexed_clip(12, 32, 32)
    path = _write_clip(tmp_path, frames)
    idx = [0, 5, 3, 9, 5]
    full = video.read_video(path, backend=backend)
    picked = video.read_video(path, backend=backend, indices=idx)
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
    back = video.read_video(path, backend="imageio")
    assert back.shape[0] >= frames.shape[0] - 1  # codecs may drop/add a frame
    assert back.shape[1:] == (64, 48, 3)


@pytest.mark.parametrize("backend", ROUND_TRIP)
def test_color_channel_order_preserved(tmp_path, backend):
    # Solid red clip: a BGR/RGB mixup in a backend would surface here.
    red = np.zeros((6, 32, 32, 3), np.uint8)
    red[..., 0] = 220
    path = _write_clip(tmp_path, red, backend=backend, name=f"{backend}_red.mp4")
    out = video.read_video(path, backend=backend)
    mean = out.reshape(-1, 3).mean(0)
    assert mean[0] > mean[1] and mean[0] > mean[2]


def test_non_uint8_frames_are_clipped(tmp_path):
    frames = _gradient_clip(5, 32, 32).astype(np.float32)
    path = tmp_path / "float.mp4"
    video.write_mp4(frames, path, fps=10)  # must not raise on float input
    assert video.read_video(path).dtype == np.uint8
