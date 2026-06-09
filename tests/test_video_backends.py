"""Tests for video read/write (PyAV) and image-sequence decoding.

PyAV is the only video backend -- it decodes and encodes H.264 on the CPU.
Encoded video is lossy, so round-trips assert on frame count / shape / dtype and
coarse color, not pixel values.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import video


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


def _write_clip(tmp_path, frames, *, name="clip.mp4"):
    path = tmp_path / name
    video.write_mp4(frames, path, fps=10)
    return path


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
    # An already-materialized torch tensor passes through untouched (zero-copy).
    torch = pytest.importorskip("torch")
    t = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    x = video.to_torch(t)
    assert x is t


# -- video read / write (PyAV) -----------------------------------------------


def test_reader_roundtrip(tmp_path):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames)
    out = video.read_video(path)
    assert out.shape[0] == frames.shape[0]
    assert out.shape[1:] == (64, 48, 3)
    assert out.dtype == np.uint8


def test_reader_sequential_slice(tmp_path):
    frames = _gradient_clip(10, 32, 32)
    path = _write_clip(tmp_path, frames)
    out = video.read_video(path, start=2, stop=8, step=2)
    assert out.shape[0] == len(range(2, 8, 2))  # 3 frames


def test_stream_frames_blocks_concatenate_to_full_read(tmp_path):
    # stream_frames groups one continuous decode into <= block chunks that
    # concatenate back to the whole recording (what the streaming consumer sees).
    frames = _indexed_clip(20, 32, 32)
    path = _write_clip(tmp_path, frames)
    blocks = list(video.stream_frames(path, block=7))
    full = video.read_video(path)
    assert all(len(b) == 7 for b in blocks[:-1])  # only the last block may be short
    assert sum(len(b) for b in blocks) == len(full)
    np.testing.assert_array_equal(np.concatenate(blocks), full)


def test_random_access_matches_sequential(tmp_path):
    # Random access (PyAV seeks to each frame) must return the *same frames*, in the
    # requested order, as selecting them from a full read.
    frames = _indexed_clip(12, 32, 32)
    path = _write_clip(tmp_path, frames)
    idx = [0, 5, 3, 9, 5]
    full = video.read_video(path)
    picked = video.read_video(path, indices=idx)
    assert picked.shape[0] == len(idx)
    np.testing.assert_allclose(
        picked.reshape(len(idx), -1).mean(1),
        full[idx].reshape(len(idx), -1).mean(1),
        atol=3,
    )


def test_writer_roundtrip(tmp_path):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames)
    back = video.read_video(path)
    assert back.shape[0] >= frames.shape[0] - 1  # codecs may drop/add a frame
    assert back.shape[1:] == (64, 48, 3)


def test_color_channel_order_preserved(tmp_path):
    # Solid red clip: a BGR/RGB mixup would surface here.
    red = np.zeros((6, 32, 32, 3), np.uint8)
    red[..., 0] = 220
    path = _write_clip(tmp_path, red, name="red.mp4")
    out = video.read_video(path)
    mean = out.reshape(-1, 3).mean(0)
    assert mean[0] > mean[1] and mean[0] > mean[2]


def test_read_video_no_frames_raises(tmp_path):
    # An out-of-range slice decodes nothing -> a clear error, not an empty array.
    path = _write_clip(tmp_path, _gradient_clip(4, 16, 16))
    with pytest.raises(ValueError):
        video.read_video(path, start=100)


def test_non_uint8_frames_are_clipped(tmp_path):
    frames = _gradient_clip(5, 32, 32).astype(np.float32)
    path = tmp_path / "float.mp4"
    video.write_mp4(frames, path, fps=10)  # must not raise on float input
    assert video.read_video(path).dtype == np.uint8


# -- image-sequence reading (read_images / read_frames) ----------------------


def _write_images(tmp_path, frames, *, ext="png", name="f"):
    import cv2

    for i, fr in enumerate(frames):
        # read_images returns RGB but cv2 encodes its input as BGR, so flip color
        # frames first -- cv2 then stores them as correct RGB in the file and the
        # lossless round-trip is the identity (for any decoder). Grayscale (2-D)
        # frames are written as-is.
        if fr.ndim == 3:
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(tmp_path / f"{name}_{i:03d}.{ext}"), fr)
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


@pytest.mark.parametrize("image_backend", ["auto", "opencv", "imageio"])
def test_read_images_backend_selection(tmp_path, image_backend):
    # The image reader is selectable ("auto"/"opencv" core; "imageio" optional). PNG is
    # lossless, so every decoder must return the identical RGB array.
    if image_backend == "imageio" and "imageio" not in video.available_image_readers():
        pytest.skip("imageio extra not installed")
    frames = _gradient_clip(4, 24, 32)
    _write_images(tmp_path, frames, ext="png")
    out = video.read_images(tmp_path, image_backend=image_backend)
    assert out.shape == (4, 24, 32, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, frames)


def test_unknown_image_backend_raises(tmp_path):
    _write_images(tmp_path, _gradient_clip(2, 16, 16), ext="png")
    with pytest.raises(ValueError, match="unknown image reader"):
        video.read_images(tmp_path, image_backend="nope")


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


def test_stream_frames_image_sequence_blocks(tmp_path):
    # The image branch of stream_frames yields the sorted sequence in <= block
    # chunks (PNG is lossless, so frame identity and counts are exact).
    frames = _indexed_clip(7, 16, 16)
    _write_images(tmp_path, frames, ext="png")
    blocks = list(video.stream_frames(tmp_path, block=3))
    assert [len(b) for b in blocks] == [3, 3, 1]
    np.testing.assert_array_equal(np.concatenate(blocks), frames)


def test_read_images_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        video.read_images(tmp_path / "empty")
