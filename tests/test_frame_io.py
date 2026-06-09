"""Tests for the frame I/O readers (PyAV video, image-sequence) and MP4 writing.

Footage is read through :func:`deeperfly.io.open_reader` (or the
:class:`~deeperfly.io.VideoReader` / :class:`~deeperfly.io.ImageSequenceReader`
classes directly). PyAV is the only video backend -- it decodes and encodes H.264
on the CPU. Encoded video is lossy, so round-trips assert on frame count / shape /
dtype and coarse color, not pixel values.
"""

from __future__ import annotations

import numpy as np
import pytest

from deeperfly import io


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
    with io.VideoWriter(path, fps=10) as writer:
        writer.write_frames(frames)
    return path


# -- to_numpy / to_torch -----------------------------------------------------


def test_to_numpy_passthrough():
    a = np.zeros((2, 2, 3), np.uint8)
    assert io.to_numpy(a) is a


def test_to_torch_from_numpy():
    a = _gradient_clip(3, 8, 8)
    x = io.to_torch(a)
    assert tuple(x.shape) == a.shape
    np.testing.assert_array_equal(np.asarray(x), a)


def test_to_torch_passthrough_from_torch():
    # An already-materialized torch tensor passes through untouched (zero-copy).
    torch = pytest.importorskip("torch")
    t = torch.arange(24, dtype=torch.uint8).reshape(2, 2, 2, 3)
    x = io.to_torch(t)
    assert x is t


# -- video read / write (PyAV) -----------------------------------------------


def test_reader_roundtrip(tmp_path):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames)
    out = io.VideoReader(path).read()
    assert out.shape[0] == frames.shape[0]
    assert out.shape[1:] == (64, 48, 3)
    assert out.dtype == np.uint8


def test_reader_sequential_slice(tmp_path):
    frames = _gradient_clip(10, 32, 32)
    path = _write_clip(tmp_path, frames)
    out = io.VideoReader(path).read(start=2, stop=8, step=2)
    assert out.shape[0] == len(range(2, 8, 2))  # 3 frames


def test_stream_frames_blocks_concatenate_to_full_read(tmp_path):
    # stream() groups one continuous decode into <= block chunks that concatenate
    # back to the whole recording (what the streaming consumer sees).
    frames = _indexed_clip(20, 32, 32)
    path = _write_clip(tmp_path, frames)
    blocks = list(io.VideoReader(path).stream(block=7))
    full = io.VideoReader(path).read()
    assert all(len(b) == 7 for b in blocks[:-1])  # only the last block may be short
    assert sum(len(b) for b in blocks) == len(full)
    np.testing.assert_array_equal(np.concatenate(blocks), full)


def test_random_access_matches_sequential(tmp_path):
    # Random access (PyAV seeks to each frame) must return the *same frames*, in the
    # requested order, as selecting them from a full read.
    frames = _indexed_clip(12, 32, 32)
    path = _write_clip(tmp_path, frames)
    idx = [0, 5, 3, 9, 5]
    reader = io.VideoReader(path)
    full = reader.read()
    picked = reader.read(indices=idx)
    assert picked.shape[0] == len(idx)
    np.testing.assert_allclose(
        picked.reshape(len(idx), -1).mean(1),
        full[idx].reshape(len(idx), -1).mean(1),
        atol=3,
    )


def test_writer_roundtrip(tmp_path):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames)
    back = io.VideoReader(path).read()
    assert back.shape[0] >= frames.shape[0] - 1  # codecs may drop/add a frame
    assert back.shape[1:] == (64, 48, 3)


def test_color_channel_order_preserved(tmp_path):
    # Solid red clip: a BGR/RGB mixup would surface here.
    red = np.zeros((6, 32, 32, 3), np.uint8)
    red[..., 0] = 220
    path = _write_clip(tmp_path, red, name="red.mp4")
    out = io.VideoReader(path).read()
    mean = out.reshape(-1, 3).mean(0)
    assert mean[0] > mean[1] and mean[0] > mean[2]


def test_read_video_no_frames_raises(tmp_path):
    # An out-of-range slice decodes nothing -> a clear error, not an empty array.
    path = _write_clip(tmp_path, _gradient_clip(4, 16, 16))
    with pytest.raises(ValueError):
        io.VideoReader(path).read(start=100)


def test_non_uint8_frames_are_clipped(tmp_path):
    frames = _gradient_clip(5, 32, 32).astype(np.float32)
    path = tmp_path / "float.mp4"
    with io.VideoWriter(path, fps=10) as writer:
        writer.write_frames(frames)  # must not raise on float input
    assert io.VideoReader(path).read().dtype == np.uint8


def test_writer_frame_by_frame_matches_batch(tmp_path):
    # Writing one (H, W, 3) frame at a time produces the same clip as one batch write.
    frames = _indexed_clip(10, 32, 32)
    batch = tmp_path / "batch.mp4"
    incremental = tmp_path / "incremental.mp4"
    with io.VideoWriter(batch, fps=10) as writer:
        writer.write_frames(frames)
    with io.VideoWriter(incremental, fps=10) as writer:
        for frame in frames:  # frame-by-frame, never holding the whole clip
            writer.write_frame(frame)
    np.testing.assert_array_equal(
        io.VideoReader(batch).read(), io.VideoReader(incremental).read()
    )


def test_writer_accepts_iterator_of_blocks(tmp_path):
    # write() consumes any iterable of frames/batches -- e.g. a generator of blocks,
    # the streaming-decode shape -- so encoding can overlap production.
    frames = _indexed_clip(9, 16, 16)
    path = tmp_path / "streamed.mp4"

    def blocks():
        for pos in range(0, len(frames), 4):
            yield frames[pos : pos + 4]

    with io.VideoWriter(path, fps=10) as writer:
        writer.write_frames(blocks())
    assert io.VideoReader(path).read().shape[0] == len(frames)


def test_video_reader_name_and_fps(tmp_path):
    # open_reader resolves a video file to a VideoReader; name/fps come from metadata.
    path = _write_clip(tmp_path, _gradient_clip(8, 16, 16))
    reader = io.open_reader(path)
    assert isinstance(reader, io.VideoReader)
    assert reader.name == "pyav"
    assert reader.fps() == pytest.approx(10.0, abs=0.5)


def test_reader_is_a_context_manager(tmp_path):
    # Readers work in a `with` block (symmetric with VideoWriter); close() is safe.
    frames = _gradient_clip(6, 16, 16)
    path = _write_clip(tmp_path, frames)
    with io.open_reader(path) as reader:
        assert reader.read().shape[0] == frames.shape[0]


# -- image-sequence reading --------------------------------------------------


def _write_images(tmp_path, frames, *, ext="png", name="f"):
    import cv2

    for i, fr in enumerate(frames):
        # ImageSequenceReader returns RGB but cv2 encodes its input as BGR, so flip
        # color frames first -- cv2 then stores them as correct RGB in the file and
        # the lossless round-trip is the identity (for any decoder). Grayscale (2-D)
        # frames are written as-is.
        if fr.ndim == 3:
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(tmp_path / f"{name}_{i:03d}.{ext}"), fr)
    return tmp_path


def test_count_frames_video_and_images(tmp_path):
    clip = _write_clip(tmp_path, _gradient_clip(12, 32, 32))
    assert io.open_reader(clip).count() == 12  # container metadata, no full decode
    img_dir = _write_images(tmp_path, _gradient_clip(5, 16, 16), ext="png")
    assert io.open_reader(img_dir).count() == 5  # image sequences count files


def test_open_reader_missing_source_raises(tmp_path):
    # A missing source can't be opened (no video file, no images matched) -- fail
    # fast rather than silently yielding an empty reader.
    with pytest.raises(FileNotFoundError):
        io.open_reader(tmp_path / "missing.mp4")


def test_image_reader_name_and_no_fps(tmp_path):
    _write_images(tmp_path, _gradient_clip(3, 16, 16), ext="png")
    reader = io.open_reader(tmp_path)
    assert isinstance(reader, io.ImageSequenceReader)
    assert reader.name == "opencv"  # core default
    assert reader.fps() is None  # image sequences carry no frame rate


def test_read_images_parallel_rgb(tmp_path):
    frames = _gradient_clip(6, 40, 50)
    _write_images(tmp_path, frames, ext="png")
    out = io.ImageSequenceReader.from_pattern(tmp_path).read()
    assert out.shape == (6, 40, 50, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, frames)  # PNG is lossless
    # worker count must not change the result
    single = io.ImageSequenceReader.from_pattern(tmp_path, workers=1).read()
    np.testing.assert_array_equal(single, out)


@pytest.mark.parametrize("image_backend", ["auto", "opencv", "imageio"])
def test_read_images_backend_selection(tmp_path, image_backend):
    # The image reader is selectable ("auto"/"opencv" core; "imageio" optional). PNG is
    # lossless, so every decoder must return the identical RGB array.
    if image_backend == "imageio" and "imageio" not in io.available_image_readers():
        pytest.skip("imageio extra not installed")
    frames = _gradient_clip(4, 24, 32)
    _write_images(tmp_path, frames, ext="png")
    out = io.ImageSequenceReader.from_pattern(
        tmp_path, image_backend=image_backend
    ).read()
    assert out.shape == (4, 24, 32, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, frames)


def test_unknown_image_backend_raises(tmp_path):
    _write_images(tmp_path, _gradient_clip(2, 16, 16), ext="png")
    with pytest.raises(ValueError, match="unknown image reader"):
        io.ImageSequenceReader.from_pattern(tmp_path, image_backend="nope").read()


def test_read_images_grayscale_broadcasts_to_rgb(tmp_path):
    # A grayscale (H, W) PNG must broadcast to 3 equal channels, NOT slice width.
    gray = (np.arange(20 * 30).reshape(20, 30) % 255).astype(np.uint8)
    _write_images(tmp_path, gray[None], ext="png", name="g")
    out = io.ImageSequenceReader.from_pattern(tmp_path).read()
    assert out.shape == (1, 20, 30, 3)
    np.testing.assert_array_equal(out[0, ..., 0], out[0, ..., 2])
    np.testing.assert_array_equal(out[0, ..., 0], gray)


def test_read_images_indices_and_slice(tmp_path):
    frames = _indexed_clip(10, 16, 16)
    _write_images(tmp_path, frames, ext="png")
    reader = io.ImageSequenceReader.from_pattern(tmp_path)
    np.testing.assert_array_equal(reader.read(indices=[0, 3, 7]), frames[[0, 3, 7]])
    np.testing.assert_array_equal(reader.read(start=1, stop=9, step=2), frames[1:9:2])


def test_open_reader_dispatches_dir_vs_video(tmp_path):
    frames = _indexed_clip(6, 32, 32)
    _write_images(tmp_path, frames, ext="png")
    from_dir = io.open_reader(tmp_path).read()
    assert from_dir.shape == (6, 32, 32, 3)
    np.testing.assert_array_equal(from_dir, frames)
    mp4 = _write_clip(tmp_path, frames, name="clip.mp4")
    reader = io.open_reader(mp4)
    assert isinstance(reader, io.VideoReader)  # routed to the video reader
    assert reader.read().shape[0] == 6


def test_stream_frames_image_sequence_blocks(tmp_path):
    # The image reader's stream() yields the sorted sequence in <= block chunks
    # (PNG is lossless, so frame identity and counts are exact).
    frames = _indexed_clip(7, 16, 16)
    _write_images(tmp_path, frames, ext="png")
    blocks = list(io.open_reader(tmp_path).stream(block=3))
    assert [len(b) for b in blocks] == [3, 3, 1]
    np.testing.assert_array_equal(np.concatenate(blocks), frames)


def test_read_images_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.open_reader(tmp_path / "empty").read()
