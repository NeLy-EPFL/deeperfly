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


def _write_clip(tmp_path, frames, *, name="clip.mp4", pix_fmt=None):
    path = tmp_path / name
    kw = {} if pix_fmt is None else {"pix_fmt": pix_fmt}
    with io.VideoWriter(path, fps=10, **kw) as writer:
        writer.write_frames(frames)
    return path


def _write_rig_clip(tmp_path, luma, *, name="rig.mp4"):
    """A clip shaped like this project's real footage: full-range, flat-chroma yuvj420p.

    The writer's RGB input path cannot produce one -- ``pix_fmt="yuvj420p"`` gets the range
    right but swscale's RGB->YUV rounds the neutral chroma to 127, and 'no color' means
    exactly 128 -- so the YUV planes are written directly, which is what a monochrome camera
    records: the luma as given, both chroma planes pinned neutral.
    """
    import av

    n, h, w = luma.shape
    with av.open(str(tmp_path / name), mode="w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuvj420p"
        for i in range(n):
            frame = av.VideoFrame(w, h, "yuvj420p")
            frame.planes[0].update(np.ascontiguousarray(luma[i]).tobytes())
            neutral = np.full((h // 2, w // 2), 128, np.uint8)
            frame.planes[1].update(neutral.tobytes())
            frame.planes[2].update(neutral.tobytes())
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return tmp_path / name


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
    out = io.VideoReader(path)[:]
    assert out.shape[0] == frames.shape[0]
    assert out.shape[1:] == (64, 48, 3)
    assert out.dtype == np.uint8


def test_reader_sequential_slice(tmp_path):
    frames = _gradient_clip(10, 32, 32)
    path = _write_clip(tmp_path, frames)
    out = io.VideoReader(path)[2:8:2]
    assert out.shape[0] == len(range(2, 8, 2))  # 3 frames


def test_stream_frames_blocks_concatenate_to_full_read(tmp_path):
    # stream_blocks() groups one continuous decode into <= block_size chunks that
    # concatenate back to the whole recording.
    frames = _indexed_clip(20, 32, 32)
    path = _write_clip(tmp_path, frames)
    blocks = list(io.VideoReader(path).stream_blocks(block_size=7))
    full = io.VideoReader(path)[:]
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
    full = reader[:]
    picked = reader[idx]
    assert picked.shape[0] == len(idx)
    np.testing.assert_allclose(
        picked.reshape(len(idx), -1).mean(1),
        full[idx].reshape(len(idx), -1).mean(1),
        atol=3,
    )


def test_writer_roundtrip(tmp_path):
    frames = _gradient_clip(8, 64, 48)
    path = _write_clip(tmp_path, frames)
    back = io.VideoReader(path)[:]
    assert back.shape[0] >= frames.shape[0] - 1  # codecs may drop/add a frame
    assert back.shape[1:] == (64, 48, 3)


def test_color_channel_order_preserved(tmp_path):
    # Solid red clip: a BGR/RGB mixup would surface here.
    red = np.zeros((6, 32, 32, 3), np.uint8)
    red[..., 0] = 220
    path = _write_clip(tmp_path, red, name="red.mp4")
    out = io.VideoReader(path)[:]
    mean = out.reshape(-1, 3).mean(0)
    assert mean[0] > mean[1] and mean[0] > mean[2]


def test_read_video_no_frames_raises(tmp_path):
    # An out-of-range slice decodes nothing -> a clear error, not an empty array.
    path = _write_clip(tmp_path, _gradient_clip(4, 16, 16))
    with pytest.raises(ValueError):
        io.VideoReader(path)[100:]


def test_non_uint8_frames_are_clipped(tmp_path):
    frames = _gradient_clip(5, 32, 32).astype(np.float32)
    path = tmp_path / "float.mp4"
    with io.VideoWriter(path, fps=10) as writer:
        writer.write_frames(frames)  # must not raise on float input
    assert io.VideoReader(path)[:].dtype == np.uint8


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
        io.VideoReader(batch)[:], io.VideoReader(incremental)[:]
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
    assert io.VideoReader(path)[:].shape[0] == len(frames)


def test_video_reader_fps(tmp_path):
    # open_reader resolves a video file to a VideoReader; fps comes from metadata.
    path = _write_clip(tmp_path, _gradient_clip(8, 16, 16))
    reader = io.open_reader(path)
    assert isinstance(reader, io.VideoReader)
    assert reader.fps() == pytest.approx(10.0, abs=0.5)


def test_reader_is_a_context_manager(tmp_path):
    # Readers work in a `with` block (symmetric with VideoWriter); close() is safe.
    frames = _gradient_clip(6, 16, 16)
    path = _write_clip(tmp_path, frames)
    with io.open_reader(path) as reader:
        assert reader[:].shape[0] == frames.shape[0]


# -- concatenation: several files as one stream ------------------------------
#
# The behavioral half of the v2 schema change, and the only one whose failure is silent: a
# frame-index bug after a part boundary succeeds and mislabels every subsequent frame.
# Nothing downstream can catch that, so the gate is byte-identity against the single-file
# reader over the SAME content -- decoded once as one clip, then split with `ffmpeg -f
# segment` (here, PyAV writing the halves, which is the same thing) and read back.


def _write_all_intra(path, frames, *, fps=10):
    """Encode ``frames`` with every picture a keyframe, so ANY split point is legal."""
    import av

    n, h, w = frames.shape[:3]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.gop_size = 1  # all-intra
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frames[i]), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def _segment(src, parts):
    """Split ``src`` into ``parts`` = [(path, n_frames), ...] by REMUXING its packets.

    A stream copy, exactly like ``ffmpeg -f segment``: no second encode, so the parts
    decode to the same pixels the whole file does. Re-encoding the halves instead would
    introduce a second lossy generation and turn a byte-identity gate into a tolerance.
    """
    import av

    with av.open(str(src)) as inp:
        packets = [
            pkt for pkt in inp.demux(inp.streams.video[0]) if pkt.dts is not None
        ]
        template = inp.streams.video[0]
        at = 0
        for path, n in parts:
            with av.open(str(path), mode="w") as out:
                ostream = out.add_stream_from_template(template)
                for pkt in packets[at : at + n]:
                    pkt.stream = ostream
                    out.mux(pkt)
            at += n
    return [path for path, _ in parts]


@pytest.fixture
def split_clip(tmp_path):
    """``(reader, whole)``: the same 30 frames as two remuxed parts, and as one array."""
    frames = _indexed_clip(n=30, h=32, w=48)
    whole_path = _write_all_intra(tmp_path / "whole.mp4", frames)
    whole = io.open_reader(whole_path)[:]
    parts = _segment(
        whole_path, [(tmp_path / "part_0.mp4", 17), (tmp_path / "part_1.mp4", 13)]
    )
    reader = io.open_reader(parts)
    yield reader, whole
    reader.close()


def test_open_reader_concatenates_several_videos(tmp_path, split_clip):
    reader, whole = split_clip
    assert isinstance(reader, io.ConcatReader)
    # The count is the SUM over parts, exact -- a global index is undefined otherwise.
    assert reader.count() == len(whole)


def test_a_single_video_is_still_a_plain_reader(tmp_path):
    path = _write_clip(tmp_path, _indexed_clip(n=4))
    assert isinstance(io.open_reader([path]), io.VideoReader)


def test_random_access_across_a_boundary_is_byte_identical(split_clip):
    reader, whole = split_clip
    # Every index, including the two either side of the boundary at 17.
    for i in range(len(whole)):
        np.testing.assert_array_equal(reader[i], whole[i], err_msg=f"frame {i}")


def test_a_slice_spanning_the_boundary_is_byte_identical(split_clip):
    reader, whole = split_clip
    for key in (slice(None), slice(5, 25), slice(16, 19), slice(0, 30, 3)):
        np.testing.assert_array_equal(reader[key], whole[key], err_msg=str(key))


def test_an_explicit_index_list_keeps_the_callers_order(split_clip):
    reader, whole = split_clip
    wanted = [29, 0, 17, 16, 17]
    np.testing.assert_array_equal(reader[wanted], whole[wanted])


def test_a_full_forward_decode_is_byte_identical(split_clip):
    reader, whole = split_clip
    np.testing.assert_array_equal(np.stack(list(reader.stream_frames())), whole)


def test_blocks_are_regrouped_across_the_boundary(split_clip):
    """No short block at a split: a consumer batching a forward pass would see one."""
    reader, whole = split_clip
    blocks = list(reader.stream_blocks(block_size=8))
    assert [len(b) for b in blocks] == [8, 8, 8, 6]
    np.testing.assert_array_equal(np.concatenate(blocks), whole)


def test_the_cursor_keeps_one_decoder_open_per_part(split_clip):
    reader, whole = split_clip
    cursor = reader.cursor()
    try:
        # Crossing the boundary swaps which cursor answers; neither is re-opened.
        for i in (0, 16, 17, 29, 17, 16):
            np.testing.assert_array_equal(cursor.frame(i), whole[i])
        assert len(cursor._cursors) == 2
    finally:
        cursor.close()


def test_an_index_outside_the_stream_is_an_error_naming_the_length(split_clip):
    reader, _ = split_clip
    with pytest.raises(IndexError, match="30-frame stream"):
        reader[30]


def test_parts_that_disagree_on_frame_size_are_refused(tmp_path):
    _write_clip(tmp_path, _indexed_clip(n=4, h=32, w=48), name="a.mp4")
    _write_clip(tmp_path, _indexed_clip(n=4, h=32, w=64), name="b.mp4")
    with pytest.raises(ValueError, match="agree on frame size"):
        io.open_reader([tmp_path / "a.mp4", tmp_path / "b.mp4"])


def test_parts_that_disagree_on_frame_rate_are_refused(tmp_path):
    frames = _indexed_clip(n=4)
    with io.VideoWriter(tmp_path / "a.mp4", fps=10) as w:
        w.write_frames(frames)
    with io.VideoWriter(tmp_path / "b.mp4", fps=25) as w:
        w.write_frames(frames)
    with pytest.raises(ValueError, match="agree on frame rate"):
        io.open_reader([tmp_path / "a.mp4", tmp_path / "b.mp4"])


def test_a_part_whose_header_carries_no_count_is_decoded_once(tmp_path, monkeypatch):
    """Never estimated from duration * fps: an off-by-one there shifts every later index.

    The cost is logged, because a full decode of a part is worth knowing about.
    """
    frames = _indexed_clip(n=10)
    _write_clip(tmp_path, frames[:6], name="a.mp4")
    _write_clip(tmp_path, frames[6:], name="b.mp4")
    real_count = io.VideoReader.count

    def blind(self):
        return None if self.path.name == "a.mp4" else real_count(self)

    monkeypatch.setattr(io.VideoReader, "count", blind)
    reader = io.open_reader([tmp_path / "a.mp4", tmp_path / "b.mp4"])
    assert reader.count() == 10
    # And the boundary is still in the right place.
    whole = io.open_reader(_write_clip(tmp_path, frames, name="whole.mp4"))[:]
    np.testing.assert_array_equal(reader[5], whole[5])
    np.testing.assert_array_equal(reader[6], whole[6])


# -- single-frame reads: seek, don't walk ------------------------------------
#
# `reader[i]` used to decode every frame from the start of the file and discard the ones
# before `i` -- `continue` in the consumer loop skips the array conversion, not the decode
# -- so it cost O(i): 472 ms for frame 2999 of a 3000-frame clip against 12 ms for the seek
# that finds it. The viewer reads exactly this way, one frame per camera, which is why it
# got slower the further into a recording the operator worked. These pin the seek as the
# path taken, and the walk as the fallback that is still there.


def test_single_index_seeks_rather_than_walking(tmp_path, monkeypatch):
    # With the walk sabotaged, every single-index read must still answer -- proving none of
    # them goes near it.
    frames = _indexed_clip(12, 32, 32)
    path = _write_clip(tmp_path, frames)
    reader = io.VideoReader(path)
    full = reader[:]

    def walked(*a, **k):
        pytest.fail("a single index walked the file instead of seeking")

    monkeypatch.setattr(io.VideoReader, "_decode_range", walked)
    for i in range(len(full)):
        np.testing.assert_allclose(reader[i].mean(), full[i].mean(), atol=3)


def test_single_index_falls_back_to_the_walk(tmp_path, monkeypatch):
    # A container that cannot seek (or carries no timestamps) still reads: the walk is slow,
    # not wrong, and losing it would turn a slow viewer into a broken one.
    frames = _indexed_clip(8, 32, 32)
    path = _write_clip(tmp_path, frames)
    reader = io.VideoReader(path)
    expected = reader[5]

    def no_seeking(*a, **k):
        raise ValueError("cannot seek this container")

    monkeypatch.setattr(io.VideoReader, "_decode_indices", no_seeking)
    np.testing.assert_allclose(reader[5].mean(), expected.mean(), atol=3)


# -- cursors: the decoder held open between reads ----------------------------


def _mono_clip(n=12, h=32, w=32):
    """A clip with no color: equal RGB channels, so both chroma planes sit at 128."""
    vals = (np.arange(n) * 15 + 20).clip(0, 255)
    return np.broadcast_to(vals[:, None, None, None], (n, h, w, 3)).astype(np.uint8)


def _color_clip(n=12, h=32, w=32, *, gray_first=False):
    """A clip that really is colored -- optionally with a gray first frame."""
    frames = np.zeros((n, h, w, 3), np.uint8)
    frames[:, :, :, 0] = 200
    frames[:, :, :, 2] = 40
    if gray_first:
        frames[0] = 90
    return frames


def test_cursor_matches_indexing_in_any_order(tmp_path):
    # The cursor is an optimization, so its frames must be the reader's frames -- including
    # after a jump backwards, which is where it has to abandon its generator and re-seek.
    frames = _indexed_clip(12, 32, 32)
    path = _write_clip(tmp_path, frames)
    reader = io.VideoReader(path)
    full = reader[:]
    cursor = reader.cursor()
    try:
        for i in [0, 1, 2, 7, 8, 3, 11, 10, 9, 4, 4]:
            np.testing.assert_allclose(cursor.frame(i).mean(), full[i].mean(), atol=3)
    finally:
        cursor.close()


def test_cursor_steps_forward_without_seeking(tmp_path):
    # The whole point: a step to the NEXT frame continues the live generator, so a run of
    # forward steps costs one seek in total. Counting `_seek_to` counts seeks -- it is the
    # only place the cursor issues one.
    frames = _indexed_clip(10, 32, 32)
    path = _write_clip(tmp_path, frames)
    cursor = io.VideoReader(path).cursor()
    seeks = 0
    inner = type(cursor)._seek_to

    def counted(self, idx):
        nonlocal seeks
        seeks += 1
        return inner(self, idx)

    try:
        type(cursor)._seek_to = counted  # type: ignore[method-assign]
        for i in range(2, 9):
            cursor.frame(i)
        assert seeks == 1, f"stepping forward re-seeked {seeks} times"
        # Backwards: a codec can only walk forward, so this one has to re-seek.
        cursor.frame(4)
        assert seeks == 2
    finally:
        type(cursor)._seek_to = inner  # type: ignore[method-assign]
        cursor.close()


def test_cursor_gives_gray_only_for_pictures_with_no_color(tmp_path):
    # Monochrome footage (every fly rig) is stored as planar YUV with flat chroma. Taking it
    # as one channel skips the channel-reversing copy, three quarters of the cost of serving
    # a frame, and two thirds of what caching one costs -- but only where there is no color.
    mono = io.VideoReader(_write_clip(tmp_path, _mono_clip(), name="mono.mp4"))
    color = io.VideoReader(_write_clip(tmp_path, _color_clip(), name="color.mp4"))
    for reader, ndim in ((mono, 2), (color, 3)):
        cursor = reader.cursor(gray_ok=True)
        try:
            assert cursor.frame(0).ndim == ndim
            assert cursor.frame(5).ndim == ndim
        finally:
            cursor.close()


def test_cursor_decides_color_per_frame_not_per_file(tmp_path):
    # The trap that rules out remembering one verdict per camera: a color video whose FIRST
    # frame is uniformly gray has flat chroma there and color everywhere after it. A cached
    # verdict would serve the rest of that recording stripped of its color.
    path = _write_clip(tmp_path, _color_clip(gray_first=True), name="latecolor.mp4")
    cursor = io.VideoReader(path).cursor(gray_ok=True)
    try:
        assert cursor.frame(0).ndim == 2, "a gray frame should be taken as gray"
        assert cursor.frame(6).ndim == 3, "the color after it must survive"
    finally:
        cursor.close()


def test_cursor_keeps_rgb_unless_gray_is_allowed(tmp_path):
    # `gray_ok` is permission, and the default is no: a caller that has not said it accepts
    # one channel keeps getting three.
    path = _write_clip(tmp_path, _mono_clip(), name="mono2.mp4")
    cursor = io.VideoReader(path).cursor()
    try:
        assert cursor.frame(3).shape[-1] == 3
    finally:
        cursor.close()


def test_gray_blocks_are_the_luma_of_the_rgb_blocks(tmp_path):
    # The point of the batch gray path: it is not an approximation of the RGB decode, it is
    # the same numbers with the redundancy left out. On full-range monochrome footage
    # R == G == B == Y, and PIL's integer luma of three equal channels returns them
    # unchanged (19595 + 38470 + 7471 == 65536, so (v*65536) >> 16 == v), so the single
    # channel the decoder hands over IS what the detector used to reconstruct with a PIL
    # convert("L") after paying for the YUV->RGB conversion.
    PIL = pytest.importorskip("PIL.Image")
    vals = (np.arange(10) * 15 + 20).clip(0, 255).astype(np.uint8)
    path = _write_rig_clip(
        tmp_path, np.broadcast_to(vals[:, None, None], (10, 32, 32)).copy()
    )
    gray = np.concatenate(
        list(io.VideoReader(path).stream_blocks(block_size=5, gray_ok=True))
    )
    rgb = np.concatenate(list(io.VideoReader(path).stream_blocks(block_size=5)))
    assert gray.shape[-1] == 1, (
        "the channel axis is KEPT, so ops indexing it still work"
    )
    assert gray.shape[:-1] == rgb.shape[:-1]
    luma = np.stack(
        [np.asarray(PIL.fromarray(f, mode="RGB").convert("L")) for f in rgb]
    )
    np.testing.assert_array_equal(gray[..., 0], luma)


def test_gray_is_refused_for_tv_range_footage(tmp_path):
    # The trap the equivalence rests on. In TV range the YUV->RGB conversion EXPANDS
    # 16..235 to 0..255, so the luma plane is a different set of numbers from the RGB the
    # detector is calibrated on -- 33 against 19 on the same pixel, which would shift every
    # input. Colour-free is not enough; the range has to be full too, and it is the writer's
    # own default that is not.
    tv = _write_clip(tmp_path, _mono_clip(), name="mono_tv.mp4")  # yuv420p, TV range
    block = next(io.VideoReader(tv).stream_blocks(block_size=4, gray_ok=True))
    assert block.shape[-1] == 3, (
        "TV-range footage must keep going through the conversion"
    )


def test_gray_blocks_keep_color_where_there_is_color(tmp_path):
    # Permission, not a promise: colour footage keeps its colour even where gray is allowed.
    vals = (np.arange(8) * 15 + 20).clip(0, 255).astype(np.uint8)
    mono = _write_rig_clip(
        tmp_path, np.broadcast_to(vals[:, None, None], (8, 32, 32)).copy(), name="m.mp4"
    )
    color = _write_clip(tmp_path, _color_clip(), name="c.mp4", pix_fmt="yuvj420p")
    assert (
        next(io.VideoReader(mono).stream_blocks(block_size=4, gray_ok=True)).shape[-1]
        == 1
    )
    assert (
        next(io.VideoReader(color).stream_blocks(block_size=4, gray_ok=True)).shape[-1]
        == 3
    )


def test_gray_blocks_stack_a_mixed_block_at_three_channels(tmp_path):
    # Colour is decided per FRAME, so a clip whose opening frame is neutral and whose rest
    # is coloured puts both kinds in one block. It must be stacked, not refused -- the gray
    # frame is broadcast, which is what it means.
    from deeperfly.io import video as video_mod

    gray = np.full((32, 32, 1), 90, np.uint8)
    color = np.zeros((32, 32, 3), np.uint8)
    color[..., 0] = 200
    stacked = video_mod._stack_frames([gray, color, gray])
    assert stacked.shape == (3, 32, 32, 3)
    np.testing.assert_array_equal(stacked[0], np.repeat(gray, 3, axis=-1))
    np.testing.assert_array_equal(stacked[1], color)


def test_stream_blocks_thread_count_does_not_change_the_frames(tmp_path):
    # Frame threading is bit-exact, so capping the decode pool (which is what keeps eight
    # concurrent cameras from oversubscribing the host) is free of consequence.
    frames = _indexed_clip(14, 32, 32)
    path = _write_clip(tmp_path, frames)
    ref = np.concatenate(list(io.VideoReader(path).stream_blocks(block_size=5)))
    for threads in (1, 2, 8, None):
        got = np.concatenate(
            list(io.VideoReader(path).stream_blocks(block_size=5, thread_count=threads))
        )
        np.testing.assert_array_equal(got, ref)


def test_cursor_close_is_idempotent_and_final(tmp_path):
    # `FrameSource.release_cache` closes cursors for a recording nobody is showing, possibly
    # twice and possibly while a request is arriving; neither may crash the server.
    path = _write_clip(tmp_path, _indexed_clip(6, 32, 32), name="c.mp4")
    cursor = io.VideoReader(path).cursor()
    cursor.frame(1)
    cursor.close()
    cursor.close()
    with pytest.raises(ValueError, match="closed"):
        cursor.frame(1)


def test_image_sequence_cursor_delegates(tmp_path):
    # Every source has a cursor so callers need not know which kind they hold; an image
    # sequence has no decoder to keep open, so its cursor is the reader.
    frames = _indexed_clip(6, 16, 16)
    _write_images(tmp_path, frames, ext="png")
    reader = io.open_reader(tmp_path)
    cursor = reader.cursor(gray_ok=True)
    try:
        assert isinstance(cursor, io.FrameCursor)
        np.testing.assert_array_equal(cursor.frame(4), frames[4])
    finally:
        cursor.close()


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


def test_image_reader_no_fps(tmp_path):
    _write_images(tmp_path, _gradient_clip(3, 16, 16), ext="png")
    reader = io.open_reader(tmp_path)
    assert isinstance(reader, io.ImageSequenceReader)
    assert reader.fps() is None  # image sequences carry no frame rate


def test_read_images_parallel_rgb(tmp_path):
    frames = _gradient_clip(6, 40, 50)
    _write_images(tmp_path, frames, ext="png")
    out = io.ImageSequenceReader.from_pattern(tmp_path)[:]
    assert out.shape == (6, 40, 50, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, frames)  # PNG is lossless
    # worker count must not change the result
    single = io.ImageSequenceReader.from_pattern(tmp_path, workers=1)[:]
    np.testing.assert_array_equal(single, out)


def test_read_images_opencv(tmp_path):
    frames = _gradient_clip(4, 24, 32)
    _write_images(tmp_path, frames, ext="png")
    out = io.ImageSequenceReader.from_pattern(tmp_path)[:]
    assert out.shape == (4, 24, 32, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out, frames)


def test_read_images_grayscale_broadcasts_to_rgb(tmp_path):
    # A grayscale (H, W) PNG must broadcast to 3 equal channels, NOT slice width.
    gray = (np.arange(20 * 30).reshape(20, 30) % 255).astype(np.uint8)
    _write_images(tmp_path, gray[None], ext="png", name="g")
    out = io.ImageSequenceReader.from_pattern(tmp_path)[:]
    assert out.shape == (1, 20, 30, 3)
    np.testing.assert_array_equal(out[0, ..., 0], out[0, ..., 2])
    np.testing.assert_array_equal(out[0, ..., 0], gray)


def test_read_images_indices_and_slice(tmp_path):
    frames = _indexed_clip(10, 16, 16)
    _write_images(tmp_path, frames, ext="png")
    reader = io.ImageSequenceReader.from_pattern(tmp_path)
    np.testing.assert_array_equal(reader[[0, 3, 7]], frames[[0, 3, 7]])
    np.testing.assert_array_equal(reader[1:9:2], frames[1:9:2])


def test_open_reader_dispatches_dir_vs_video(tmp_path):
    frames = _indexed_clip(6, 32, 32)
    _write_images(tmp_path, frames, ext="png")
    from_dir = io.open_reader(tmp_path)[:]
    assert from_dir.shape == (6, 32, 32, 3)
    np.testing.assert_array_equal(from_dir, frames)
    mp4 = _write_clip(tmp_path, frames, name="clip.mp4")
    reader = io.open_reader(mp4)
    assert isinstance(reader, io.VideoReader)  # routed to the video reader
    assert reader[:].shape[0] == 6


def test_stream_frames_image_sequence_blocks(tmp_path):
    # The image reader's stream_blocks() yields the sorted sequence in <= block_size chunks
    # (PNG is lossless, so frame identity and counts are exact).
    frames = _indexed_clip(7, 16, 16)
    _write_images(tmp_path, frames, ext="png")
    blocks = list(io.open_reader(tmp_path).stream_blocks(block_size=3))
    assert [len(b) for b in blocks] == [3, 3, 1]
    np.testing.assert_array_equal(np.concatenate(blocks), frames)


def test_read_images_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.open_reader(tmp_path / "empty")


# --- CursorFrames: a clip's array surface without the clip in memory -------------------


def test_cursor_frames_matches_the_eager_decode_in_any_order(tmp_path):
    # The whole contract: `frames[t]` must be `reader[:][t]`, for reads in any order --
    # forward runs, out-of-order reads inside the window, and jumps backwards out of it,
    # which is where it has to let the cursor re-seek.
    frames = _indexed_clip(20, 32, 32)
    path = _write_clip(tmp_path, frames)
    full = io.VideoReader(path)[:]
    lazy = io.CursorFrames(io.VideoReader(path), window=4)
    try:
        assert lazy.shape == full.shape  # states the clip it stands for
        assert len(lazy) == len(full)
        assert lazy.dtype == full.dtype
        for i in [0, 1, 2, 3, 4, 9, 8, 7, 19, 0, 10, 11, 12, 2]:
            np.testing.assert_array_equal(lazy[i], full[i])
        np.testing.assert_array_equal(lazy[-1], full[-1])  # negative index
    finally:
        lazy.close()


def test_cursor_frames_holds_only_its_window(tmp_path):
    # The reason this class exists: a walk of the whole clip must never accumulate it.
    frames = _indexed_clip(40, 32, 32)
    path = _write_clip(tmp_path, frames)
    lazy = io.CursorFrames(io.VideoReader(path), window=6)
    try:
        for i in range(40):
            lazy[i]
            assert len(lazy._cache) <= 6
    finally:
        lazy.close()


def test_cursor_frames_walks_forward_instead_of_seeking(tmp_path):
    # A near-sequential consumer must not pay a seek per frame: a gap inside the window is
    # closed by stepping (and every frame stepped over is cached, so the consumer's
    # out-of-order look-ahead reads are free). Counting `_seek_to` counts seeks.
    frames = _indexed_clip(30, 32, 32)
    path = _write_clip(tmp_path, frames)
    lazy = io.CursorFrames(io.VideoReader(path), window=4)
    seeks = []
    cursor = lazy._cursor
    real = cursor._seek_to
    cursor._seek_to = lambda idx: (seeks.append(idx), real(idx))[1]
    try:
        for i in range(1, 20):  # strictly forward
            lazy[i]
        assert seeks == []  # priming already decoded frame 0; the rest are steps
        # An out-of-order read still inside the window is served from the cache.
        lazy[17]
        assert seeks == []
        # A jump FURTHER than the window seeks once, rather than decoding everything
        # in between -- 19 -> 29 is a gap of 10 against a window of 4.
        lazy[29]
        assert seeks == [29]
        # A gap within the window walks instead, and caches what it steps over.
        seeks.clear()
        lazy[2]  # backwards, long since evicted: one seek
        lazy[5]  # gap of 3 <= 4: stepped, so 3 and 4 land in the cache too
        assert seeks == [2]
        assert {3, 4}.issubset(lazy._cache)
    finally:
        cursor._seek_to = real
        lazy.close()


def test_cursor_frames_refuses_slices_and_out_of_range(tmp_path):
    # It serves one frame at a time on purpose; a slice would rebuild what it avoids.
    frames = _indexed_clip(5, 16, 16)
    lazy = io.CursorFrames(io.VideoReader(_write_clip(tmp_path, frames)))
    try:
        with pytest.raises(TypeError, match="one frame at a time"):
            lazy[0:2]
        with pytest.raises(IndexError, match="out of range"):
            lazy[5]
        with pytest.raises(IndexError, match="out of range"):
            lazy[-6]
    finally:
        lazy.close()


def test_cursor_frames_needs_a_known_frame_count(tmp_path):
    # `shape` has to state a count, so an unknown one is refused rather than guessed --
    # unlike `count()` itself, which is a progress-bar hint and may be None.
    class _NoCount(io.VideoReader):
        def count(self):
            return None

    path = _write_clip(tmp_path, _indexed_clip(4, 16, 16))
    with pytest.raises(ValueError, match="frame count is unknown"):
        io.CursorFrames(_NoCount(path))
    # ... but a caller who knows it from elsewhere may supply it.
    lazy = io.CursorFrames(_NoCount(path), n_frames=4)
    try:
        assert lazy.shape[0] == 4
    finally:
        lazy.close()


def test_cursor_frames_is_a_context_manager(tmp_path):
    path = _write_clip(tmp_path, _indexed_clip(4, 16, 16))
    with io.CursorFrames(io.VideoReader(path)) as lazy:
        assert lazy[0].shape == (16, 16, 3)
    assert lazy._cache == {}  # closing releases the window too
