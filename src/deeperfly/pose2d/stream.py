"""Detector loading, frame-rate resolution and streaming 2D detection."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..config import Config
from ..recordings import source_sources

log = logging.getLogger("deeperfly")


@contextmanager
def _null_progress(total, description):
    """A no-op progress factory: a context manager yielding a pass-through wrapper.

    The library default for :func:`detect_2d` (and the render stage), so a caller
    that passes no ``progress`` factory gets no progress bar and pulls in no Rich
    dependency on the hot path. The CLI injects a real factory instead. The factory
    contract is ``progress(total, description)`` returning a context manager that
    yields ``wrap``, where ``wrap(rng)`` iterates ``rng`` (advancing the bar once
    per item).

    Parameters
    ----------
    total
        The task's total item count (ignored here).
    description
        The task label (ignored here).

    Yields
    ------
    wrap : callable
        ``wrap(rng)`` yields each item of ``rng`` unchanged.
    """

    def wrap(rng):
        yield from rng

    yield wrap


def load_models(plan) -> dict:
    """Load every model the plan references -> ``name -> LoadedModel``.

    Each model's weights are loaded once (downloading the cached DeepFly2D
    checkpoint on demand for a model with no explicit ``weights``); a pathway
    then forwards through the model named in its ``model`` key.

    Parameters
    ----------
    plan
        The detection plan (:class:`~deeperfly.pose2d.pathways.DetectionPlan`).

    Returns
    -------
    dict of str to LoadedModel
        The loaded models, keyed by name.
    """
    from .models import load_model

    loaded = {name: load_model(spec) for name, spec in plan.models.items()}
    for name, model in loaded.items():
        _check_channel_names(name, model, plan)
    return loaded


def _check_channel_names(name: str, model, plan) -> None:
    """Refuse a model whose recorded channel order is not this config's skeleton.

    A dense detector's channels ARE a skeleton, and a count check cannot tell two
    skeletons apart: ``fly38`` and the DeepFly3D set are both 38 points, share 32 of them, and
    are in a different order, so routing one through the other's config attaches six
    points to the wrong joints and shifts the rest -- a wrong limb, not a crash.

    This runs on EVERY run, which is the point. The same comparison exists in
    a generator, but a generator only ever sees the moment it writes the
    file; it cannot see a ``weights`` path later repointed at a different checkpoint, a
    ``[skeleton]`` swapped underneath, or a mapping line edited by hand. Those are exactly
    the cases where the channel order silently stops meaning what the config says.

    Only checkpoints that RECORD their channel names can be checked. The shipped
    19-channel hourglass does not carry any -- its channels are one side of the animal and
    mean different points in different views, which is why such a config declares an
    explicit ``[pose2d.output_points]`` table instead of relying on the identity default.
    """
    names = list(getattr(model.module, "point_names", None) or [])
    if not names or not plan.point_names:
        return
    skeleton = list(plan.point_names)
    if names == skeleton:
        return
    extra = [n for n in names if n not in skeleton]
    gone = [n for n in skeleton if n not in names]
    raise SystemExit(
        f"model {name!r} was trained on a different channel set from this config's "
        f"skeleton, so its channels would attach points to the wrong joints.\n"
        f"  config skeleton : {len(skeleton)} points\n"
        f"  model           : {len(names)} points\n"
        f"  only in model   : {extra or 'none (the ORDER differs)'}\n"
        f"  only in config  : {gone or 'none (the ORDER differs)'}\n"
        f"Stamp the matching skeleton into the config "
        f"([skeleton] name, or file = the skeleton.toml the model was trained on) or "
        f"point 'weights' at a "
        f"checkpoint trained on this one."
    )


# -- frame-rate resolution ---------------------------------------------------


#: Frame rate used when none can be detected from the recording (e.g. an image
#: sequence carries no intrinsic rate). Matches the historical default.
_FPS_FALLBACK = 100.0


def detect_input_fps(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> float | None:
    """First detectable per-camera video frame rate, or ``None``.

    Walks the configured camera sources and returns the first video file's frame
    rate (:meth:`deeperfly.io.VideoReader.fps`); image-sequence cameras have none.
    Guarded so a missing recording (a cache-only resume) yields ``None`` rather
    than raising.

    Parameters
    ----------
    config
        The run config.
    sources, input
        The footage to resolve (see :func:`deeperfly.recordings.camera_sources`).

    Returns
    -------
    float or None
        The first detectable video frame rate, or ``None``.
    """
    from .. import io

    try:
        cam_sources = [
            src for _, src in source_sources(config, sources=sources, input=input)
        ]
    except SystemExit:
        return None
    for src in cam_sources:
        try:
            fps = io.open_reader(src).fps()
        except Exception:  # noqa: BLE001
            fps = None
        if fps:
            return float(fps)
    return None


def resolve_fps(
    config: Config, *, sources: dict[str, list[Path]] | None = None, input=None
) -> float:
    """The recording's frame rate, used as the visualization base playback rate.

    Detects the rate from the input videos (:func:`detect_input_fps`), falling
    back to :data:`_FPS_FALLBACK` when none is available -- e.g. an image
    sequence, or a cache-only resume with no recording -- logging a hint to set
    ``[visualization].output_fps`` for an explicit output rate.

    Parameters
    ----------
    config
        The run config.
    sources, input
        The footage to resolve (see :func:`deeperfly.recordings.camera_sources`).

    Returns
    -------
    float
        The resolved frame rate.
    """
    detected = detect_input_fps(config, sources=sources, input=input)
    if detected is not None:
        log.info("detected input fps %.4g from the recording", detected)
        return detected
    log.warning(
        "could not detect the input fps (image sequence, or no recording available); "
        "using %g fps -- set [visualization].output_fps to override",
        _FPS_FALLBACK,
    )
    return _FPS_FALLBACK


def decode_thread_budget(n_sources: int) -> int:
    """Decode threads per stream when ``n_sources`` of them are decoded at once.

    ``thread_type = "AUTO"`` alone sizes each stream's pool from the host's core count --
    16 on a 32-core box -- so a rig's eight cameras open 128 frame threads for 32 cores and
    spend their time in the scheduler. Dividing the cores between the streams keeps the
    total pool at roughly one thread per core, with a floor of 2 so a single walk never
    serializes (see :func:`deeperfly.io.video._cursor_thread_count`, which makes the same
    trade for the viewer's seeks).
    """
    import os

    return max(2, (os.cpu_count() or 4) // max(1, n_sources))


def prefetch_windows(
    sources,
    *,
    block,
    transforms=None,
    depth=1,
    workers=None,
    gray_ok=False,
):
    """Yield ``(window, n)`` multi-camera frame blocks from continuous decode.

    A background producer opens **one continuous forward decoder per source**
    (:meth:`deeperfly.io.FrameReader.stream_blocks`) and walks them all together, grouping
    ``block`` frames at a time into a multi-camera ``window`` (a list of
    ``(T, H, W, 3)`` arrays, one per source). Each source is decoded in a single
    linear pass -- no per-window re-open or re-seek -- overlapped with the GPU
    forward.

    ``depth`` bounds the queue: when it is full the producer blocks on ``put`` and
    the (lazy) decoders suspend mid-stream, so a decoder faster than the GPU runs
    at most ``depth`` blocks ahead rather than filling memory. Peak frame memory is
    therefore ``~(depth + 2)`` blocks (queue + the one the producer is blocked
    enqueueing + the one the consumer is forwarding), independent of recording
    length. A deeper queue absorbs more decode jitter; the detector sets ``block``
    to the forward batch and ``depth`` to ``[pose2d] decode_buffer`` (see
    :class:`~deeperfly.config.Pose2dParams`).

    ``transforms`` is an optional per-source
    :class:`~deeperfly.preprocessing.FrameTransform` (aligned to ``sources``)
    applied to each block before yielding, so detection sees the canonical
    (pathway-transformed) frames.

    The producer treats each source as an opaque forward stream -- it never asks
    for a frame count or a seek -- so an unbounded *live-camera* source (a future
    :class:`~deeperfly.io.base.FrameReader` subclass) drives this loop unchanged.

    Parameters
    ----------
    sources
        One footage source per camera (each opened with
        :func:`deeperfly.io.open_reader` and streamed).
    block
        Frames per yielded window (the detector's forward batch).
    transforms
        Optional per-source :class:`~deeperfly.preprocessing.FrameTransform`
        (aligned to ``sources``) applied to each block, so detection sees the
        canonical (pathway-transformed) frames.
    depth
        Queue depth bounding how far the decoder runs ahead of the GPU.
    workers
        Optional worker count for image-sequence decode.
    gray_ok
        Permission to decode color-free footage as one channel -- ``(T, H, W, 1)`` rather
        than ``(T, H, W, 3)``. Granted only when every model that reads these frames wants
        grayscale anyway (see :meth:`deeperfly.pose2d.models.LoadedModel.accepts_gray`),
        because it skips the decoder's YUV->RGB conversion, which is the largest single
        cost in the decode, and thirds every byte after it.

    Yields
    ------
    window : list of np.ndarray
        One ``(T, H, W, C)`` block per source, aligned across cameras.
    n : int
        The number of frames in the window.

    Notes
    -----
    EOF: the first source to run out (a short or exhausted block) ends the stream;
    a read failure before any window is emitted propagates, later ones are EOF.
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from .. import io, preprocessing

    if transforms is None:
        transforms = [preprocessing.FrameTransform()] * len(sources)
    q: queue.Queue = queue.Queue(maxsize=depth)
    DONE = object()

    def produce():
        emitted = False
        # One worker per source: PyAV releases the GIL during decode, so the
        # cameras' next blocks decode concurrently instead of one after another
        # (with the per-stream FFmpeg threading of VideoReader, ~10x the serial
        # decode ceiling on a multi-core host). The transform.apply is fanned out
        # over the same pool.
        pool = ThreadPoolExecutor(max_workers=max(1, len(sources)))
        threads = decode_thread_budget(len(sources))
        try:
            streams = [
                io.open_reader(s, workers=workers).stream_blocks(
                    block_size=block, gray_ok=gray_ok, thread_count=threads
                )
                for s in sources
            ]
            while True:
                blocks = list(pool.map(lambda s: next(s, None), streams))
                # A source ran dry (None) or yielded nothing -> end of recording.
                if any(b is None or len(b) == 0 for b in blocks):
                    q.put(DONE)
                    return
                n = min(len(b) for b in blocks)  # align cameras (synced rigs match)
                window = list(
                    pool.map(lambda tb: tb[0].apply(tb[1][:n]), zip(transforms, blocks))
                )
                q.put(("win", window, n))
                emitted = True
                if any(len(b) < block for b in blocks):  # a short block is the last
                    q.put(DONE)
                    return
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc) if not emitted else DONE)
        finally:
            pool.shutdown(wait=False)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = q.get()
        if item is DONE:
            return
        if item[0] == "err":
            raise item[1]
        yield item[1], item[2]


def _prepared_windows(stream, src_names, plan, models, *, depth: int = 1):
    """Yield ``(windows, prepared)``, preparing one window ahead of the consumer.

    The two halves of a detection step run on different processors and neither needs the
    other: resizing and normalizing a window is host work (cv2, a copy), forwarding it is
    the GPU's. Run one after the other and each waits for the other -- on this rig the
    preparation is a third of the step, spent with the GPU idle. So a worker thread
    prepares the *next* window while the caller is still forwarding the current one, and
    the preparation stops costing wall time until it grows past the network.

    ``depth`` bounds how many prepared windows may be waiting, and with it the peak memory:
    the pathway inputs are small next to the frames they came from, but the *frames* are
    held too (the coordinate inversion needs each source's size), so this stays shallow
    and the decode queue upstream does the real buffering.

    The pathway pool is created once here rather than per window: one pool per two frames
    would spend a recording starting threads.
    """
    import queue
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from . import inference

    q: queue.Queue = queue.Queue(maxsize=max(1, depth))
    DONE = object()
    pool = ThreadPoolExecutor(max_workers=max(1, len(plan.pathways)))

    def produce():
        try:
            for window, _ in stream:
                windows = {name: window[i] for i, name in enumerate(src_names)}
                prepared = inference.prepare_pathways(plan, models, windows, pool=pool)
                q.put(("win", windows, prepared))
            q.put(DONE)
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the consumer's thread
            q.put(("err", exc))

    threading.Thread(target=produce, daemon=True).start()
    try:
        while True:
            item = q.get()
            if item is DONE:
                return
            if item[0] == "err":
                raise item[1]
            yield item[1], item[2]
    finally:
        pool.shutdown(wait=False)


def detect_2d(
    config: Config,
    plan,
    models: dict,
    *,
    sources: dict[str, list[Path]] | None = None,
    input=None,
    want_candidates,
    k,
    progress=None,
):
    """Stream 2D detection over decode blocks -> ``(pts2d, conf, candidates)``.

    Decodes each **source** in one continuous forward pass (CPU), handing the
    detector one ``[pose2d] batch_size``-frame block at a time and
    freeing it before the next, so peak frame memory is bounded by the decode
    buffer, not the recording length. Each block feeds every pathway on that
    source (the front source is decoded once, read by both its pathways).
    Per-block results are concatenated along time. End-of-file comes from the
    decoder (a short or exhausted block), so it doesn't depend on
    :meth:`deeperfly.io.FrameReader.count` being exact -- that is only the
    progress-bar total.

    Parameters
    ----------
    config
        The run config (I/O backends, batch size, decode buffer).
    plan
        The detection plan (:class:`~deeperfly.pose2d.pathways.DetectionPlan`).
    models
        ``name -> LoadedModel`` for every model the plan references.
    sources, input
        The footage to detect over (see
        :func:`deeperfly.recordings.source_sources`).
    want_candidates
        Whether to also extract the top-K candidate peaks (for pictorial
        structures, which are not cached).
    k
        Number of candidate peaks per joint when ``want_candidates``.
    progress
        Optional progress factory ``progress(total, description) -> (wrap, close)``;
        defaults to :func:`_null_progress` (no bar). The CLI injects a Rich-backed
        factory.

    Returns
    -------
    pts2d : np.ndarray
        Detected 2D of shape ``(V, T, P, 2)``.
    conf : np.ndarray
        Per-point confidence of shape ``(V, T, P)``.
    candidates : deeperfly.pictorial.Candidates or None
        The top-K candidate set when ``want_candidates``, else ``None``.

    Raises
    ------
    SystemExit
        If the detector received no frames.
    """
    from .. import io
    from ..pictorial import Candidates
    from ..recordings import source_sources
    from . import inference

    pose2d = config.pose2d
    workers = config.io.image_workers
    # Two knobs: the GPU forward batch (images/forward), and the decode buffer in
    # multiples of it. A block holds one batch of frames; the reader keeps up to
    # `depth` of them queued (>= 1 so the queue stays bounded -- 0 is unbounded).
    batch_size = pose2d.batch_size
    depth = pose2d.decode_buffer
    block = batch_size
    src_list = source_sources(config, sources=sources, input=input)
    src_names = [name for name, _ in src_list]
    src_files = [files for _, files in src_list]
    # One head reader for the first source serves the progress-bar total -- the
    # source kind is resolved once here.
    head = io.open_reader(src_files[0]) if src_files else None
    total = head.count() if head is not None else 0

    # Grayscale is asked for only when EVERY model would throw the color away itself, so
    # a plan mixing the multiview transformer with a 3-channel detector decodes RGB as
    # before. It stays a permission: footage that carries color is decoded as RGB whatever
    # this says, and the models see what they always saw.
    gray_ok = bool(models) and all(
        # getattr: `models` is duck-typed on this path too, and a stub that says nothing
        # gets colour, which is what every caller got before.
        bool(getattr(m, "accepts_gray", False))
        for m in models.values()
    )
    log.info(
        "streaming frames: forward batch %d, decode buffer %d batches (%d frames/source), "
        "%d decode thread(s)/source, grayscale decode %s",
        batch_size,
        depth,
        depth * batch_size,
        decode_thread_budget(len(src_files)),
        "requested" if gray_ok else "off (a model wants color)",
    )

    make_progress = progress or _null_progress
    pts_parts, conf_parts, cand_xy, cand_score = [], [], [], []

    stream = prefetch_windows(
        src_files,
        block=block,
        depth=depth,
        workers=workers,
        gray_ok=gray_ok,
    )
    with make_progress(total, "detect 2D") as wrap:
        if want_candidates:
            # The candidate path forwards frame by frame and keeps whole heatmaps, so its
            # cost is the network and not the preparation; it is left on the simple loop.
            for window, _ in stream:
                windows = {name: window[i] for i, name in enumerate(src_names)}
                p, c, cand = inference.detect_candidates_sequence(
                    plan, models, windows, k=k, progress=wrap
                )
                cand_xy.append(cand.xy)
                cand_score.append(cand.score)
                pts_parts.append(p)
                conf_parts.append(c)
                del window
        else:
            for windows, prepared in _prepared_windows(stream, src_names, plan, models):
                p, c = inference.detect_sequence(
                    plan,
                    models,
                    windows,
                    batch_size=batch_size,
                    progress=wrap,
                    prepared=prepared,
                )
                pts_parts.append(p)
                conf_parts.append(c)
                del windows, prepared  # release before the next window is consumed

    if not pts_parts:
        raise SystemExit("detector received no frames")
    pts2d = np.concatenate(pts_parts, axis=1)
    conf = np.concatenate(conf_parts, axis=1)
    candidates = (
        Candidates(
            xy=np.concatenate(cand_xy, axis=1), score=np.concatenate(cand_score, axis=1)
        )
        if cand_xy
        else None
    )
    return pts2d, conf, candidates
