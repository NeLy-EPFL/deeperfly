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

    return {name: load_model(spec) for name, spec in plan.models.items()}


# -- frame-rate resolution ---------------------------------------------------


#: Frame rate used when ``[pipeline].fps`` is unset and none can be detected from
#: the recording (e.g. an image sequence carries no intrinsic rate). Matches the
#: historical default.
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

    Uses ``[pipeline].fps`` when set; otherwise detects it from the input videos
    (:func:`detect_input_fps`). Falls back to :data:`_FPS_FALLBACK` when neither
    is available -- e.g. an image sequence, or a cache-only resume with no
    recording -- logging a hint to set ``[pipeline].fps`` explicitly.

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
    if config.fps is not None:
        return config.fps
    detected = detect_input_fps(config, sources=sources, input=input)
    if detected is not None:
        log.info("detected input fps %.4g from the recording", detected)
        return detected
    log.warning(
        "could not detect the input fps (image sequence, or no recording available); "
        "using %g fps -- set [pipeline].fps to override",
        _FPS_FALLBACK,
    )
    return _FPS_FALLBACK


def prefetch_windows(
    sources,
    *,
    block,
    transforms=None,
    depth=1,
    workers=None,
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
    to the forward batch and ``depth`` to ``[pipeline.pose2d] decode_buffer`` (see
    :class:`~deeperfly.config.Pose2dParams`).

    ``transforms`` is an optional per-source
    :class:`~deeperfly.preprocessing.FrameTransform` (aligned to ``sources``)
    applied to each block before yielding, so detection sees the canonical
    (``[cameras.*].preprocess``-transformed) frames.

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
        canonical (``[cameras.*].preprocess``-transformed) frames.
    depth
        Queue depth bounding how far the decoder runs ahead of the GPU.
    workers
        Optional worker count for image-sequence decode.

    Yields
    ------
    window : list of np.ndarray
        One ``(T, H, W, 3)`` block per source, aligned across cameras.
    n : int
        The number of frames in the window.

    Notes
    -----
    EOF: the first source to run out (a short or exhausted block) ends the stream;
    a read failure before any window is emitted propagates, later ones are EOF.
    """
    import queue
    import threading

    from .. import io, preprocessing

    if transforms is None:
        transforms = [preprocessing.FrameTransform()] * len(sources)
    q: queue.Queue = queue.Queue(maxsize=depth)
    DONE = object()

    def produce():
        emitted = False
        try:
            streams = [
                io.open_reader(s, workers=workers).stream_blocks(block_size=block)
                for s in sources
            ]
            while True:
                blocks = [next(s, None) for s in streams]
                # A source ran dry (None) or yielded nothing -> end of recording.
                if any(b is None or len(b) == 0 for b in blocks):
                    q.put(DONE)
                    return
                n = min(len(b) for b in blocks)  # align cameras (synced rigs match)
                window = [t.apply(b[:n]) for t, b in zip(transforms, blocks)]
                q.put(("win", window, n))
                emitted = True
                if any(len(b) < block for b in blocks):  # a short block is the last
                    q.put(DONE)
                    return
        except Exception as exc:  # noqa: BLE001
            q.put(("err", exc) if not emitted else DONE)

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = q.get()
        if item is DONE:
            return
        if item[0] == "err":
            raise item[1]
        yield item[1], item[2]


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
    detector one ``[pipeline.pose2d] batch_size``-frame block at a time and
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

    log.info(
        "streaming frames: forward batch %d, decode buffer %d batches (%d frames/source)",
        batch_size,
        depth,
        depth * batch_size,
    )

    make_progress = progress or _null_progress
    pts_parts, conf_parts, cand_xy, cand_score = [], [], [], []

    with make_progress(total, "detect 2D") as wrap:
        for window, _ in prefetch_windows(
            src_files,
            block=block,
            depth=depth,
            workers=workers,
        ):
            windows = {name: window[i] for i, name in enumerate(src_names)}
            if want_candidates:
                p, c, cand = inference.detect_candidates_sequence(
                    plan, models, windows, k=k, progress=wrap
                )
                cand_xy.append(cand.xy)
                cand_score.append(cand.score)
            else:
                p, c = inference.detect_sequence(
                    plan,
                    models,
                    windows,
                    batch_size=batch_size,
                    progress=wrap,
                )
            pts_parts.append(p)
            conf_parts.append(c)
            del window  # release this window's frames before the next is consumed

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
