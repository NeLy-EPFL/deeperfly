"""Command-line interface: ``deeperfly <subcommand>``.

Subcommands are thin wrappers over :mod:`deeperfly.pipeline`, :mod:`deeperfly.io`,
:mod:`deeperfly.video` and :mod:`deeperfly.pose2d`. Everything a run needs lives
in one merged config TOML (``deeperfly init`` writes a default to edit). The
commands:

- ``init`` -- write a default config.toml.
- ``run`` -- the pipeline's enabled stages (``pose2d`` -> ``bundle_adjustment`` ->
  ``pictorial_structures`` -> ``triangulation`` -> ``smoothing`` ->
  ``visualization``). Recordings are the positional arguments; a wildcard
  (``fly*``, ``data/*``) and several inputs fan out to every matching recording
  (non-recording matches skipped), and ``-r``/``--recursive`` runs every recording
  nested under each argument. ``-o`` is the output *directory* (default
  ``<input>/deeperfly_outputs``) collecting ``poses.h5``, the videos and a copy of
  the config. ``-c``/``--config`` is the config TOML; a run prefers the
  ``config.toml`` already in the output dir (over ``-c``, notifying), then ``-c``,
  else the packaged default. ``[pipeline].do_<stage>`` toggles which stages run.
  Detector weights download on first use.
- ``inspect`` -- print a summary of a result file.
- ``doctor`` -- print installation/runtime details: version, CPU/GPU inference,
  installed video backends, detector weights, and the default config path.

The pipeline is a linear sequence (:data:`STAGES`); each stage's input comes from
the enabled stage before it or the cached output, so disabling the finished
stages resumes a partial run.
"""

from __future__ import annotations

import argparse
import copy
import glob
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text


import numpy as np  # noqa: E402

from .cameras import CameraGroup  # noqa: E402
from .config import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    STAGE_DEFAULTS,  # noqa: F401  re-exported for tests (cli.STAGE_DEFAULTS)
    STAGES,
    Config,
)
from .io import PoseResult  # noqa: E402

#: rich output: status/results to stdout, logs and the progress bar to stderr, so
#: piping stdout stays clean and progress never clobbers a log line.
console = Console()
err_console = Console(stderr=True)

log = logging.getLogger("deeperfly")


class _FPSColumn(ProgressColumn):
    """Detection throughput in frames/second (rich ships no built-in FPS column).

    ``task.speed`` is the smoothed completed-per-second rate; since the bar ticks
    once per frame, that is frames/second. ``finished_speed`` holds the final
    average once the bar completes.
    """

    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        if not speed:
            return Text("  ?.? fps", style="progress.data.speed")
        return Text(f"{speed:5.1f} fps", style="progress.data.speed")


def _frame_progress() -> Progress:
    """A frames/second progress bar on the stderr console (detection, rendering).

    Shown only while INFO logging is on (so ``--log-level warning+`` hides it) and
    stderr is a TTY (tqdm-style); otherwise it is a no-op, so log lines and the bar
    never overwrite each other.
    """
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("frames"),
        _FPSColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        disable=not (log.isEnabledFor(logging.INFO) and err_console.is_terminal),
    )


def _configure_logging(level_name: str) -> None:
    """Configure the root log level from a ``--log-level`` name (default ``info``).

    ``info`` surfaces the per-stage messages and the progress bar; ``warning`` or
    higher hides them (the "quiet" mode). Records render through rich's
    :class:`~rich.logging.RichHandler` on the same stderr console as the bar, so
    log lines and the bar never overwrite each other.
    """
    level = getattr(logging, level_name.upper())
    handler = RichHandler(
        console=err_console,
        show_time=False,
        show_path=False,
        markup=False,  # log messages carry dict/list reprs; don't parse their brackets
        rich_tracebacks=True,
    )
    logging.basicConfig(level=level, format="%(message)s", handlers=[handler])
    log.setLevel(level)
    # JAX warns when the TPU plugin's libtpu.so is absent (the normal case). Mute
    # that noise unless we're at debug.
    if level > logging.DEBUG:
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def _load_detector(checkpoint: str | None):
    """Load the PyTorch detector from a ``.pth`` checkpoint.

    With no explicit ``checkpoint`` the cached weights are used, downloading the
    released DeepFly2D checkpoint on demand. An explicit but missing ``checkpoint``
    is an error (we never write to a user-named path).
    """
    from .pose2d import backends
    from .pose2d.download import download_torch_weights

    if checkpoint is not None and not Path(checkpoint).exists():
        raise SystemExit(
            f"no detector checkpoint at {checkpoint}. Remove [pipeline.pose2d].checkpoint "
            "to use the auto-provisioned cache, or point it at a valid .pth."
        )
    path = checkpoint or download_torch_weights()
    return backends.load_detector(path)


# -- frame-rate resolution ---------------------------------------------------


#: Frame rate used when ``[pipeline].fps`` is unset and none can be detected from
#: the recording (e.g. an image sequence carries no intrinsic rate). Matches the
#: historical default.
_FPS_FALLBACK = 100.0


def _detect_input_fps(args: argparse.Namespace, config: Config) -> float | None:
    """First detectable per-camera video frame rate, or ``None``.

    Walks the configured camera sources and returns the first video file's frame
    rate (:func:`deeperfly.video.video_fps`); image-sequence cameras have none.
    Guarded so a missing recording (a cache-only resume) yields ``None`` rather
    than raising.
    """
    from . import video

    try:
        sources = [src for _, src in _camera_sources(args, config)]
    except SystemExit:
        return None
    for src in sources:
        try:
            fps = video.video_fps(src)
        except Exception:  # noqa: BLE001
            fps = None
        if fps:
            return float(fps)
    return None


def _resolve_fps(args: argparse.Namespace, config: Config) -> float:
    """The recording's frame rate, for one_euro smoothing and the visualization base.

    Uses ``[pipeline].fps`` when set; otherwise detects it from the input videos
    (:func:`_detect_input_fps`). Falls back to :data:`_FPS_FALLBACK` when neither
    is available -- e.g. an image sequence, or a cache-only resume with no
    recording -- logging a hint to set ``[pipeline].fps`` explicitly.
    """
    if config.fps is not None:
        return config.fps
    detected = _detect_input_fps(args, config)
    if detected is not None:
        log.info("detected input fps %.4g from the recording", detected)
        return detected
    log.warning(
        "could not detect the input fps (image sequence, or no recording available); "
        "using %g fps -- set [pipeline].fps to override",
        _FPS_FALLBACK,
    )
    return _FPS_FALLBACK


# -- input -> camera frame resolution ----------------------------------------


def _footage_exts() -> tuple[str, ...]:
    """Footage extensions deeperfly can read, in priority order (video before image).

    Recognizes a camera's frames and, when a folder mixes several, picks the one to
    keep (earliest wins). Imported lazily so resolving filenames doesn't pull in the
    video stack.
    """
    from .video.io import _IMAGE_EXTS, _VIDEO_EXTS

    return _VIDEO_EXTS + _IMAGE_EXTS


def _is_video_ext(suffix: str) -> bool:
    """Whether a file ``suffix`` (e.g. ``.mp4``) is a video extension -- as opposed to
    an image one. Video footage is a single file per camera; images form a sequence."""
    from .video.io import _VIDEO_EXTS

    return suffix.lower() in _VIDEO_EXTS


def _camera_glob(pattern: str) -> str:
    """A camera's ``input`` value as a filename glob.

    A value that already names a file (a known footage suffix like
    ``camera_0.mp4``) or carries its own wildcard (``camera_0/*``, ``cam*``) is used
    verbatim; a bare name (``camera_0``) is treated as a *prefix*, so ``camera_0``
    becomes ``camera_0*`` and matches both ``camera_0.mp4`` and an image sequence
    ``camera_0_000123.jpg ...``.
    """
    has_wildcard = any(c in pattern for c in "*?[")
    if has_wildcard or Path(pattern).suffix.lower() in _footage_exts():
        return pattern
    return f"{pattern}*"


def _camera_files(root: Path, pattern: str) -> list[Path]:
    """A camera's footage files under ``root`` matching its ``input`` ``pattern``.

    Globs ``pattern`` (see :func:`_camera_glob`), keeps files with a known footage
    extension, and -- when several extensions match -- keeps the highest-priority
    one. Naturally sorted. Video footage is a single file, so several matching
    videos keep only the first (warned); images stay as the whole sequence. Empty
    when nothing footage-like matches, so the caller can treat the camera as absent.
    """
    from natsort import natsorted

    exts = _footage_exts()
    files = [
        p
        for p in root.glob(_camera_glob(pattern))
        if p.is_file() and p.suffix.lower() in exts
    ]
    present = {p.suffix.lower() for p in files}
    if len(present) > 1:
        keep = min(present, key=exts.index)
        files = [p for p in files if p.suffix.lower() == keep]
    return _first_if_video(root, pattern, natsorted(files))


def _first_if_video(root: Path, name: str, files: list[Path]) -> list[Path]:
    """Reduce a camera's video footage to its first file (warning), leaving an image
    sequence untouched -- a camera's video is one file, but images are a sequence."""
    if len(files) > 1 and _is_video_ext(files[0].suffix):
        log.warning(
            "recording %s: camera %s matches %d video files %s; using only the first "
            "(%s) -- video footage is a single file per camera",
            root,
            name,
            len(files),
            [p.name for p in files],
            files[0].name,
        )
        return files[:1]
    return files


def _camera_patterns(config: Config | dict) -> dict[str, str]:
    """``camera-name -> footage glob`` (the per-camera ``input`` key), in config order.

    A camera with no ``input`` entry defaults to its own name as the pattern. Accepts
    a parsed dict too (the recording-discovery configs in the tests do this).
    """
    return Config.coerce(config).camera_patterns()


def _camera_sources(
    args: argparse.Namespace, config: Config
) -> list[tuple[str, list[Path]]]:
    """``(name, footage-files)`` per camera (in ``[cameras]`` order).

    Prefers the files ``deeperfly run`` already resolved (``args.sources``) so
    footage is globbed once per run; otherwise resolves each camera from
    ``args.input`` with the per-camera ``input`` globs (a library caller). With
    neither, every camera resolves to an empty list. Each source is the list passed
    to :func:`deeperfly.video.read_frames`.
    """
    patterns = config.camera_patterns()
    pre = getattr(args, "sources", None)
    if pre and all(name in pre for name in patterns):
        return [(name, pre[name]) for name in patterns]
    root = getattr(args, "input", None)
    if root is None:
        return [(name, []) for name in patterns]
    return [(name, _camera_files(Path(root), pat)) for name, pat in patterns.items()]


def _camera_image_sizes(args, config: Config) -> dict[str, tuple[int, int]]:
    """``name -> (height, width)`` from a single frame per camera.

    Used to infer each view's principal point. Reads only frame 0 (host), so it is
    cheap and independent of the full streaming decode.
    """
    from . import video

    backend = config.io.video_reader
    image_backend = config.io.image_reader
    # Size the principal point on the *transformed* frame -- the detector and the
    # overlays use the preprocess-transformed footage, so a rot90 that swaps
    # H/W must swap here too.
    transforms = config.frame_transforms()
    sizes: dict[str, tuple[int, int]] = {}
    for name, src in _camera_sources(args, config):
        head = video.read_frames(
            src, backend=backend, image_backend=image_backend, indices=[0]
        )
        head = transforms.get(name, video.FrameTransform()).apply(head)
        sizes[name] = tuple(int(d) for d in head.shape[1:3])
    return sizes


def _prefetch_windows(
    sources,
    *,
    backend,
    block,
    transforms=None,
    depth=1,
    image_backend="auto",
    workers=None,
):
    """Yield ``(window, n)`` multi-camera frame blocks from continuous decode.

    A background producer opens **one continuous forward decoder per source**
    (:func:`deeperfly.video.stream_frames`) and walks them all together, grouping
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
    :class:`~deeperfly.video.FrameTransform` (aligned to ``sources``) applied to
    each block before yielding, so detection sees the ``[cameras.*.preprocess]``
    orientation.

    The producer treats each source as an opaque forward stream -- it never asks
    for a frame count or a seek -- so an unbounded *live-camera* source (a future
    :func:`deeperfly.video.stream_frames` branch) drives this loop unchanged.

    EOF: the first source to run out (a short or exhausted block) ends the stream;
    a read failure before any window is emitted propagates, later ones are EOF.
    """
    import queue
    import threading

    from . import video

    if transforms is None:
        transforms = [video.FrameTransform()] * len(sources)
    q: queue.Queue = queue.Queue(maxsize=depth)
    DONE = object()

    def produce():
        emitted = False
        try:
            streams = [
                video.stream_frames(
                    s,
                    backend=backend,
                    image_backend=image_backend,
                    workers=workers,
                    block=block,
                )
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


def _detect_2d(args, config: Config, model, sides, flips, *, want_candidates, k):
    """Stream 2D detection over decode blocks -> ``(pts2d, conf, candidates)``.

    Decodes each camera in one continuous forward pass (CPU), handing the detector
    one ``[pipeline.pose2d] batch_size``-frame block at a time and freeing it before
    the next, so peak frame memory is bounded by the decode buffer, not the recording
    length. Per-block results are concatenated along time. End-of-file comes from
    the decoder (a short or exhausted block), so it doesn't depend on
    :func:`deeperfly.video.count_frames` being exact -- that is only the
    progress-bar total.
    """
    from . import video
    from .pictorial import Candidates
    from .pose2d import inference

    pose2d = config.pose2d
    backend = config.io.video_reader
    image_backend = config.io.image_reader
    workers = config.io.image_workers
    # Two knobs: the GPU forward batch (images/forward), and the decode buffer in
    # multiples of it. A block holds one batch of frames; the reader keeps up to
    # `depth` of them queued (>= 1 so the queue stays bounded -- 0 is unbounded).
    batch_size = pose2d.batch_size
    depth = pose2d.decode_buffer
    block = batch_size
    cam_sources = _camera_sources(args, config)
    sources = [src for _, src in cam_sources]
    # Apply each camera's preprocess transform to its decoded block, so the detector
    # sees the corrected orientation (and 2D points land in that frame).
    transforms_by_name = config.frame_transforms()
    transforms = [
        transforms_by_name.get(name, video.FrameTransform()) for name, _ in cam_sources
    ]
    total = video.count_frames(sources[0]) if sources else 0

    # One-line summary instead of a per-block read log (that's at -vv now).
    # reader_name mirrors read_frames's dispatch off the actual source, so the
    # reported decoder matches what really runs; guard a forced-but-uninstalled one.
    try:
        reader = video.reader_name(
            sources[0], backend=backend, image_backend=image_backend
        )
    except Exception:  # noqa: BLE001
        reader = backend
    log.info(
        "streaming frames via '%s' backend: forward batch %d, decode buffer %d "
        "batches (%d frames/camera)",
        reader,
        batch_size,
        depth,
        depth * batch_size,
    )

    pts_parts, conf_parts, cand_xy, cand_score = [], [], [], []
    bar = _frame_progress()

    with bar:
        task = bar.add_task("detect 2D", total=total)

        def progress(rng):  # advance the single bar once per completed frame
            for t in rng:
                yield t
                bar.advance(task)

        for window, _ in _prefetch_windows(
            sources,
            backend=backend,
            block=block,
            transforms=transforms,
            depth=depth,
            image_backend=image_backend,
            workers=workers,
        ):
            if want_candidates:
                p, c, cand = inference.detect_candidates_sequence(
                    model, window, sides, flips, k=k, progress=progress
                )
                cand_xy.append(cand.xy)
                cand_score.append(cand.score)
            else:
                p, c = inference.detect_sequence(
                    model,
                    window,
                    sides,
                    flips,
                    batch_size=batch_size,
                    progress=progress,
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


# -- stage resolution --------------------------------------------------------


def _default_outdir(inp: str | Path) -> Path:
    """Default output dir when ``-o`` is omitted: ``<input>/deeperfly_outputs``.

    ``<input>`` is the recording directory; for a glob/file input the sibling
    ``deeperfly_outputs`` next to it is used.
    """
    p = Path(inp)
    base = p if p.is_dir() else p.parent
    return base / "deeperfly_outputs"


#: Sentinel injected by :func:`_normalize_overwrite_argv` for a bare ``--overwrite``
#: (no stage names), meaning "recompute every stage".
_OVERWRITE_ALL = "__all__"


def _overwrite_stages(overwrite: list[str] | None) -> set[str]:
    """Stage names selected by ``--overwrite`` (empty set = reuse all cached).

    ``None`` / empty -> reuse everything (the default). The ``_OVERWRITE_ALL``
    sentinel (a bare ``--overwrite``) selects every stage. Otherwise the listed
    stage names are validated against :data:`STAGES`.
    """
    if not overwrite:
        return set()
    if _OVERWRITE_ALL in overwrite:
        return set(STAGES)
    unknown = [s for s in overwrite if s not in STAGES]
    if unknown:
        raise SystemExit(
            f"--overwrite got unknown stage(s) {', '.join(unknown)}; choose from "
            f"{', '.join(STAGES)} (or a bare --overwrite to recompute everything)"
        )
    return set(overwrite)


def _visualization_cached(config: Config, outdir: Path) -> bool:
    """Whether every ``[[pipeline.visualization.videos]]`` MP4 already exists in ``outdir``."""
    specs = config.videos
    return bool(specs) and all(
        (outdir / f"{spec.video_name}.mp4").exists() for spec in specs
    )


def _stage_cached(
    stage: str, cached: PoseResult | None, config: Config, outdir: Path
) -> bool:
    """Whether ``stage``'s output already exists in the output dir (so it can be reused).

    The pose stages read the cached ``poses.h5`` loaded as ``cached`` (its arrays /
    ``meta`` markers); ``visualization`` checks for the rendered MP4s. Used to skip
    recomputation by default -- ``--overwrite`` forces a stage to recompute even
    when this returns ``True``.
    """
    if stage == "visualization":
        return _visualization_cached(config, outdir)
    if cached is None:
        return False
    if stage == "pose2d":
        return cached.pts2d is not None
    if stage == "bundle_adjustment":
        return bool(cached.meta.get("bundle_adjustment"))
    if stage == "pictorial_structures":
        return bool(cached.meta.get("pictorial"))
    if stage == "triangulation":
        return cached.pts3d is not None
    if stage == "smoothing":
        return cached.pts3d_smoothed is not None
    return False


# -- pipeline stages ---------------------------------------------------------


def _stage_pose2d(
    args: argparse.Namespace, config: Config, *, want_candidates: bool
) -> tuple[PoseResult, object | None, None, dict]:
    """Run 2D detection -> a 2D-only :class:`PoseResult` + in-memory artifacts.

    Returns ``(result, candidates, None, image_sizes)``. ``candidates`` is the
    top-K peak set, extracted only when ``want_candidates`` (the
    ``pictorial_structures`` stage, since candidates aren't cached). Frames are not
    held in memory (detection streams them in windows -- see :func:`_detect_2d`),
    so the third slot is ``None``; a visualization stage re-sources the overlay
    cameras it needs.
    """
    from .pose2d import backends, inference
    from .triangulate import apply_visibility

    transforms = config.frame_transforms()
    image_sizes = _camera_image_sizes(args, config)
    log.info(
        "input image sizes (h, w, after preprocess): %s",
        {n: tuple(s) for n, s in image_sizes.items()},
    )
    cameras = config.camera_group(image_sizes=image_sizes)
    unknown = set(transforms) - set(cameras.names)
    if unknown:
        log.warning(
            "preprocess entries for unknown cameras are ignored: %s",
            sorted(unknown),
        )
    active = {
        n: t
        for n, t in transforms.items()
        if n in cameras.names and not t.is_identity()
    }
    if active:
        log.info(
            "frame preprocessing (per camera): %s",
            {
                n: f"fliplr={t.fliplr}, flipud={t.flipud}, rot90={t.rot90}"
                for n, t in active.items()
            },
        )
    skeleton = config.skeleton()

    pose2d = config.pose2d
    log.info("loading detector (checkpoint: %s)", pose2d.checkpoint or "cached")
    model = _load_detector(pose2d.checkpoint)
    precision = pose2d.precision
    backends.set_precision(
        model, precision
    )  # float16 -> CUDA autocast (no-op on CPU/MPS)
    log.info(
        "detector ready on device %s (precision: %s)",
        backends.detector_device(model),
        precision,
    )

    k = config.pictorial.k
    sides, flips = inference.fly_camera_layout(cameras.names)
    n_passes = len(inference.expand_passes(sides, flips)[0])
    batch_size = pose2d.batch_size
    log.info(
        "detecting 2D poses: %d views, %d forward passes/frame, network input %dx%d, "
        "forward batch %d frames",
        len(image_sizes),
        n_passes,
        inference.IMG_SIZE[0],
        inference.IMG_SIZE[1],
        batch_size,
    )
    pts2d, conf, candidates = _detect_2d(
        args,
        config,
        model,
        sides,
        flips,
        want_candidates=want_candidates,
        k=k,
    )
    # Mask (camera, point) pairs the rig cannot see once, here, so the cached 2D
    # and every downstream stage see the same visibility-masked points.
    pts2d = apply_visibility(pts2d, skeleton, cameras.names)

    result = PoseResult(cameras=cameras, skeleton=skeleton, pts2d=pts2d, conf=conf)
    return result, candidates, None, image_sizes


def _config_camera_rig(args: argparse.Namespace, config: Config) -> CameraGroup:
    """The un-refined camera rig straight from the config -- the BA stage's *input*.

    On a resume that reuses the cached 2D, ``result.cameras`` are the *previous* BA
    output, so re-running bundle adjustment (e.g. after editing
    ``[pipeline.bundle_adjustment]`` or the ``[cameras]`` rig) must restart from
    this fresh config rig -- otherwise it just re-refines already-refined cameras
    and the edits look ignored. Reads one frame per camera for the image sizes, as
    ``pose2d`` does, so principal-point inference matches.
    """
    image_sizes = _camera_image_sizes(args, config)
    return config.camera_group(image_sizes=image_sizes)


def _stage_bundle_adjustment(config: Config, result: PoseResult) -> PoseResult:
    """Refine ``result.cameras`` with bundle adjustment (fly-as-calibration-target).

    Calibrates on the arg-max 2D in ``result`` and replaces its cameras in place.
    The caller hands in the un-refined config rig (see :func:`_config_camera_rig`),
    so editing the rig or ``[pipeline.bundle_adjustment]`` and recomputing this
    stage recalibrates from the edited config rather than the prior BA output.
    """
    from .pipeline import calibrate
    from .triangulate import reprojection_error, triangulate

    log.info(
        "bundle adjustment: refining cameras (%d frames, %d views)",
        result.n_frames,
        result.n_views,
    )
    ba = config.bundle_adjustment
    result.cameras, _ = calibrate(
        result.cameras,
        result.pts2d,
        result.conf,
        result.skeleton,
        ba_keypoints=ba.keypoints,
        fixed=ba.fixed,
        shared=ba.shared,
        **ba.least_squares,
    )
    result.meta["bundle_adjustment"] = True  # marks the cameras as BA-refined (cache)

    # Report the refined rig's pixel reprojection error (triangulate the committed
    # 2D with the new cameras and reproject); the triangulation stage refines it.
    err = reprojection_error(
        result.cameras, triangulate(result.cameras, result.pts2d), result.pts2d
    )
    log.info(
        "bundle adjustment: reprojection error median %.3f px  max %.3f px",
        np.nanmedian(err),
        np.nanmax(err),
    )
    return result


def _stage_pictorial_structures(
    config: Config, result: PoseResult, candidates: object
) -> PoseResult:
    """DeepFly3D pictorial-structures recovery over the detector's top-K candidates.

    Commits a corrected per-view 2D (``result.pts2d``) and an initial 3D estimate
    (``result.pts3d``); the triangulation stage, if enabled, then re-triangulates
    the committed 2D. ``candidates`` must come from a ``pose2d`` run in this process
    (they are not cached).
    """
    from . import pictorial

    ps = config.pictorial
    log.info(
        "pictorial structures: recovering peaks (%d frames, %d views)",
        result.n_frames,
        result.n_views,
    )
    pts3d, pts2d, reproj = pictorial.reconstruct(
        result.cameras,
        result.skeleton,
        candidates,
        result.pts2d,
        temporal=ps.temporal,
        lam=ps.lam,
    )
    result.pts2d, result.pts3d, result.reproj_error = pts2d, pts3d, reproj
    result.meta["pictorial"] = True
    return result


def _stage_triangulation(config: Config, result: PoseResult) -> PoseResult:
    """Triangulate the 2D in ``result`` to 3D by the configured method.

    Sets ``result.pts3d`` (and the cleaned ``result.pts2d`` / ``reproj_error`` for
    the outlier-rejecting methods) using ``result.cameras``. ``ransac`` builds each
    point from its largest multi-view consensus, ``greedy`` drops the
    worst-reprojecting view, ``dlt`` is plain least squares (see
    :func:`deeperfly.pipeline._resolve_triangulation`).
    """
    from .pipeline import _resolve_triangulation, reconstruct, reconstruct_ransac
    from .triangulate import reprojection_error, triangulate

    opts = config.triangulation
    method = _resolve_triangulation(opts.method)
    log.info(
        "triangulation: method=%s (%d frames, %d views)",
        method,
        result.n_frames,
        result.n_views,
    )
    if method == "ransac":
        pts3d, pts2d, reproj = reconstruct_ransac(
            result.cameras,
            result.pts2d,
            threshold=opts.ransac_threshold,
            min_inliers=opts.min_inliers,
        )
    elif method == "greedy":
        pts3d, pts2d, reproj = reconstruct(
            result.cameras,
            result.pts2d,
            reproj_threshold=opts.reproj_threshold,
            max_drops=opts.max_drops,
        )
    else:  # "dlt": plain least-squares triangulation, no outlier handling
        pts2d = result.pts2d
        pts3d = triangulate(result.cameras, pts2d)
        reproj = reprojection_error(result.cameras, pts3d, pts2d)
    result.pts2d, result.pts3d, result.reproj_error = pts2d, pts3d, reproj
    result.meta["triangulation"] = method
    return result


def _stage_smoothing(
    args: argparse.Namespace, config: Config, result: PoseResult
) -> PoseResult:
    """Temporal smoothing of ``result.pts3d`` -> ``result.pts3d_smoothed``."""
    from .correction import smooth_gaussian, smooth_one_euro

    smoothing = config.smoothing
    method, kwargs = smoothing.method, smoothing.kwargs
    fps = _resolve_fps(args, config)
    log.info("smoothing: method=%s", method)
    if method == "gaussian":
        result.pts3d_smoothed = smooth_gaussian(result.pts3d, **kwargs)
    elif method == "one_euro":
        result.pts3d_smoothed = smooth_one_euro(result.pts3d, fps, **kwargs)
    else:
        raise SystemExit(
            f"unknown [pipeline.smoothing].method {method!r} (gaussian|one_euro)"
        )
    result.meta["fps"] = fps
    return result


def _source_view_frames(
    args: argparse.Namespace,
    config: Config,
    result: PoseResult,
    views: list[str],
    in_memory: list | None = None,
) -> dict[str, np.ndarray]:
    """Per-view footage for the visualization stage's ``imshow`` panels.

    Uses ``in_memory`` frames (indexed by camera order) when available; otherwise
    the footage ``deeperfly run`` resolved up front (``args.sources``). A resume
    that re-renders just re-passes the recording, re-resolving the footage the same
    way. Errors if neither is available.
    """
    from . import video

    if not views:
        return {}
    names = result.cameras.names
    # The detector ran on transformed frames, so the 2D/3D overlays live in
    # transformed-frame coordinates; the overlay footage must match (apply the
    # same per-camera preprocess transform).
    transforms = config.frame_transforms()
    backend = config.io.video_reader
    image_backend = config.io.image_reader
    workers = config.io.image_workers

    def transform(v, frames):
        return transforms.get(v, video.FrameTransform()).apply(frames)

    if in_memory is not None:
        return {v: transform(v, in_memory[names.index(v)]) for v in views}

    sources = getattr(args, "sources", None) or {}
    if all(sources.get(v) for v in views):
        return {
            v: transform(
                v,
                video.read_frames(
                    sources[v],
                    backend=backend,
                    image_backend=image_backend,
                    workers=workers,
                ),
            )
            for v in views
        }
    raise SystemExit(
        "image (imshow) panels need the original frames, but none are in memory and "
        "the run resolved no footage. Re-run with the recording as the input "
        "('deeperfly run <recording> --overwrite visualization'), or drop the imshow "
        "panels from [[pipeline.visualization.videos]]."
    )


def _stage_visualization(
    args: argparse.Namespace,
    config: Config,
    result: PoseResult,
    frames: list | None,
    outdir: Path,
) -> None:
    """Render every ``[[pipeline.visualization.videos]]`` to ``<outdir>/<name>.mp4``.

    Each video is composited by :mod:`deeperfly.viz.compose` from its panels (see
    the config's ``[pipeline.visualization]`` section), overwriting any existing MP4
    (the visualization stage recomputes when enabled; disable it to keep prior
    renders). A video whose panels reproject the 3D skeleton is skipped with a
    reason when the result has no 3D pose (e.g. no triangulation/pictorial stage
    ran); frames for ``imshow`` panels are sourced only across the videos that
    actually render.
    """
    from . import video
    from .viz import compose

    specs = config.videos
    if not specs:
        log.info(
            "no [[pipeline.visualization.videos]] in the config; nothing to render"
        )
        return

    pending = []
    for spec in specs:
        if result.pts3d is None and any(p.plot == "skeleton_3d" for p in spec.panels):
            log.warning(
                "skipping video %r: it reprojects the 3D skeleton but the result has "
                "no 3D pose (enable [pipeline].do_triangulation or do_pictorial_structures)",
                spec.video_name,
            )
        else:
            pending.append(spec)
    if not pending:
        return

    input_fps = _resolve_fps(args, config)
    # The visualization stage *writes* MP4s -- the output encoder is [io.video].writer
    # (the shared I/O config), distinct from [io.video].reader used to read footage.
    backend = config.io.video_writer
    views = sorted(
        {p.view for spec in pending for p in spec.panels if p.plot == "imshow"}
    )
    src = compose.Sources(
        skeleton=result.skeleton,
        camera_group=result.cameras,
        frames=_source_view_frames(args, config, result, views, in_memory=frames),
        pts2d=result.pts2d,
        pts3d=result.pts3d,
        conf=result.conf,
    )
    with _frame_progress() as bar:
        for spec in pending:
            path = outdir / f"{spec.video_name}.mp4"
            fps = spec.resolve_fps(input_fps)
            log.info("rendering %s -> %s @ %g fps", spec.video_name, path, fps)
            task = bar.add_task(f"render {spec.video_name}", total=src.n_frames())

            def progress(rng, _task=task):  # advance the bar once per composited frame
                for t in rng:
                    yield t
                    bar.advance(_task)

            clip = compose.render_video(spec, src, progress=progress)
            video.write_mp4(clip, path, fps=fps, backend=backend)
            log.info("wrote %s", path)


# -- subcommands -------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    dst = Path(args.output)
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} already exists (pass --force to overwrite)")
    dst.write_text(DEFAULT_CONFIG_PATH.read_text())
    console.print(f"[green]wrote[/green] {dst}")
    # markup=False: the message shows literal [cameras] config sections, which rich
    # would otherwise try to parse as style tags.
    console.print(
        "next: edit [cameras] to match your rig, then "
        f"'deeperfly run <recording> -c {dst}' "
        "(outputs land in <recording>/deeperfly_outputs/; override with -o <dir>)",
        markup=False,
        highlight=False,
    )


def _has_glob(pattern: str) -> bool:
    """Whether ``pattern`` carries a shell wildcard (so it should be expanded)."""
    return any(c in pattern for c in "*?[")


@dataclass(frozen=True)
class Recording:
    """One unit of work: a camera -> footage-files map and where its results go.

    ``sources`` maps a camera name to its naturally-sorted footage files (a single
    video, or an image sequence), already reconciled to one extension and validated
    to share a file and frame count with the other cameras. Empty only for a
    directory kept so a resume can reuse a cached result though its footage is
    absent (see :func:`_resolve_recordings`).

    ``outdir`` is this recording's output directory (see :func:`_run_outdir`) --
    the run's durable identity, holding the config snapshot and cached ``poses.h5``.
    The input directory is not retained; a resume re-passes the recording, which
    re-resolves ``sources`` the same way.
    """

    sources: dict[str, list[Path]]
    outdir: Path


def _frame_counts_match(root: Path, sources: dict[str, list[Path]]) -> bool:
    """Whether every camera under ``root`` has the same file and frame count.

    File counts are compared directly. For image sequences equal file counts
    already imply equal frame counts, so only *video* footage is probed for its
    frame count, and only when knowable (an unreadable file's ``None`` count is
    skipped rather than falsely rejecting the recording). Warns when counts differ.
    """
    file_counts = {n: len(ps) for n, ps in sources.items()}
    if len(set(file_counts.values())) > 1:
        log.warning(
            "recording %s has an uneven file count across cameras %s; skipping it",
            root,
            file_counts,
        )
        return False
    sample = next((ps for ps in sources.values() if ps), [])
    if not sample or not _is_video_ext(sample[0].suffix):
        return True  # image sequence (or empty): the file count already settled it
    from . import video

    frame_counts = {n: video.count_frames(ps) for n, ps in sources.items()}
    known = {c for c in frame_counts.values() if c is not None}
    if len(known) > 1:
        log.warning(
            "recording %s has an uneven frame count across cameras %s; skipping it",
            root,
            frame_counts,
        )
        return False
    return True


def _find_recording(root: Path, config: Config) -> dict[str, list[Path]] | None:
    """``root``'s ``camera -> footage-files`` map if it is a recording, else ``None``.

    A *recording* is a directory holding footage for every configured camera (its
    ``input`` glob); the footage is a single video file or an image sequence. A
    directory matching *no* camera is silently not a recording (an intermediate or
    output dir); the rest warn and skip:

    - footage for only some cameras (a malformed recording, or a wrong ``input``);
    - files matched but none with a known footage extension;
    - several footage extensions in one folder (the highest-priority one is kept,
      and any camera then left with nothing counts as missing);
    - an unequal file or frame count across cameras (see :func:`_frame_counts_match`).
    """
    if not root.is_dir():
        return None
    from natsort import natsorted

    exts = _footage_exts()
    patterns = _camera_patterns(config)
    # Raw matches (any file) per camera, so "no match" is distinguishable from
    # "matched, but not footage".
    raw = {
        name: [p for p in root.glob(_camera_glob(pat)) if p.is_file()]
        for name, pat in patterns.items()
    }
    present = {name: ps for name, ps in raw.items() if ps}
    if not present:
        return None  # nothing here looks like a camera's files: not a recording
    missing = [name for name in patterns if name not in present]
    if missing:
        log.warning(
            "recording %s has footage for only %s (missing %s); skipping it",
            root,
            sorted(present),
            missing,
        )
        return None
    sources = {
        name: [p for p in ps if p.suffix.lower() in exts]
        for name, ps in present.items()
    }
    no_ext = sorted(name for name, ps in sources.items() if not ps)
    if no_ext:
        log.warning(
            "recording %s: camera(s) %s matched files but none with a known footage "
            "extension %s; skipping it",
            root,
            no_ext,
            list(exts),
        )
        return None
    seen = {p.suffix.lower() for ps in sources.values() for p in ps}
    if len(seen) > 1:
        keep = min(seen, key=exts.index)
        log.warning(
            "recording %s mixes footage extensions %s; using %s",
            root,
            sorted(seen, key=exts.index),
            keep,
        )
        sources = {
            name: [p for p in ps if p.suffix.lower() == keep]
            for name, ps in sources.items()
        }
        gone = sorted(name for name, ps in sources.items() if not ps)
        if gone:
            log.warning(
                "recording %s has no %s footage for %s; skipping it", root, keep, gone
            )
            return None
    sources = {
        name: _first_if_video(root, name, natsorted(ps)) for name, ps in sources.items()
    }
    if not _frame_counts_match(root, sources):
        return None
    return sources


def _expand_pattern(pattern: str) -> tuple[list[Path], bool]:
    """One ``run`` input argument -> ``(paths, is_glob)``.

    A wildcard (``fly*``, ``data/*``) expands to its sorted matches (possibly
    empty); a literal argument yields just itself. ``is_glob`` flags which it was,
    so a wildcard's incidental non-recording matches are skipped silently while a
    literal path the user typed is reported when invalid.
    """
    if _has_glob(pattern):
        return [Path(p) for p in sorted(glob.glob(pattern))], True
    return [Path(pattern)], False


def _dedup_found(
    found: Iterable[tuple[Path, dict[str, list[Path]]]],
) -> list[tuple[Path, dict[str, list[Path]]]]:
    """Drop ``(dir, sources)`` pairs whose directory repeats (overlapping inputs/roots
    can match one twice), keeping the first occurrence so run order stays predictable."""
    seen: set = set()
    out: list[tuple[Path, dict[str, list[Path]]]] = []
    for d, src in found:
        key = d.resolve()
        if key not in seen:
            seen.add(key)
            out.append((d, src))
    return out


def _run_outdir(output: str | None, recording: Path, *, batch: bool) -> Path:
    """Output directory for one recording.

    Default (no ``-o``): the recording's own ``deeperfly_outputs``. With ``-o``:
    that directory for a single recording, or a per-recording subdirectory under it
    for a wildcard/recursive batch (so the runs don't overwrite each other).
    """
    if not output:
        return _default_outdir(recording)
    base = Path(output)
    return base / recording.name if batch else base


def _plan_recordings(
    found: list[tuple[Path, dict[str, list[Path]]]], output: str | None
) -> list[Recording]:
    """Turn discovered ``(dir, sources)`` pairs into :class:`Recording`\\ s.

    The output directory is resolved per recording (:func:`_run_outdir`); whether
    this is a *batch* (several recordings, so ``-o`` nests per recording) is known
    only here, once every input has been resolved.
    """
    batch = len(found) > 1
    return [Recording(src, _run_outdir(output, d, batch=batch)) for d, src in found]


def _resolve_recordings(
    inputs: list[Path], *, recursive: bool, config: Config, output: str | None = None
) -> list[Recording]:
    """Expand the ``run`` inputs into the recordings to process (footage + output dir).

    ``inputs`` is one or more input arguments, each a literal path or a wildcard
    pattern expanded against the filesystem (:func:`_expand_pattern`). A *recording*
    is a directory holding footage for every configured camera, resolved to a
    ``camera -> files`` map by :func:`_find_recording` (which warns and skips a
    malformed one). Each kept recording is paired with its output directory
    (:func:`_run_outdir`, honoring ``output`` = ``-o``); the input directory is not
    retained past this point. The behaviors:

    - A single literal path is taken as that one recording -- kept (with empty
      sources) even when it is not valid footage, so a resume from its cached result
      still works -- with a warning naming it when it is not a valid recording.
    - Several inputs and/or a wildcard run as a batch: only the valid recordings are
      kept (a wildcard's incidental non-recording matches are dropped silently);
      nothing valid is a warned error.
    - With ``--recursive`` each input is a *parent* directory whose subtree is walked
      for recordings; an empty result is an error.

    De-duplicated by directory (overlapping inputs) keeping first-seen order.
    """
    candidates: list[tuple[Path, bool]] = []
    for arg in inputs:
        paths, is_glob = _expand_pattern(str(arg))
        if is_glob and not paths:
            log.warning("input pattern %r matched no paths", str(arg))
        candidates += [(p, is_glob) for p in paths]

    if recursive:
        found: list[tuple[Path, dict[str, list[Path]]]] = []
        for root, is_glob in candidates:
            if not root.is_dir():
                if not is_glob:  # a literal parent the user named but that is absent
                    log.warning(
                        "%s is not a directory -- --recursive searches a parent "
                        "directory for recordings; skipping",
                        root.resolve(),
                    )
                continue
            for d in [root, *sorted(root.rglob("*"))]:
                if d.is_dir() and (src := _find_recording(d, config)) is not None:
                    found.append((d, src))
        found = _dedup_found(found)
        if not found:
            log.warning(
                "no recordings found under %s (searched recursively); a recording is "
                "a directory holding footage for every configured camera",
                [str(p) for p, _ in candidates] or [str(a) for a in inputs],
            )
            raise SystemExit("no recordings to run")
        return _plan_recordings(found, output)

    # Non-recursive. A single explicit path is honored as-is (resume-friendly): keep
    # it even when it is not valid footage, so resuming from its cache still works.
    if len(candidates) == 1 and not candidates[0][1]:
        path = candidates[0][0]
        src = _find_recording(path, config)
        if src is None:
            log.warning(
                "%s is not a valid recording directory -- it does not hold footage "
                "for every configured camera (it can still resume from a cached "
                "result in its output dir)",
                path.resolve(),
            )
            src = {}
        return _plan_recordings([(path, src)], output)

    # Several inputs and/or a wildcard: a batch. Keep only the valid recordings; only
    # warn (and error) when the inputs yield no valid recording at all.
    found = _dedup_found(
        (p, src)
        for p, _ in candidates
        if (src := _find_recording(p, config)) is not None
    )
    if not found:
        log.warning(
            "none of the inputs is a valid recording directory (a directory holding "
            "footage for every configured camera)",
        )
        raise SystemExit("no valid recording directories among the inputs")
    return _plan_recordings(found, output)


def _has_2d(result: PoseResult | None) -> bool:
    return result is not None and result.pts2d is not None


def _require_input_footage(args: argparse.Namespace, config: Config) -> None:
    """Fail (before any output dir is created) if the run's recording is unreadable.

    Checked only when ``pose2d`` will actually decode frames; a resume that reuses
    a cached 2D pose needs no footage. The footage was resolved up front by
    :func:`_resolve_recordings` (``args.sources``); a library caller that set only
    ``args.input`` is validated directly. Raising here keeps a fresh run that can't
    read its input from leaving an empty ``deeperfly_outputs`` behind.
    """
    patterns = _camera_patterns(config)
    inp = getattr(args, "input", None)
    if getattr(args, "sources", None) is None and inp is not None:
        root = Path(inp)
        if not root.exists():
            raise SystemExit(
                f"input recording {root} does not exist -- pass an existing directory "
                "holding the per-camera video/images for this run"
            )
        if not root.is_dir():
            raise SystemExit(
                f"input recording {root} is not a directory -- the run input is a "
                "directory of per-camera footage, not a single file"
            )
        for name, pat in patterns.items():
            if not _camera_files(root, pat):
                raise SystemExit(f"no video or images for camera {name!r} under {root}")
        return

    sources = getattr(args, "sources", None) or {}
    missing = [name for name in patterns if not sources.get(name)]
    if missing:
        raise SystemExit(
            f"this run needs footage for pose2d but the recording resolved no files "
            f"for camera(s) {missing} (see the warning above) -- pass a recording that "
            "holds video/images for every camera, or resume from a cached poses.h5"
        )


def _run_one(args: argparse.Namespace, outdir: Path) -> None:
    """Run the config's enabled stages for a single recording.

    The config is resolved against ``outdir`` (see :meth:`Config.read_for_run`) and
    its ``[pipeline].do_<stage>`` toggles decide which stages run
    (:meth:`Config.stage_flags`). An enabled stage reuses its result if it's already
    in the output dir,
    recomputing only when missing or ``--overwrite`` selects it; recomputing a stage
    cascades to every enabled stage downstream (their inputs changed).

    Each stage runs only if its input is available -- footage for ``pose2d``, a 2D
    pose for ``bundle_adjustment`` / ``triangulation``, candidates for
    ``pictorial_structures``, a 3D pose for ``smoothing``, a result for
    ``visualization`` -- from an upstream stage or the cached ``poses.h5``; a stage
    whose input is missing is skipped with the reason logged. A disabled stage never
    runs but its cached output still feeds downstream.
    """
    config = Config.read_for_run(args.config, outdir)
    stages = config.stage_flags()  # config validated at construction
    overwrite = _overwrite_stages(getattr(args, "overwrite", None))

    # `cached` is the result already in the output dir; `result` starts there, a
    # reused stage keeps it, a recomputed stage replaces it. The first recompute
    # flips `recomputed` so every later enabled stage recomputes too (cascade).
    h5_path = outdir / "poses.h5"
    cached = PoseResult.load(h5_path) if h5_path.exists() else None

    # Validate the footage *before* creating the output dir, so a fresh run that
    # can't read its input fails cleanly instead of leaving an empty dir behind.
    # Only pose2d decodes the recording, and only when it recomputes; a resume
    # reusing a cached 2D pose needs no footage.
    if stages["pose2d"] and (
        "pose2d" in overwrite or not _stage_cached("pose2d", cached, config, outdir)
    ):
        _require_input_footage(args, config)

    outdir.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", outdir)
    config.save_snapshot(outdir)
    log.info(
        "stages: %s",
        ", ".join(f"{n}={'on' if stages[n] else 'off'}" for n in STAGES),
    )

    result = cached
    frames = candidates = None
    produced = False  # whether we computed new 2D/3D worth persisting
    recomputed = False  # has any enabled stage recomputed this run? -> cascade
    fresh_rig = (
        False  # are result.cameras the un-refined config rig (vs. cached BA output)?
    )

    def _recompute(stage: str) -> bool:
        """Whether enabled ``stage`` should recompute rather than reuse its cache."""
        if stage in overwrite or recomputed:
            return True
        if _stage_cached(stage, cached, config, outdir):
            log.info(
                "reusing cached %s (pass --overwrite %s to recompute)", stage, stage
            )
            return False
        return True

    if stages["pose2d"] and _recompute("pose2d"):
        result, candidates, frames, _ = _stage_pose2d(
            args, config, want_candidates=stages["pictorial_structures"]
        )
        produced = recomputed = True
        fresh_rig = True  # pose2d built the cameras from the config

    if stages["bundle_adjustment"] and _recompute("bundle_adjustment"):
        if _has_2d(result):
            # If the rig to refine is itself a *previous* BA output (cached and
            # marked), rebuild the un-refined config rig first, so this recompute
            # starts from the (edited) config instead of re-refining already-refined
            # cameras. Cached cameras never BA-refined are kept as-is.
            if (
                not fresh_rig
                and cached is not None
                and cached.meta.get("bundle_adjustment")
            ):
                try:
                    result.cameras = _config_camera_rig(args, config)
                    fresh_rig = True
                except SystemExit as exc:
                    log.warning(
                        "bundle_adjustment: could not rebuild the camera rig from the "
                        "config (%s) -- refining the cached cameras instead; re-run "
                        "from pose2d with the recording to recalibrate from scratch",
                        exc,
                    )
            result = _stage_bundle_adjustment(config, result)
            produced = recomputed = True
        else:
            log.warning(
                "skipping bundle_adjustment: no 2D pose available -- enable "
                "[pipeline].do_pose2d or leave a cached poses.h5 with 2D in %s",
                outdir,
            )

    if stages["pictorial_structures"] and _recompute("pictorial_structures"):
        if candidates is not None and _has_2d(result):
            result = _stage_pictorial_structures(config, result, candidates)
            produced = recomputed = True
        elif not _has_2d(result):
            log.warning(
                "skipping pictorial_structures: no 2D pose available -- enable "
                "[pipeline].do_pose2d or leave a cached poses.h5 with 2D in %s",
                outdir,
            )
        else:
            log.warning(
                "skipping pictorial_structures: it needs the detector's top-K "
                "candidates, which a cached 2D result does not store -- enable "
                "[pipeline].do_pose2d to re-detect from the recording"
            )

    if stages["triangulation"] and _recompute("triangulation"):
        if _has_2d(result):
            result = _stage_triangulation(config, result)
            produced = recomputed = True
        else:
            log.warning(
                "skipping triangulation: no 2D pose available -- enable "
                "[pipeline].do_pose2d or leave a cached poses.h5 with 2D in %s",
                outdir,
            )

    if stages["smoothing"] and _recompute("smoothing"):
        if result is not None and result.pts3d is not None:
            result = _stage_smoothing(args, config, result)
            produced = recomputed = True
        else:
            log.warning(
                "skipping smoothing: no 3D pose available -- enable "
                "[pipeline].do_triangulation or do_pictorial_structures, or leave a "
                "cached poses.h5 with 3D in %s",
                outdir,
            )

    if produced and result is not None:
        result.save(h5_path)
        log.info(
            "wrote %s  (%d frames, %d views)", h5_path, result.n_frames, result.n_views
        )

    if stages["visualization"] and _recompute("visualization"):
        if result is not None:
            _stage_visualization(args, config, result, frames, outdir)
        else:
            log.warning(
                "skipping visualization: no pose result available -- enable a stage "
                "above or leave a cached poses.h5 in %s",
                outdir,
            )


def _cmd_run(args: argparse.Namespace) -> None:
    """Run the pipeline for each recording the inputs resolve to.

    ``inputs`` is one or more recording directories and/or wildcard/recursive
    patterns (see :func:`_resolve_recordings`). In a batch each run is independent
    and a failure is logged and skipped (non-zero exit if any failed); a single
    recording fails fast.
    """
    if not args.inputs:
        raise SystemExit("give at least one recording directory (or wildcard) to run")
    # Only used to recognize recording directories while resolving the inputs; each
    # run then resolves its own config against its output dir (Config.read_for_run).
    discovery_config = Config.read(args.config) if args.config else Config.default()
    recordings = _resolve_recordings(
        args.inputs,
        recursive=args.recursive,
        config=discovery_config,
        output=args.output,
    )
    batch = len(recordings) > 1
    if batch:
        log.info(
            "matched %d recordings (output dirs): %s",
            len(recordings),
            [str(r.outdir) for r in recordings],
        )

    failures: list[Path] = []
    for i, rec in enumerate(recordings, 1):
        if batch:
            console.rule(
                Text(f"{rec.outdir}  ({i}/{len(recordings)})", style="bold cyan")
            )
        run_args = copy.copy(args)
        run_args.input = None  # the recording dir is not threaded; sources/outdir are
        run_args.sources = rec.sources  # footage resolved up front by discovery
        try:
            _run_one(run_args, rec.outdir)
        except (Exception, SystemExit) as exc:  # noqa: BLE001
            if not batch:
                raise  # a single recording fails fast (unchanged behavior)
            log.error("recording %s failed: %s", rec.outdir, exc)
            failures.append(rec.outdir)

    if failures:
        raise SystemExit(
            f"{len(failures)}/{len(recordings)} recordings failed: "
            + ", ".join(str(r) for r in failures)
        )


def _info_line(label: str, value: object) -> None:
    """Print one ``label   value`` row with a colored label.

    Built as :class:`rich.text.Text` (not markup) so values containing brackets
    (e.g. the camera-name list) are never parsed as style tags.
    """
    line = Text(label, style="bold cyan")
    line.append(str(value))
    console.print(line)


def _cmd_inspect(args: argparse.Namespace) -> None:
    result = PoseResult.load(args.input)
    _info_line("file:     ", args.input)
    _info_line("views:    ", f"{result.n_views}  {result.cameras.names}")
    _info_line("frames:   ", result.n_frames)
    _info_line(
        "skeleton: ", f"{result.skeleton.name}  ({result.skeleton.n_points} points)"
    )
    _info_line("has 3D:   ", result.pts3d is not None)
    if result.reproj_error is not None:
        _info_line(
            "reproj:   ",
            f"median {np.nanmedian(result.reproj_error):.3f} px"
            f"  max {np.nanmax(result.reproj_error):.3f} px",
        )


# -- doctor: installation / runtime report -----------------------------------


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size (``1.2 GiB``)."""
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def _by_priority(available, *orders) -> list[str]:
    """``available`` backend names in their ``backend="auto"`` preference order.

    Walks each preference tuple in ``orders``, keeping the first occurrence of each
    installed backend, then appends any remaining installed ones (alphabetically).
    Mirrors how ``select_reader``/``select_writer`` pick a backend, so the report
    lists the highest-priority installed decoder first.
    """
    avail = set(available)
    ranked: list[str] = []
    for order in orders:
        ranked += [b for b in order if b in avail and b not in ranked]
    ranked += sorted(b for b in avail if b not in ranked)
    return ranked


def _doctor_header(title: str) -> None:
    """Print a blank line then a section title (its own colored line)."""
    console.print()
    console.print(Text(title, style="bold magenta"))


def _doctor_row(label: str, value: object, *, width: int = 18) -> None:
    """Print one indented ``label   value`` row, label padded to ``width``.

    Built as :class:`~rich.text.Text` (not markup) so values containing brackets
    (e.g. JAX's ``[cuda:0]`` device list) are never parsed as style tags.
    """
    line = Text("  ")
    line.append(f"{label:<{width}}", style="bold cyan")
    line.append(str(value))
    console.print(line)


def _probe_torch() -> dict:
    """PyTorch presence + accelerator availability, without raising.

    Probing CUDA/MPS can fail on a broken install, so every query is guarded and
    missing keys mean "unknown/no".
    """
    info: dict = {"installed": False}
    try:
        import torch
    except Exception:  # noqa: BLE001
        return info
    info.update(installed=True, version=torch.__version__)
    try:
        if torch.cuda.is_available():
            info["cuda"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    try:
        info["mps"] = bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        pass
    return info


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Report the installation and what this machine can run.

    Covers version + location, Python/OS, CPU/GPU inference (torch CUDA/MPS), the
    installed video backends, whether the detector weights are downloaded and
    where, and the default config path. Imports are lazy and each probe guarded, so
    a missing or broken piece is reported rather than crashing.
    """
    import importlib.metadata
    import platform

    from . import video
    from .pose2d import backends, download
    from .video.base import IMAGE_READ_ORDER, READ_ORDER, WRITE_ORDER

    _doctor_header("deeperfly")
    try:
        version = importlib.metadata.version("deeperfly")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown (not installed as a package)"
    _doctor_row("version", version)
    _doctor_row("location", Path(__file__).resolve().parent)

    _doctor_header("system")
    _doctor_row(
        "python", f"{platform.python_version()} ({platform.python_implementation()})"
    )
    _doctor_row("platform", platform.platform())

    torch_info = _probe_torch()
    _doctor_header("inference")
    if torch_info["installed"]:
        accel = []
        if "cuda" in torch_info:
            accel.append(f"CUDA: {torch_info['cuda']}")
        if torch_info.get("mps"):
            accel.append("Metal (MPS)")
        _doctor_row(
            "torch",
            f"{torch_info['version']}  ({', '.join(accel) if accel else 'CPU only'})",
        )
    else:
        _doctor_row("torch", "not installed")

    gpu = "cuda" in torch_info or torch_info.get("mps")
    mem = backends.gpu_memory_bytes()
    if gpu:
        _doctor_row(
            "GPU inference",
            f"available ({_fmt_bytes(mem)} memory)" if mem else "available",
        )
    else:
        _doctor_row("GPU inference", "not available -- CPU only")
    _doctor_row("detector", "torch" if torch_info["installed"] else "none")

    _doctor_header("frame I/O backends")
    read_avail = video.available_read_backends()
    write_avail = video.available_write_backends()
    image_avail = video.available_image_readers()
    _doctor_row("video read", ", ".join(_by_priority(read_avail, READ_ORDER)) or "none")
    _doctor_row(
        "video write", ", ".join(_by_priority(write_avail, WRITE_ORDER)) or "none"
    )
    _doctor_row(
        "image read", ", ".join(_by_priority(image_avail, IMAGE_READ_ORDER)) or "none"
    )
    missing = sorted(
        set(
            video.list_read_backends()
            + video.list_write_backends()
            + video.list_image_readers()
        )
        - set(read_avail)
        - set(write_avail)
        - set(image_avail)
    )
    if missing:
        _doctor_row("not installed", ", ".join(missing))

    _doctor_header("weights")
    _doctor_row("cache dir", download.cache_dir())
    path = download.torch_weights_path()
    if path.exists():
        state = f"downloaded ({_fmt_bytes(path.stat().st_size)}) -- {path.name}"
    else:
        state = f"not downloaded -- would cache as {path.name}"
    _doctor_row("detector", state)

    _doctor_header("config")
    _doctor_row("default config", DEFAULT_CONFIG_PATH)


# -- typer app ---------------------------------------------------------------
#
# The CLI is built with Typer: typed signatures over click, with usage/--help
# rendered through rich. Each command declares its options, configures logging,
# then hands an argparse-style namespace to the matching ``_cmd_*`` worker; the
# workers stay namespace-driven so they remain callable as a library and from the
# tests. Constrained options are ``str``-valued Enums; commands pass their
# ``.value``, so workers keep receiving plain strings.


class LogLevel(str, Enum):
    """``--log-level`` choices, shared by every subcommand. A ``str`` enum, so each
    member's ``.value`` is the name :func:`_configure_logging` expects."""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


#: The shared ``--log-level`` option, declared once as a reusable parameter
#: annotation and spread across every command. ``case_sensitive=False`` accepts
#: INFO/Info/info.
LogLevelOption = Annotated[
    LogLevel,
    typer.Option(
        case_sensitive=False,
        help="logging verbosity; 'warning' or higher hides the per-stage logs and "
        "the progress bar",
    ),
]

app = typer.Typer(
    add_completion=False,  # no shell-completion options; keep the surface minimal
    rich_markup_mode="rich",  # rich-rendered (boxed, colored) usage and --help
    no_args_is_help=True,  # bare 'deeperfly' prints help instead of a usage error
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Markerless 3D pose estimation of tethered Drosophila from a multi-camera "
    "rig. 'deeperfly init' writes a config to edit; 'deeperfly run' detects 2D "
    "pose, reconstructs 3D and renders a video; 'deeperfly inspect' summarizes a "
    "result file; 'deeperfly doctor' reports the installation/runtime.",
)


@app.command()
def init(
    output: Annotated[
        str, typer.Argument(help="destination (defaults to config.toml)")
    ] = "config.toml",
    force: Annotated[
        bool, typer.Option("--force", help="overwrite an existing file")
    ] = False,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Write a default config.toml to edit (destination defaults to config.toml)."""
    _configure_logging(log_level.value)
    _cmd_init(argparse.Namespace(output=output, force=force))


@app.command()
def run(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            metavar="INPUT...",
            help="one or more recording dirs or wildcard patterns (per-camera videos "
            "or image folders); several inputs / a wildcard run as a batch",
        ),
    ],
    recursive: Annotated[
        bool,
        typer.Option(
            "-r",
            "--recursive",
            help="treat each INPUT as a parent directory and run every recording "
            "nested under it (each subdirectory holding the configured per-camera "
            "footage)",
        ),
    ] = False,
    config: Annotated[
        str | None,
        typer.Option(
            "-c",
            "--config",
            help="merged config TOML (from 'deeperfly init'); "
            "defaults to the packaged default config",
        ),
    ] = None,
    output: Annotated[
        str | None,
        typer.Option(
            "-o",
            "--output-dir",
            help="output directory (default: <input>/deeperfly_outputs; created if "
            "missing)",
        ),
    ] = None,
    overwrite: Annotated[
        list[str] | None,
        typer.Option(
            "--overwrite",
            help="recompute stages instead of reusing results already in the output "
            "dir. A bare --overwrite recomputes everything; name stages to recompute "
            "only those (e.g. --overwrite pose2d visualization). Recomputing a stage "
            "also refreshes the stages after it.",
        ),
    ] = None,
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """detect 2D -> reconstruct 3D -> visualization (the enabled stages, reusing cache).

    INPUT is one or more recording directories (per-camera videos or image folders)
    and/or wildcards matching several (e.g. 'fly*' -> fly1/, fly2/, ...), each run
    in turn. Several inputs or a wildcard run as a batch, keeping only the valid
    recordings. With -r/--recursive, each INPUT is a parent directory and every
    recording nested under it is run in turn.

    By default a stage already in the output dir is reused, so re-running a finished
    recording is a cheap no-op. Pass --overwrite to recompute: bare redoes every
    stage, or name stages to redo only those (plus the stages after them).

    Everything else is set in the config: the do_<stage> toggles choose which stages
    run, alongside fps, background and each stage's parameters. With no -c, a run
    reuses the config.toml already in the output dir, else the packaged default.
    """
    _configure_logging(log_level.value)
    _cmd_run(
        argparse.Namespace(
            inputs=inputs,
            recursive=recursive,
            config=config,
            output=output,
            overwrite=overwrite,
            log_level=log_level.value,
        )
    )


@app.command()
def inspect(
    input: Annotated[str, typer.Argument(help="path to a result .h5 file")],
    log_level: LogLevelOption = LogLevel.info,
) -> None:
    """Print a summary of a result .h5 file."""
    _configure_logging(log_level.value)
    _cmd_inspect(argparse.Namespace(input=input))


@app.command()
def doctor(log_level: LogLevelOption = LogLevel.info) -> None:
    """Report installation/runtime: accelerators, video backends, weights."""
    _configure_logging(log_level.value)
    _cmd_doctor(argparse.Namespace())


def _normalize_overwrite_argv(argv: list[str]) -> list[str]:
    """Let ``run``'s ``--overwrite`` take zero or more space-separated stage names.

    click options can't be variadic, so rewrite a bare ``--overwrite`` into
    ``--overwrite <_OVERWRITE_ALL>`` and ``--overwrite a b`` into the repeated
    ``--overwrite a --overwrite b`` the ``multiple=True`` option accepts. Only known
    stage names (:data:`STAGES`) are consumed after the flag, leaving the positional
    argument and later options untouched; ``--overwrite=...`` passes through as-is.
    """
    if not (argv and argv[0] == "run"):
        return argv
    out: list[str] = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok != "--overwrite":
            out.append(tok)
            i += 1
            continue
        j = i + 1
        picked: list[str] = []
        while j < len(argv) and argv[j] in STAGES:
            picked.append(argv[j])
            j += 1
        for stage in picked or [_OVERWRITE_ALL]:
            out += ["--overwrite", stage]
        i = j
    return out


def main(argv: list[str] | None = None) -> None:
    """Entry point: parse ``argv`` (default ``sys.argv``) and dispatch a subcommand.

    Runs the Typer app in standalone mode so usage errors and ``--help`` render
    through rich, but swallows the ``SystemExit(0)`` of a clean exit so
    ``main([...])`` returns normally as a library / from the tests. Real failures
    still propagate. ``argv`` is normalized first
    (:func:`_normalize_overwrite_argv`).
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    argv = _normalize_overwrite_argv(argv)
    command = typer.main.get_command(app)
    try:
        command(args=argv, prog_name="deeperfly")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
