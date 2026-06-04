"""Command-line interface: ``deeperfly <subcommand>``.

Subcommands are thin wrappers over :mod:`deeperfly.pipeline`, :mod:`deeperfly.io`,
:mod:`deeperfly.video` and :mod:`deeperfly.pose2d`, so everything is equally
usable as a library. Everything a run needs lives in one merged config TOML
(``deeperfly init`` writes a default to edit): the camera rig, the input
filename->camera map, the 2D detector, the pipeline options, bundle adjustment
and the skeleton. The commands:

- ``init`` -- write a default config.toml.
- ``run`` -- the pipeline's enabled stages (``pose2d`` 2D -> ``bundle_adjustment``
  -> ``pictorial_structures`` -> ``triangulation`` -> ``smoothing`` ->
  ``visualization``). One or more recordings are the positional arguments; a
  wildcard pattern (``fly*``, ``data/* data2/*``) and several inputs fan out to
  every matching recording (only the valid ones, non-recording matches skipped),
  and ``-r``/``--recursive`` treats each argument as a parent directory and runs
  every recording nested beneath it -- each run in turn into its own output dir.
  ``-o`` is an output *directory* (default ``<input>/deeperfly_outputs``) that
  collects the result ``poses.h5``, the videos and a copy of the config used.
  ``-c``/``--config`` is the merged config TOML; a run prefers the ``config.toml``
  already snapshotted in the output dir (its source of truth -- even over ``-c``,
  notifying when it ignores one), then ``-c`` if given, else the packaged default.
  The ``[pipeline].do_<stage>`` booleans toggle which stages run: an enabled stage
  recomputes, a disabled one is reused from the cached ``poses.h5``, and an enabled
  stage whose input is missing is skipped with the reason logged. Detector weights
  are downloaded and converted automatically on first use.
- ``inspect`` -- print a summary of a result file.
- ``doctor`` -- print installation/runtime details: package version, whether
  CPU/GPU inference is available, the installed video backends, whether the
  detector weights have been downloaded (and where), and the default config path.

The pipeline is a linear sequence of stages (:data:`STAGES`); each stage's input is
produced by the enabled stage before it or read from the cached artifacts in the
output directory, so disabling the finished stages resumes a partial run.
"""

from __future__ import annotations

import argparse
import copy
import glob
import logging
import os
import sys
import tomllib
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


def _prep_gpu_memory_policy() -> None:
    """Cap JAX's GPU memory pool so the detector can share VRAM with on-device
    video frames, *before* anything initializes the JAX backend.

    ``deeperfly run`` decodes each camera's frames onto the GPU (NVDEC) when the
    JAX detector runs there; JAX's default ~75% preallocation then collides with
    those frame tensors and XLA logs alarming ``CUDA_ERROR_OUT_OF_MEMORY`` lines
    (it recovers, but they look like a crash). The detector's forward pass is
    small, so half the GPU is ample and leaves room for the frames. This must run
    before any import initializes the JAX backend (e.g. ``import .geometry`` ->
    ``import ._jax_cpu``, which calls ``jax.devices`` and would lock in the memory
    policy), and uses an import-free GPU probe so it does not pull in or initialize
    torch/JAX. Only for ``run``; honors an override.
    """
    if not (len(sys.argv) > 1 and sys.argv[1] == "run"):
        return
    if any(k.startswith("XLA_PYTHON_CLIENT_") for k in os.environ):
        return  # respect a user-chosen JAX memory policy
    if glob.glob("/dev/nvidia[0-9]*"):  # an NVIDIA GPU is present
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"


_prep_gpu_memory_policy()  # before the imports below initialize the JAX backend

import numpy as np  # noqa: E402  (must follow the GPU memory policy above)

from .cameras import CameraGroup  # noqa: E402
from .io import PoseResult  # noqa: E402

#: Packaged template emitted by ``deeperfly init`` (also the run-config example).
DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "default_config.toml"


#: The linear pipeline stages, in run order. Each is independently toggled by a
#: ``[pipeline].do_<stage>`` boolean (see :func:`_stage_flags`) and parameterized
#: by its own ``[pipeline.<stage>]`` sub-table.
STAGES = (
    "pose2d",
    "bundle_adjustment",
    "pictorial_structures",
    "triangulation",
    "smoothing",
    "visualization",
)

#: Default for each ``do_<stage>`` when the key is omitted: detection,
#: calibration, triangulation and visualization run by default; pictorial
#: structures and smoothing are opt-in.
STAGE_DEFAULTS = {
    "pose2d": True,
    "bundle_adjustment": True,
    "pictorial_structures": False,
    "triangulation": True,
    "smoothing": False,
    "visualization": True,
}

#: Frames decoded + detected per streaming window, per camera (overridable via
#: ``[detector] chunk_frames``). Bounds peak frame memory, so arbitrarily long
#: recordings run in constant memory. A *memory* knob, not a speed one: detection
#: is the bottleneck, so every decoder outpaces it and a small window costs no
#: throughput. 64 holds ~0.6 GB of frames for a 7-camera 480x960 rig.
DEFAULT_CHUNK_FRAMES = 64

#: Human-facing output goes through rich: status/results to stdout, while logs and
#: the detection progress bar share the stderr console (so piping stdout to a file
#: stays clean and progress never clobbers a log line).
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


def _load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _configure_logging(level_name: str) -> None:
    """Configure the root log level from a ``--log-level`` name (default ``info``).

    The default ``info`` surfaces the per-stage progress messages (and the bar);
    ``warning`` or higher hides them, so it doubles as the "quiet" mode. The
    progress bar follows the same line: it shows only while INFO logging is
    enabled (see :func:`_detect_2d`).

    Records render through rich's :class:`~rich.logging.RichHandler` (colored level
    column, messages wrapped to the terminal) on the same stderr console as the
    progress bar, so log lines and the bar never overwrite each other.
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
    # JAX probes every platform on first use and warns when the TPU plugin's
    # libtpu.so is absent (the normal case on a CPU/GPU box). Mute that noise
    # unless we're at debug, where seeing every backend probe is useful.
    if level > logging.DEBUG:
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)


def _load_detector(checkpoint: str | None, backend: str):
    """Load the JAX detector (native .eqx) or the PyTorch detector (.pth).

    With no explicit ``checkpoint`` the cached weights are used, provisioning them
    on demand: the PyTorch checkpoint is downloaded, and for the JAX backend it is
    also converted to a native checkpoint (:func:`ensure_jax_weights`). An explicit
    but missing ``checkpoint`` is an error (we never write to a user-named path).
    """
    from .pose2d import backends

    if backend == "torch":
        from .pose2d.download import download_torch_weights

        path = checkpoint or download_torch_weights()
        return backends.load_detector("torch", path)

    from .pose2d.download import ensure_jax_weights

    if checkpoint is not None:
        if not Path(checkpoint).exists():
            raise SystemExit(
                f"no JAX checkpoint at {checkpoint}. Remove [detector].checkpoint "
                "to use the auto-provisioned cache, or point it at a valid .eqx."
            )
        path: str | Path = checkpoint
    else:
        path = ensure_jax_weights()
    return backends.load_detector("jax", path)


# -- config-driven option resolution -----------------------------------------


def _pose2d_config(config: dict) -> dict:
    """The ``[pipeline.pose2d]`` detector/decoder sub-table (empty if absent)."""
    return config.get("pipeline", {}).get("pose2d", {})


#: Frame rate used when ``[pipeline].fps`` is unset and none can be detected from
#: the recording (e.g. an image sequence carries no intrinsic rate). Matches the
#: historical default.
_FPS_FALLBACK = 100.0


def _detect_input_fps(args: argparse.Namespace, config: dict) -> float | None:
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


def _resolve_fps(args: argparse.Namespace, config: dict) -> float:
    """The recording's frame rate, for one_euro smoothing and the visualization base.

    Uses ``[pipeline].fps`` when set; otherwise detects it from the input videos
    (:func:`_detect_input_fps`). Falls back to :data:`_FPS_FALLBACK` when neither
    is available -- e.g. an image sequence, or a cache-only resume with no
    recording -- logging a hint to set ``[pipeline].fps`` explicitly.
    """
    fps = config.get("pipeline", {}).get("fps")
    if fps is not None:
        return float(fps)
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


def _bundle_adjustment_kwargs(config: dict) -> dict:
    """Options for :func:`deeperfly.pipeline.calibrate` from ``[pipeline.bundle_adjustment]``.

    Reads ``keypoints`` (-> ``ba_keypoints``), ``fixed``, ``shared`` and the solver
    sub-table (e.g. ``[pipeline.bundle_adjustment.least_squares_scipy]``, forwarded
    as solver kwargs like ``max_nfev`` / ``loss``). Anything omitted falls through
    to ``calibrate``'s own defaults.
    """
    ba = config.get("pipeline", {}).get("bundle_adjustment", {})
    out: dict = {}
    if "keypoints" in ba:
        out["ba_keypoints"] = ba["keypoints"]
    if "fixed" in ba:
        out["fixed"] = ba["fixed"]
    if "shared" in ba:
        out["shared"] = ba["shared"]
    sub = ba
    for part in ba.get("solver", "least_squares_scipy").split("."):
        sub = sub.get(part, {}) if isinstance(sub, dict) else {}
    out.update(sub)  # e.g. max_nfev, loss
    return out


def _pictorial_kwargs(config: dict) -> dict:
    """Keyword args for :func:`deeperfly.pictorial.reconstruct` from ``[pipeline.pictorial_structures]``."""
    ps = config.get("pipeline", {}).get("pictorial_structures", {})
    return {"temporal": ps.get("temporal", False), "lam": ps.get("lam", 1.0)}


def _triangulation_options(config: dict) -> dict:
    """Method + thresholds for the triangulation stage from ``[pipeline.triangulation]``."""
    tri = config.get("pipeline", {}).get("triangulation", {})
    return {
        "method": tri.get("method", "ransac"),
        "ransac_threshold": tri.get("ransac_threshold", 15.0),
        "min_inliers": tri.get("min_inliers", 2),
        "reproj_threshold": tri.get("reproj_threshold", 40.0),
        "max_drops": tri.get("max_drops", 5),
    }


def _smoothing_options(config: dict) -> tuple[str, dict]:
    """``(method, extra_kwargs)`` for the smoothing stage from ``[pipeline.smoothing]``."""
    sm = dict(config.get("pipeline", {}).get("smoothing", {}))
    method = sm.pop("method", "gaussian")
    return method, sm


# -- input -> camera frame resolution ----------------------------------------


def _footage_exts() -> tuple[str, ...]:
    """Footage extensions deeperfly can read, in priority order (video before image).

    Used to recognize a camera's frames and, when a recording folder mixes several,
    to pick the one to keep (the earliest here wins). Imported lazily so this module
    does not pull in the video stack just to resolve filenames.
    """
    from .video.io import _IMAGE_EXTS, _VIDEO_EXTS

    return _VIDEO_EXTS + _IMAGE_EXTS


def _is_video_ext(suffix: str) -> bool:
    """Whether a file ``suffix`` (e.g. ``.mp4``) is a video extension -- as opposed to
    an image one. Video footage is a single file per camera; images form a sequence."""
    from .video.io import _VIDEO_EXTS

    return suffix.lower() in _VIDEO_EXTS


def _camera_glob(pattern: str) -> str:
    """A camera's ``[inputs]`` value as a filename glob.

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
    """A camera's footage files under ``root`` matching its ``[inputs]`` ``pattern``.

    Globs ``pattern`` (see :func:`_camera_glob`), keeps only files with a known
    footage extension, and -- when several extensions match (e.g. a stray
    ``camera_0.png`` next to ``camera_0.mp4``) -- keeps the single highest-priority
    one. Naturally sorted (so ``img2`` precedes ``img10``). Video footage is a single
    file, so if several videos match only the first is kept (warned); images stay as
    the whole sequence. Empty when nothing footage-like matches, so the caller can
    treat the camera as absent.
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


def _frame_read_device(config: dict) -> str:
    """Where to decode frames for the detector. Defaults to the CPU.

    CPU decode is within a few percent of GPU/NVDEC once each window is uploaded in
    one shot (see :func:`deeperfly.pose2d.inference._window_to_device`) -- decode is
    never the bottleneck, the forward is -- and it keeps the decoder off the GPU and
    out of the CUDA-video dependency stack (see ``dev/bench_video.py``). Opt into
    on-device (NVDEC) decode, which feeds the JAX network zero-copy via DLPack, with
    ``[detector] decode_device = "cuda"`` (alias ``"gpu"``, or ``"auto"``); it is
    worth it only on the fastest GPUs. ``read_frames`` still falls back to the CPU
    if no GPU video backend can actually decode here. The torch detector backend
    always decodes on the CPU (it copies inputs to host internally, so GPU decode
    would not help).
    """
    det = _pose2d_config(config)
    device = det.get("decode_device", "cpu")
    if device == "cpu" or det.get("backend", "jax") != "jax":
        return "cpu"
    return device  # "cuda"/"gpu" / "auto": opt-in on-device decode for the JAX backend


def _camera_patterns(config: dict) -> dict[str, str]:
    """``camera-name -> [inputs] glob pattern``, in ``[cameras]`` order.

    A camera with no ``[inputs]`` entry defaults to its own name as the pattern.
    Falls back to the ``[inputs]`` keys when the config carries no ``[cameras]``
    table (the recording-discovery configs in the tests do this).
    """
    inputs = config.get("inputs", {})
    names = config.get("cameras", {}) or inputs
    return {name: inputs.get(name, name) for name in names}


def _camera_sources(
    args: argparse.Namespace, config: dict
) -> list[tuple[str, list[Path]]]:
    """``(name, footage-files)`` per camera (in ``[cameras]`` order).

    Prefers the files ``deeperfly run`` already resolved for this recording
    (``args.sources``, see :func:`_resolve_recordings`) so the footage is globbed
    once per run; otherwise -- a library caller that set only ``args.input`` -- it
    resolves each camera from ``args.input`` with the ``[inputs]`` patterns. With
    neither, every camera resolves to an empty list. Each source is the list passed
    to :func:`deeperfly.video.read_frames` (a single video file, or an image sequence).
    """
    patterns = _camera_patterns(config)
    pre = getattr(args, "sources", None)
    if pre and all(name in pre for name in patterns):
        return [(name, pre[name]) for name in patterns]
    root = getattr(args, "input", None)
    if root is None:
        return [(name, []) for name in patterns]
    return [(name, _camera_files(Path(root), pat)) for name, pat in patterns.items()]


def _camera_image_sizes(args, config: dict) -> dict[str, tuple[int, int]]:
    """``name -> (height, width)`` from a single frame per camera.

    Used to infer each view's principal point. Reads only frame 0 (host), so it is
    cheap and independent of the full streaming decode.
    """
    from . import video

    backend = _pose2d_config(config).get("video_backend", "auto")
    device = _frame_read_device(config)  # match detection (CPU by default)
    # Size the principal point on the *transformed* frame -- the detector and the
    # overlays use the [preprocess.*]-transformed footage, so a rot90 that swaps
    # H/W must swap here too.
    transforms = video.parse_frame_transforms(config)
    sizes: dict[str, tuple[int, int]] = {}
    for name, src in _camera_sources(args, config):
        head = video.read_frames(src, backend=backend, device=device, indices=[0])
        head = transforms.get(name, video.FrameTransform()).apply(head)
        sizes[name] = tuple(int(d) for d in head.shape[1:3])
    return sizes


def _prefetch_windows(sources, *, backend, device, chunk, transforms=None, depth=1):
    """Yield ``(window, n)`` decoded frame windows, decoding ``depth`` ahead.

    A background producer decodes the next window while the consumer detects the
    current one, so decode overlaps the GPU forward instead of running before it.
    The win is largest for CPU decode (it runs on otherwise-idle cores fully in
    parallel with the GPU); GPU decode shares the device, so it overlaps less.

    ``transforms`` is an optional per-source :class:`~deeperfly.video.FrameTransform`
    (aligned to ``sources``) applied to each decoded window before it is yielded,
    so detection sees the configured ``[preprocess.*]`` orientation.

    EOF semantics match the old serial loop: a short or empty window ends the
    stream, and a read failure on the *very first* window propagates (anything
    later is treated as "past the end").
    """
    import queue
    import threading

    from . import video

    if transforms is None:
        transforms = [video.FrameTransform()] * len(sources)
    q: queue.Queue = queue.Queue(maxsize=depth)
    DONE = object()

    def produce():
        done = 0
        while True:
            try:
                window = [
                    transforms[i].apply(
                        video.read_frames(
                            s,
                            backend=backend,
                            device=device,
                            start=done,
                            stop=done + chunk,
                        )
                    )
                    for i, s in enumerate(sources)
                ]
            except Exception as exc:  # noqa: BLE001
                q.put(("err", exc) if done == 0 else DONE)
                return
            n = len(window[0])
            if n == 0:
                q.put(DONE)
                return
            q.put(("win", window, n))
            done += n
            if n < chunk:  # a short window is the last one
                q.put(DONE)
                return

    threading.Thread(target=produce, daemon=True).start()
    while True:
        item = q.get()
        if item is DONE:
            return
        if item[0] == "err":
            raise item[1]
        yield item[1], item[2]


def _detect_2d(args, config: dict, model, sides, flips, *, want_candidates, k):
    """Stream 2D detection over fixed-size frame windows -> ``(pts2d, conf, candidates)``.

    Decodes and detects ``[detector] chunk_frames`` frames at a time per camera and
    frees each window before the next, so peak frame memory is bounded by the chunk
    size, **not** the recording length -- the key to handling long videos. On the
    GPU path each window is decoded onto the device (NVDEC), detected zero-copy, and
    released. Per-window ``(V, w, ...)`` results are concatenated along time.

    End-of-file is taken from the decoder (a short or empty window), so it does not
    depend on :func:`deeperfly.video.count_frames` being exact -- that is only the
    progress-bar total.
    """
    from . import video
    from .pictorial import Candidates
    from .pose2d import auto_batch_size, inference

    det = _pose2d_config(config)
    backend = det.get("video_backend", "auto")
    device = _frame_read_device(config)
    chunk = max(1, int(det.get("chunk_frames", DEFAULT_CHUNK_FRAMES)))
    cam_sources = _camera_sources(args, config)
    sources = [src for _, src in cam_sources]
    # Apply each camera's [preprocess.*] transform to its decoded window, so the
    # detector sees the corrected orientation (and 2D points land in that frame).
    transforms_by_name = video.parse_frame_transforms(config)
    transforms = [
        transforms_by_name.get(name, video.FrameTransform()) for name, _ in cam_sources
    ]
    total = video.count_frames(sources[0]) if sources else 0
    # Size the forward to the GPU so each window is a few big batches, not one
    # batch per frame -- this only changes dispatch granularity, not the result.
    batch_size = auto_batch_size(inference.IMG_SIZE)

    # One-line summary instead of a per-camera/per-window read log (that spam is at
    # -vv now; see deeperfly.video.io). reader_name mirrors read_frames's dispatch
    # off the actual source (video file -> the decode backend; image folder/glob ->
    # imageio/nvjpeg, *not* the video backend), so the reported decoder matches what
    # really runs; guard since a forced GPU backend may be uninstalled.
    try:
        reader = video.reader_name(sources[0], backend=backend, device=device)
    except Exception:  # noqa: BLE001
        reader = backend
    log.info(
        "reading frames via '%s' backend: %d/read per camera (device=%s), forward batch %d",
        reader,
        chunk,
        device,
        batch_size,
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
            sources, backend=backend, device=device, chunk=chunk, transforms=transforms
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


#: Keys removed in the per-stage refactor, mapped to their new home so an old
#: config fails with a fix-it message instead of being silently ignored.
_REMOVED_PIPELINE_KEYS = {
    "calibrate": "do_bundle_adjustment",
    "do_calibrate": "do_bundle_adjustment",
    "do_pictorial": "do_pictorial_structures",
    "do_visualize": "do_visualization",
    "triangulation": "[pipeline.triangulation].method",
    "ransac_threshold": "[pipeline.triangulation].ransac_threshold",
    "min_inliers": "[pipeline.triangulation].min_inliers",
    "smooth": "do_smoothing + [pipeline.smoothing].method",
}


def _stage_flags(config: dict) -> dict[str, bool]:
    """Which stages are enabled, from the ``[pipeline].do_<stage>`` booleans.

    Each stage (:data:`STAGES`) has its own boolean defaulting to
    :data:`STAGE_DEFAULTS`: an enabled stage runs (so a "resume" is just disabling
    the stages already done, whose cached results in the output dir then feed the
    enabled ones); a disabled stage never executes but its cached output, if
    present, still feeds downstream. Unknown ``do_*`` keys -- and keys removed in
    the per-stage refactor (the old ``[stages]`` table, ``[pipeline].calibrate`` /
    ``triangulation`` / ``smooth``) -- fail loudly, pointing at the new location.
    """
    if "stages" in config:
        raise SystemExit(
            "[stages] was removed; the stage toggles now live in [pipeline] as "
            + ", ".join(f"do_{n}" for n in STAGES)
        )
    pipe = config.get("pipeline", {})
    for old, new in _REMOVED_PIPELINE_KEYS.items():
        # A scalar at the old key is the removed usage; a sub-table (dict) of the
        # same name -- e.g. the new [pipeline.triangulation] -- is fine.
        if old in pipe and not isinstance(pipe[old], dict):
            raise SystemExit(f"[pipeline].{old} was removed; use {new}")
    valid = {f"do_{name}" for name in STAGES}
    unknown = {k for k in pipe if k.startswith("do_")} - valid
    if unknown:
        raise SystemExit(
            f"[pipeline] has unknown stage toggle(s) {', '.join(sorted(unknown))}; "
            f"the stages are {', '.join(STAGES)}"
        )
    return {n: bool(pipe.get(f"do_{n}", STAGE_DEFAULTS[n])) for n in STAGES}


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


def _visualization_cached(config: dict, outdir: Path) -> bool:
    """Whether every ``[[pipeline.visualization.videos]]`` MP4 already exists in ``outdir``."""
    from .viz import compose

    specs = compose.read_video_specs(config)
    return bool(specs) and all(
        (outdir / f"{spec.video_name}.mp4").exists() for spec in specs
    )


def _stage_cached(
    stage: str, cached: PoseResult | None, config: dict, outdir: Path
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
    args: argparse.Namespace, config: dict, *, want_candidates: bool
) -> tuple[PoseResult, object | None, None, dict]:
    """Run 2D detection -> a 2D-only :class:`PoseResult` + in-memory artifacts.

    Returns ``(result, candidates, None, image_sizes)``. ``candidates`` is the
    top-K peak set, extracted only when ``want_candidates`` (i.e. the
    ``pictorial_structures`` stage is enabled, since the candidates are not cached).
    Frames are **not** held in memory (detection streams them in windows -- see
    :func:`_detect_2d`), so the third slot is ``None``; the recording path is
    recorded in ``result.meta`` and a visualization stage re-sources the one overlay
    camera it needs.
    """
    from . import video
    from .pose2d import backends, inference
    from .skeleton import Skeleton
    from .triangulate import apply_visibility

    transforms = video.parse_frame_transforms(config)
    image_sizes = _camera_image_sizes(args, config)
    log.info(
        "input image sizes (h, w, after [preprocess.*]): %s",
        {n: tuple(s) for n, s in image_sizes.items()},
    )
    cameras = CameraGroup.from_config(config, image_sizes=image_sizes)
    unknown = set(transforms) - set(cameras.names)
    if unknown:
        log.warning(
            "[preprocess.*] entries for unknown cameras are ignored: %s",
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
    skeleton = Skeleton.from_config(config) if "skeleton" in config else Skeleton.fly()

    det = _pose2d_config(config)
    backend = det.get("backend", "jax")
    log.info(
        "loading %s detector (checkpoint: %s)",
        backend,
        det.get("checkpoint") or "cached",
    )
    model = _load_detector(det.get("checkpoint"), backend)
    log.info("%s detector ready on device %s", backend, backends.detector_device(model))

    k = config.get("pipeline", {}).get("pictorial_structures", {}).get("k", 5)
    sides, flips = inference.fly_camera_layout(cameras.names)
    n_passes = len(inference.expand_passes(sides, flips)[0])
    chunk = max(1, int(det.get("chunk_frames", DEFAULT_CHUNK_FRAMES)))
    log.info(
        "detecting 2D poses: %d views, %d forward passes/frame, network input %dx%d, "
        "streaming in chunks of %d frames",
        len(image_sizes),
        n_passes,
        inference.IMG_SIZE[0],
        inference.IMG_SIZE[1],
        chunk,
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
    # Mask (camera, point) pairs the rig cannot see once, here -- the spot the old
    # monolithic run_from_points2d masked -- so the cached 2D and every downstream
    # stage (BA, pictorial, triangulation) see the same visibility-masked points.
    pts2d = apply_visibility(pts2d, skeleton, cameras.names)

    result = PoseResult(cameras=cameras, skeleton=skeleton, pts2d=pts2d, conf=conf)
    return result, candidates, None, image_sizes


def _config_camera_rig(args: argparse.Namespace, config: dict) -> CameraGroup:
    """The un-refined camera rig straight from the config -- the BA stage's *input*.

    Bundle adjustment refines this config rig; the refined rig is what lands in the
    cached ``poses.h5``. So on a resume that reuses the cached 2D, ``result.cameras``
    are the *previous* BA output, and re-running bundle adjustment (e.g. after editing
    ``[pipeline.bundle_adjustment]`` or the ``[cameras]`` rig) must restart from this
    fresh config rig -- otherwise it just re-refines already-refined cameras and barely
    moves, so the edits look ignored. Reads one frame per camera for the image sizes,
    exactly as ``pose2d`` does, so the principal-point inference matches.
    """
    image_sizes = _camera_image_sizes(args, config)
    return CameraGroup.from_config(config, image_sizes=image_sizes)


def _stage_bundle_adjustment(config: dict, result: PoseResult) -> PoseResult:
    """Refine ``result.cameras`` with bundle adjustment (fly-as-calibration-target).

    Calibrates on the arg-max 2D in ``result`` and replaces its cameras in place.
    The caller hands in the un-refined config rig -- built from the config during
    ``pose2d``, or rebuilt by :func:`_config_camera_rig` on a resume -- so editing
    the ``[cameras]`` rig or ``[pipeline.bundle_adjustment]`` and recomputing this
    stage recalibrates from the edited config rather than from the prior BA output.
    """
    from .pipeline import calibrate
    from .triangulate import reprojection_error, triangulate

    log.info(
        "bundle adjustment: refining cameras (%d frames, %d views)",
        result.n_frames,
        result.n_views,
    )
    result.cameras, _ = calibrate(
        result.cameras,
        result.pts2d,
        result.conf,
        result.skeleton,
        **_bundle_adjustment_kwargs(config),
    )
    result.meta["bundle_adjustment"] = True  # marks the cameras as BA-refined (cache)

    # Report the pixel reprojection error of the refined rig (triangulate the
    # committed 2D with the new cameras and reproject). INFO so it shows at the
    # default log level and above; the triangulation stage refines this further.
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
    config: dict, result: PoseResult, candidates: object
) -> PoseResult:
    """DeepFly3D pictorial-structures recovery over the detector's top-K candidates.

    Commits a corrected per-view 2D (``result.pts2d``) and an initial 3D estimate
    (``result.pts3d``); the triangulation stage, if enabled, then re-triangulates
    the committed 2D. ``candidates`` must come from a ``pose2d`` run in this process
    (they are not cached).
    """
    from . import pictorial

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
        **_pictorial_kwargs(config),
    )
    result.pts2d, result.pts3d, result.reproj_error = pts2d, pts3d, reproj
    result.meta["pictorial"] = True
    return result


def _stage_triangulation(config: dict, result: PoseResult) -> PoseResult:
    """Triangulate the 2D in ``result`` to 3D by the configured method.

    Sets ``result.pts3d`` (and the cleaned ``result.pts2d`` / ``reproj_error`` for
    the outlier-rejecting methods) using ``result.cameras``. ``ransac`` builds each
    point from its largest multi-view consensus, ``greedy`` drops the
    worst-reprojecting view, ``dlt`` is plain least squares (see
    :func:`deeperfly.pipeline._resolve_triangulation`).
    """
    from .pipeline import _resolve_triangulation, reconstruct, reconstruct_ransac
    from .triangulate import reprojection_error, triangulate

    opts = _triangulation_options(config)
    method = _resolve_triangulation(opts["method"])
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
            threshold=opts["ransac_threshold"],
            min_inliers=opts["min_inliers"],
        )
    elif method == "greedy":
        pts3d, pts2d, reproj = reconstruct(
            result.cameras,
            result.pts2d,
            reproj_threshold=opts["reproj_threshold"],
            max_drops=opts["max_drops"],
        )
    else:  # "dlt": plain least-squares triangulation, no outlier handling
        pts2d = result.pts2d
        pts3d = triangulate(result.cameras, pts2d)
        reproj = reprojection_error(result.cameras, pts3d, pts2d)
    result.pts2d, result.pts3d, result.reproj_error = pts2d, pts3d, reproj
    result.meta["triangulation"] = method
    return result


def _stage_smoothing(
    args: argparse.Namespace, config: dict, result: PoseResult
) -> PoseResult:
    """Temporal smoothing of ``result.pts3d`` -> ``result.pts3d_smoothed``."""
    from .correction import smooth_gaussian, smooth_one_euro

    method, kwargs = _smoothing_options(config)
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
    config: dict,
    result: PoseResult,
    views: list[str],
    in_memory: list | None = None,
) -> dict[str, np.ndarray]:
    """Per-view footage for the visualization stage's ``imshow`` panels.

    Uses ``in_memory`` frames (a list indexed by camera order) when available;
    otherwise the footage ``deeperfly run`` resolved for this recording up front
    (``args.sources``). A resume that re-renders the videos just re-passes the
    recording (``deeperfly run <recording> --overwrite visualization``), which
    re-resolves the footage exactly the same way -- so nothing about the source
    recording needs to be stored in the result. Errors if neither is available.
    """
    from . import video

    if not views:
        return {}
    names = result.cameras.names
    # The detector ran on transformed frames, so the 2D/3D overlays live in
    # transformed-frame coordinates; the overlay footage must match (apply the
    # same per-camera [preprocess.*] transform).
    transforms = video.parse_frame_transforms(config)
    backend = _pose2d_config(config).get("video_backend", "auto")

    def transform(v, frames):
        return transforms.get(v, video.FrameTransform()).apply(frames)

    if in_memory is not None:
        return {v: transform(v, in_memory[names.index(v)]) for v in views}

    sources = getattr(args, "sources", None) or {}
    if all(sources.get(v) for v in views):
        return {
            v: transform(v, video.read_frames(sources[v], backend=backend))
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
    config: dict,
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

    specs = compose.read_video_specs(config)
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
    backend = _pose2d_config(config).get("video_backend", "auto")
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
    # markup=False: the message shows literal [inputs]/[cameras] config sections,
    # which rich would otherwise try to parse as style tags.
    console.print(
        "next: edit [inputs]/[cameras] to match your rig, then "
        f"'deeperfly run <recording> -c {dst}' "
        "(outputs land in <recording>/deeperfly_outputs/; override with -o <dir>)",
        markup=False,
        highlight=False,
    )


def _resolve_config(cli_config: str | None, outdir: Path) -> tuple[dict, Path]:
    """Pick the config for one run, preferring the snapshot already in ``outdir``.

    A previous run snapshots its config to ``<outdir>/config.toml``; that snapshot
    is the output dir's source of truth (it owns the cached results and the stage
    toggles that drive a resume), so it wins -- even over an explicit ``-c``,
    notifying that the passed config is ignored. To change it, edit that file (or
    point ``-o`` at a fresh dir). With no snapshot, ``-c`` is used if given, else
    the packaged default config.
    """
    snapshot = outdir / "config.toml"
    if snapshot.exists():
        if cli_config is not None:
            log.warning(
                "using the config already in %s (ignoring -c %s); edit that file to "
                "change the run (e.g. to toggle [pipeline].do_<stage>), or point -o at "
                "a new dir",
                snapshot,
                cli_config,
            )
        log.info("using config %s (snapshot in the output dir)", snapshot)
        return _load_config(snapshot), snapshot
    if cli_config:
        path = Path(cli_config)
        log.info("using config %s (from -c)", path)
    else:
        path = DEFAULT_CONFIG_PATH
        log.info("using config %s (packaged default; pass -c to override)", path)
    return _load_config(path), path


def _save_config_snapshot(config_path: Path, outdir: Path) -> None:
    """Snapshot the run config into ``<outdir>/config.toml`` for reproducibility.

    A no-op rewrite when the config already came from there (see
    :func:`_resolve_config`); otherwise it records the ``-c``/default config that
    produced this run's results so a later resume reuses the very same config.
    """
    (outdir / "config.toml").write_text(config_path.read_text())


def _has_glob(pattern: str) -> bool:
    """Whether ``pattern`` carries a shell wildcard (so it should be expanded)."""
    return any(c in pattern for c in "*?[")


@dataclass(frozen=True)
class Recording:
    """One unit of work: a camera -> footage-files map and where its results go.

    ``sources`` maps a camera name to its naturally-sorted footage files -- a single
    video file, or an image sequence -- already reconciled to one extension and
    validated to share a file and frame count with the other cameras (so
    :func:`deeperfly.video.read_frames` decodes them as one aligned clip). It is empty
    only for a directory kept so a resume can reuse a cached result though its footage
    is absent (see :func:`_resolve_recordings`).

    ``outdir`` is the resolved output directory for this recording (see
    :func:`_run_outdir`); it is the run's durable identity -- it holds the config
    snapshot and the cached ``poses.h5``. The input recording directory is not
    retained: it is only used to resolve ``sources`` and ``outdir`` up front. A resume
    that re-renders just re-passes the recording, which re-resolves ``sources`` the
    same way, so nothing about the source recording is stored in the result.
    """

    sources: dict[str, list[Path]]
    outdir: Path


def _frame_counts_match(root: Path, sources: dict[str, list[Path]]) -> bool:
    """Whether every camera under ``root`` has the same file and frame count.

    File counts are compared directly (cheap). Equal file counts already imply equal
    frame counts for image sequences (one frame per file), so only *video* footage is
    probed for its frame count, and only when the count is actually knowable -- an
    unreadable / metadata-less file (count ``None``) is left out rather than falsely
    rejecting the recording. Warns naming the offending counts when they differ.
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


def _find_recording(root: Path, config: dict) -> dict[str, list[Path]] | None:
    """``root``'s ``camera -> footage-files`` map if it is a recording, else ``None``.

    A *recording* is a directory holding footage for every configured camera (its
    ``[inputs]`` glob, see :func:`_camera_glob`); the footage is a single video file
    or an image sequence. A directory matching *no* camera is silently not a
    recording (an intermediate or output directory); the rest warn and skip:

    - footage for only some cameras (a malformed recording, or a wrong ``[inputs]``);
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

    A wildcard pattern (``fly*``, ``data/*``) is expanded against the filesystem to
    its sorted matches (possibly empty); a literal argument yields just itself. The
    shell usually expands an unquoted ``*`` before the program runs -- so a wildcard
    only reaches us quoted/escaped -- but either way every argument is normalized to
    a list of paths here. ``is_glob`` flags which it was, so a wildcard's incidental
    non-recording matches (e.g. ``data/deeperfly_outputs``) are skipped silently
    while a single literal path the user typed is reported when it is not valid.
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
    inputs: list[Path], *, recursive: bool, config: dict, output: str | None = None
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


def _require_input_footage(args: argparse.Namespace, config: dict) -> None:
    """Fail (before any output dir is created) if the run's recording is unreadable.

    Checked only when the ``pose2d`` stage will actually decode frames; a resume
    that reuses a cached 2D pose needs no footage, so the recording may legitimately
    be absent then. The footage was already resolved up front by
    :func:`_resolve_recordings` (``args.sources``): a recording missing a camera's
    files was warned there and arrives with that camera absent. A library caller that
    set only ``args.input`` is validated directly (naming a missing recording
    directory or camera). Raising here keeps a fresh run that cannot read its input
    from leaving an empty ``deeperfly_outputs`` behind.
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

    The config is resolved against ``outdir`` (see :func:`_resolve_config`) and its
    ``[pipeline].do_<stage>`` toggles decide which stages are part of the pipeline
    (see :func:`_stage_flags`). An enabled stage **reuses its result if it is already
    in the output dir** and only recomputes when that output is missing or
    ``--overwrite`` selects it (see :func:`_stage_cached` / :func:`_overwrite_stages`)
    -- so re-running a finished recording is a cheap no-op by default. Recomputing a
    stage invalidates the ones after it, so once any stage recomputes every enabled
    stage downstream recomputes too (their inputs changed).

    Each stage still runs only if the input it needs is available -- recording
    footage for ``pose2d``, a 2D pose for ``bundle_adjustment`` / ``triangulation``,
    detector candidates for ``pictorial_structures``, a 3D pose for ``smoothing``, a
    pose result for ``visualization`` -- coming either from an upstream stage in this
    run or from the cached ``poses.h5`` in ``outdir``; a stage whose input is missing
    is skipped with the reason logged. A disabled stage never runs but its cached
    output still feeds downstream.
    """
    config, config_path = _resolve_config(args.config, outdir)
    stages = _stage_flags(config)  # validates before we touch the output dir
    overwrite = _overwrite_stages(getattr(args, "overwrite", None))

    # `cached` is the result already in the output dir; `result` starts there and a
    # reused stage keeps it, while a recomputed stage replaces/updates it. Reuse is
    # only safe while nothing upstream has changed, so the first recompute flips
    # `recomputed` and every later enabled stage recomputes too (cascade).
    h5_path = outdir / "poses.h5"
    cached = PoseResult.load(h5_path) if h5_path.exists() else None

    # Validate the recording's footage *before* creating the output dir, so a fresh
    # run that cannot read its input fails cleanly instead of leaving an empty
    # deeperfly_outputs behind. Only pose2d decodes the recording, and only when it
    # recomputes (mirrors `_recompute("pose2d")` -- the first stage, so no cascade
    # yet); a resume reusing a cached 2D pose needs no footage.
    if stages["pose2d"] and (
        "pose2d" in overwrite or not _stage_cached("pose2d", cached, config, outdir)
    ):
        _require_input_footage(args, config)

    outdir.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", outdir)
    _save_config_snapshot(config_path, outdir)
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
            # If the rig we're about to refine is itself a *previous* BA output
            # (cached and marked), rebuild the un-refined config rig first, so this
            # recompute starts from the (edited) config instead of re-refining
            # already-refined cameras and barely moving. Cached cameras that were
            # never BA-refined are legitimate input, so we keep them as-is.
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
    patterns (see :func:`_resolve_recordings`) that fan out to several recordings.
    In a multi-recording batch each run is independent and a failure is logged and
    skipped (so one bad recording does not abort the rest), with a non-zero exit if
    any failed; a single recording fails fast as before.
    """
    if not args.inputs:
        raise SystemExit("give at least one recording directory (or wildcard) to run")
    # Only used to recognize recording directories while resolving the inputs (it
    # reads [cameras]/[inputs]); each run then resolves its own config against its
    # output dir (see _resolve_config), which may differ per recording.
    discovery_config = _load_config(
        Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    )
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

    Built as a :class:`rich.text.Text` (not markup) so dynamic values that contain
    brackets -- e.g. the camera-name list -- are never parsed as style tags.
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

    Walks each preference tuple in ``orders`` (e.g. the CPU then GPU read order),
    keeping the first occurrence of each installed backend, then appends any
    remaining installed ones (alphabetically) so nothing is dropped. Mirrors how
    ``select_reader``/``select_writer`` actually pick a backend, so the report
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
    -- e.g. JAX's ``[cuda:0]`` device list -- are never parsed as style tags.
    """
    line = Text("  ")
    line.append(f"{label:<{width}}", style="bold cyan")
    line.append(str(value))
    console.print(line)


def _probe_torch() -> dict:
    """PyTorch presence + accelerator availability, without raising.

    Torch is a core dependency, but probing CUDA/MPS can fail on a broken
    install, so every query is guarded and missing keys mean "unknown/no".
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


def _probe_jax() -> dict:
    """JAX presence + default backend/devices, without raising."""
    info: dict = {"installed": False}
    try:
        import jax
    except Exception:  # noqa: BLE001
        return info
    info.update(installed=True, version=jax.__version__)
    try:
        info["backend"] = jax.default_backend()
        info["devices"] = [str(d) for d in jax.devices()]
    except Exception:  # noqa: BLE001
        pass
    return info


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Report the installation and what this machine can actually run.

    Covers package version + location, the Python/OS, whether CPU/GPU inference
    is available (torch CUDA/MPS and the JAX backend), the installed video
    read/write backends (flagging GPU/NVDEC decoders), whether the detector
    weights have been downloaded and where, and the default config path. The
    framework imports are lazy and each probe is guarded, so a missing or broken
    optional piece is reported rather than crashing the command.
    """
    import importlib.metadata
    import importlib.util
    import platform

    from . import video
    from .pose2d import backends, download
    from .video.base import CPU_READ_ORDER, GPU_READ_ORDER, WRITE_ORDER

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
    jax_info = _probe_jax()
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
    if jax_info["installed"]:
        be = jax_info.get("backend", "?")
        devices = ", ".join(jax_info.get("devices", [])) or "?"
        _doctor_row(
            "jax", f"{jax_info['version']}  (backend: {be}; devices: {devices})"
        )
    else:
        _doctor_row("jax", "not installed")

    gpu = (
        "cuda" in torch_info
        or torch_info.get("mps")
        or (jax_info.get("backend") not in (None, "cpu"))
    )
    mem = backends.gpu_memory_bytes()
    if gpu:
        _doctor_row(
            "GPU inference",
            f"available ({_fmt_bytes(mem)} memory)" if mem else "available",
        )
    else:
        _doctor_row("GPU inference", "not available -- CPU only")
    detectors = ", ".join(
        f"{b} (default)" if b == backends.DEFAULT_BACKEND else b
        for b in backends.BACKENDS
        if importlib.util.find_spec(b) is not None
    )
    _doctor_row("detectors", detectors or "none")

    _doctor_header("video backends")
    read_avail = video.available_read_backends()
    write_avail = video.available_write_backends()
    read = _by_priority(read_avail, CPU_READ_ORDER, GPU_READ_ORDER)
    gpu_read = [b for b in GPU_READ_ORDER if b in read_avail]
    _doctor_row("read", ", ".join(read) or "none")
    _doctor_row("GPU decoders", ", ".join(gpu_read) or "none (CPU decode only)")
    _doctor_row("write", ", ".join(_by_priority(write_avail, WRITE_ORDER)) or "none")
    missing = sorted(
        set(video.list_read_backends() + video.list_write_backends())
        - set(read_avail)
        - set(write_avail)
    )
    if missing:
        _doctor_row("not installed", ", ".join(missing))

    _doctor_header("weights")
    _doctor_row("cache dir", download.cache_dir())
    for label, path in (
        ("PyTorch", download.torch_weights_path()),
        ("JAX", download.jax_weights_path()),
    ):
        if path.exists():
            state = f"downloaded ({_fmt_bytes(path.stat().st_size)}) -- {path.name}"
        else:
            state = f"not downloaded -- would cache as {path.name}"
        _doctor_row(label, state)

    _doctor_header("config")
    _doctor_row("default config", DEFAULT_CONFIG_PATH)


# -- typer app ---------------------------------------------------------------
#
# The CLI is built with Typer (https://typer.tiangolo.com): typed function
# signatures layered over click, with usage and --help rendered through rich -- so
# the help prints on the same rich stack as the rest of the output. Each command
# declares its options as parameters, configures logging, then hands an
# argparse-style namespace to the matching ``_cmd_*`` worker above; those workers
# stay plain and namespace-driven so they remain equally callable as a library and
# directly from the tests. Constrained options are typed as ``str``-valued Enums
# (Typer's native choice mechanism); the commands pass their ``.value`` on, so the
# workers keep receiving the same plain strings as before.


class LogLevel(str, Enum):
    """``--log-level`` choices, shared by every subcommand (it follows the command,
    e.g. ``deeperfly run REC --log-level debug``). A ``str`` enum, so each member's
    ``.value`` is the name :func:`_configure_logging` expects."""

    debug = "debug"
    info = "info"
    warning = "warning"
    error = "error"
    critical = "critical"


#: The shared ``--log-level`` option. Typer has no click-style shared-option
#: decorator, so it is declared once as a reusable parameter annotation and spread
#: across every command. ``case_sensitive=False`` accepts INFO/Info/info.
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
    and/or wildcard patterns matching several (e.g. 'fly*' -> fly1/, fly2/, ... or
    'data/* data2/*'), each run in turn. Several inputs or a wildcard run as a batch:
    only the valid recordings are kept (non-recording matches are skipped). With
    -r/--recursive, each INPUT is a parent directory and every recording nested under
    it is run in turn (e.g. 'deeperfly run -r data' -> data/fly1/, data/fly2/, ...).

    By default a stage whose result is already in the output dir is reused, so
    re-running a finished recording is a cheap no-op. Pass --overwrite to recompute:
    a bare --overwrite redoes every stage, or name stages to redo only those (e.g.
    '--overwrite pose2d visualization'); recomputing a stage also refreshes the
    stages downstream of it.

    Everything else is set in the config: the do_<stage> toggles choose which
    stages are part of the pipeline, alongside the fps, canvas background and each
    stage's own parameters. With no -c, a run reuses the config.toml already in the
    output dir, else the packaged default.
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

    click options cannot be variadic, so rewrite a bare ``--overwrite`` into
    ``--overwrite <_OVERWRITE_ALL>`` (recompute everything) and ``--overwrite a b``
    into the repeated ``--overwrite a --overwrite b`` that the underlying
    ``multiple=True`` option accepts. Only the known stage names (:data:`STAGES`)
    are consumed after the flag, so the positional recording argument and any later
    options are left untouched; the ``--overwrite=...`` form is passed through as-is.
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

    Runs the Typer app -- via its underlying click command -- in standalone mode so
    usage errors and ``--help`` render through rich, but swallows the
    ``SystemExit(0)`` raised on a clean exit so calling ``main([...])`` as a library
    (and from the tests) returns normally. Real failures -- our own
    ``SystemExit("message")`` and click's non-zero usage exits -- still propagate.

    ``argv`` is normalized first (:func:`_normalize_overwrite_argv`) so ``run``'s
    ``--overwrite`` can take space-separated stage names or stand alone.
    """
    argv = sys.argv[1:] if argv is None else list(argv)
    argv = _normalize_overwrite_argv(argv)
    command = typer.main.get_command(app)
    try:
        command(args=argv, prog_name="deeperfly")
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise
