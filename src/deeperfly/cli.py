"""Command-line interface: ``deeperfly <subcommand>``.

Subcommands are thin wrappers over :mod:`deeperfly.pipeline`, :mod:`deeperfly.io`,
:mod:`deeperfly.video` and :mod:`deeperfly.pose2d`, so everything is equally
usable as a library. Everything a run needs lives in one merged config TOML
(``deeperfly init`` writes a default to edit): the camera rig, the input
filename->camera map, the 2D detector, the pipeline options, bundle adjustment
and the skeleton. The commands:

- ``init`` -- write a default config.toml.
- ``run`` -- the whole pipeline (``detect`` 2D -> ``pose3d`` calibrate+triangulate
  -> ``visualize``) or any prefix of it. The recording is the positional argument;
  ``-c``/``--config`` is the merged config TOML (defaults to the packaged default
  config when omitted) and ``-o`` is an output *directory* (default
  ``<input>/deeperfly_outputs``) that collects the result ``poses.h5``, the videos
  and a copy of the config. Each run reuses whatever is already cached in that
  directory and computes only what is missing; ``--overwrite`` recomputes
  everything and ``--until`` stops early. Detector weights are downloaded and
  converted automatically on first use.
- ``inspect`` -- print a summary of a result file.
- ``doctor`` -- print installation/runtime details: package version, whether
  CPU/GPU inference is available, the installed video backends, whether the
  detector weights have been downloaded (and where), and the default config path.

The 2D->3D pipeline is a linear sequence of three stages; ``run`` resumes from the
furthest-along artifact cached in the output directory, so partial work is never
recomputed.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import tomllib
from pathlib import Path

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
    before ``import .geometry`` (which probes ``jax.default_backend()`` at import
    and would lock in the memory policy), and uses an import-free GPU probe so it
    does not pull in or initialize torch/JAX. Only for ``run``; honors an override.
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
from .pipeline import run_from_points2d  # noqa: E402

#: Packaged template emitted by ``deeperfly init`` (also the run-config example).
DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "default_config.toml"

#: The linear pipeline stages, in order. ``run`` executes a contiguous range.
STAGES = ("detect", "pose3d", "visualize")

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


def _load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _configure_logging(verbose: int, quiet: bool) -> None:
    """Map ``-v``/``-q`` onto a root log level (default ``WARNING``).

    Records render through rich's :class:`~rich.logging.RichHandler` (colored level
    column, messages wrapped to the terminal) on the same stderr console as the
    progress bar, so log lines and the bar never overwrite each other.
    """
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
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
    # unless we're at -vv (debug), where seeing every backend probe is useful.
    if verbose < 2:
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


def _calibrate_kwargs(config: dict) -> dict:
    """Bundle-adjustment options for :func:`deeperfly.pipeline.calibrate`.

    Reads ``[bundle_adjustment]``: ``keypoints`` (-> ``ba_keypoints``), ``fixed``,
    ``shared`` and the solver sub-table (e.g.
    ``[bundle_adjustment.least_squares_scipy]``, forwarded as solver kwargs like
    ``max_nfev`` / ``loss``). Anything omitted falls through to ``calibrate``'s
    own defaults.
    """
    ba = config.get("bundle_adjustment", {})
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


def _run_kwargs(config: dict) -> dict:
    """Keyword arguments for :func:`deeperfly.pipeline.run_from_points2d`.

    Built entirely from the config's ``[pipeline]`` / ``[bundle_adjustment]``
    sections; an empty config yields the library defaults (merge stripes on,
    calibrate on, legs-only BA, reproject, no smoothing).
    """
    pipe = config.get("pipeline", {})
    ps = pipe.get("pictorial", {})
    smooth = pipe.get("smooth") or None
    if isinstance(smooth, str) and smooth.lower() == "none":
        smooth = None
    return dict(
        merge_stripes=pipe.get("merge_stripes", True),
        do_calibrate=pipe.get("calibrate", True),
        calibrate_kwargs=_calibrate_kwargs(config),
        correct=pipe.get("correct", "reproject"),
        ps_kwargs={"temporal": ps.get("temporal", False), "lam": ps.get("lam", 1.0)},
        smooth=smooth,
        fps=pipe.get("fps", 100.0),
    )


# -- input -> camera frame resolution ----------------------------------------


def _camera_source(root: str | Path, prefix: str) -> Path | str:
    """Locate a camera's frames under ``root`` given its filename ``prefix``.

    Tries, in order, a video file ``<prefix>.<ext>``, a subdirectory
    ``<prefix>/`` of images, then the image sequence glob ``<prefix>*`` (e.g.
    ``camera_0_img_000123.jpg``). Returns a path/glob ready for
    :func:`deeperfly.video.read_frames`; raises ``SystemExit`` if nothing matches.
    """
    from .video.io import _VIDEO_EXTS

    root = Path(root)
    for ext in _VIDEO_EXTS:
        cand = root / f"{prefix}{ext}"
        if cand.exists():
            return cand
    subdir = root / prefix
    if subdir.is_dir():
        return subdir
    if sorted(glob.glob(str(root / f"{prefix}*"))):
        return str(root / f"{prefix}*")
    raise SystemExit(f"no video or images for camera {prefix!r} under {root}")


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
    det = config.get("detector", {})
    device = det.get("decode_device", "cpu")
    if device == "cpu" or det.get("backend", "jax") != "jax":
        return "cpu"
    return device  # "cuda"/"gpu" / "auto": opt-in on-device decode for the JAX backend


def _camera_sources(input_dir: str | Path, config: dict) -> list[tuple[str, object]]:
    """``(name, source)`` per camera (in camera order), mapped via ``[inputs]``."""
    root = Path(input_dir)
    inputs = config.get("inputs", {})
    return [
        (name, _camera_source(root, inputs.get(name, name)))
        for name in config.get("cameras", {})
    ]


def _camera_image_sizes(args, config: dict) -> dict[str, tuple[int, int]]:
    """``name -> (height, width)`` from a single frame per camera.

    Used to infer each view's principal point. Reads only frame 0 (host), so it is
    cheap and independent of the full streaming decode.
    """
    from . import video

    backend = config.get("detector", {}).get("video_backend", "auto")
    device = _frame_read_device(config)  # match detection (CPU by default)
    sizes: dict[str, tuple[int, int]] = {}
    for name, src in _camera_sources(args.input, config):
        head = video.read_frames(src, backend=backend, device=device, indices=[0])
        sizes[name] = tuple(int(d) for d in head.shape[1:3])
    return sizes


def _prefetch_windows(sources, *, backend, device, chunk, depth=1):
    """Yield ``(window, n)`` decoded frame windows, decoding ``depth`` ahead.

    A background producer decodes the next window while the consumer detects the
    current one, so decode overlaps the GPU forward instead of running before it.
    The win is largest for CPU decode (it runs on otherwise-idle cores fully in
    parallel with the GPU); GPU decode shares the device, so it overlaps less.

    EOF semantics match the old serial loop: a short or empty window ends the
    stream, and a read failure on the *very first* window propagates (anything
    later is treated as "past the end").
    """
    import queue
    import threading

    from . import video

    q: queue.Queue = queue.Queue(maxsize=depth)
    DONE = object()

    def produce():
        done = 0
        while True:
            try:
                window = [
                    video.read_frames(
                        s, backend=backend, device=device, start=done, stop=done + chunk
                    )
                    for s in sources
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


def _detect_2d(args, config: dict, model, sides, flips, *, correct, k, quiet):
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

    det = config.get("detector", {})
    backend = det.get("video_backend", "auto")
    device = _frame_read_device(config)
    chunk = max(1, int(det.get("chunk_frames", DEFAULT_CHUNK_FRAMES)))
    sources = [src for _, src in _camera_sources(args.input, config)]
    total = video.count_frames(sources[0]) if sources else 0
    # Size the forward to the GPU so each window is a few big batches, not one
    # batch per frame -- this only changes dispatch granularity, not the result.
    batch_size = auto_batch_size(inference.IMG_SIZE)

    # One-line summary instead of a per-camera/per-window read log (that spam is at
    # -vv now; see deeperfly.video.io). select_reader resolves "auto" to the real
    # backend; guard since a forced GPU backend may be uninstalled.
    try:
        reader_name = video.select_reader(backend, device=device).name
    except Exception:  # noqa: BLE001
        reader_name = backend
    log.info(
        "reading frames via '%s' backend: %d/read per camera (device=%s), forward batch %d",
        reader_name,
        chunk,
        device,
        batch_size,
    )

    pts_parts, conf_parts, cand_xy, cand_score = [], [], [], []
    bar = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("frames"),
        _FPSColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=err_console,
        # mirror tqdm's auto-disable: off under -q and when stderr isn't a TTY.
        disable=quiet or not err_console.is_terminal,
    )

    with bar:
        task = bar.add_task("detect 2D", total=total)

        def progress(rng):  # advance the single bar once per completed frame
            for t in rng:
                yield t
                bar.advance(task)

        for window, _ in _prefetch_windows(
            sources, backend=backend, device=device, chunk=chunk
        ):
            if correct == "pictorial":
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


def _resolve_stages(
    args: argparse.Namespace, *, have_2d: bool, have_3d: bool
) -> tuple[int, int]:
    """Inclusive ``(start, stop)`` indices into :data:`STAGES`.

    The start is inferred from what is already cached in the output directory:
    nothing cached starts at ``detect``; a cached 2D-only result resumes at
    ``pose3d``; a cached result that already has 3D resumes at ``visualize``.
    ``--overwrite`` ignores the cache and starts at ``detect``. ``--until`` caps
    the stop; when everything requested is already cached (``stop < start``) the
    caller treats the run as a no-op.
    """
    if args.overwrite or not have_2d:
        start = STAGES.index("detect")
    elif have_3d:
        start = STAGES.index("visualize")
    else:
        start = STAGES.index("pose3d")
    stop = STAGES.index(args.until) if args.until else len(STAGES) - 1
    return start, stop


def _stage_in_range(stage: str, start: int, stop: int) -> bool:
    return start <= STAGES.index(stage) <= stop


# -- pipeline stages ---------------------------------------------------------


def _stage_detect(
    args: argparse.Namespace, config: dict
) -> tuple[PoseResult, object | None, None, dict]:
    """Run 2D detection -> a 2D-only :class:`PoseResult` + in-memory artifacts.

    Returns ``(result, candidates, None, image_sizes)``. ``candidates`` is the
    top-K peak set (only when ``correct = "pictorial"``). Frames are **not** held
    in memory (detection streams them in windows -- see :func:`_detect_2d`), so the
    third slot is ``None``; the recording path is recorded in ``result.meta`` and a
    visualize stage re-sources the one overlay camera it needs.
    """
    from .pose2d import backends, inference
    from .skeleton import Skeleton

    image_sizes = _camera_image_sizes(args, config)
    log.info(
        "input image sizes (h, w): %s", {n: tuple(s) for n, s in image_sizes.items()}
    )
    cameras = CameraGroup.from_config(config, image_sizes=image_sizes)
    skeleton = Skeleton.from_config(config) if "skeleton" in config else Skeleton.fly()

    det = config.get("detector", {})
    backend = det.get("backend", "jax")
    log.info(
        "loading %s detector (checkpoint: %s)",
        backend,
        det.get("checkpoint") or "cached",
    )
    model = _load_detector(det.get("checkpoint"), backend)
    log.info("%s detector ready on device %s", backend, backends.detector_device(model))

    pipe = config.get("pipeline", {})
    correct = pipe.get("correct", "reproject")
    k = pipe.get("pictorial", {}).get("k", 5)
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
        args, config, model, sides, flips, correct=correct, k=k, quiet=args.quiet
    )

    result = PoseResult(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=conf,
        meta={"input": str(Path(args.input).resolve())},
    )
    return result, candidates, None, image_sizes


def _stage_pose3d(
    args: argparse.Namespace,
    config: dict,
    result: PoseResult,
    candidates: object | None,
) -> PoseResult:
    """Calibrate + triangulate + correct + smooth -> a full :class:`PoseResult`.

    Uses ``result.cameras`` -- the rig built from the config with image sizes
    during ``detect``, or the (possibly already-calibrated) rig stored in a
    resumed result. To triangulate against an edited rig, change the config and
    re-run from ``detect`` (frame sizes are needed to place the principal point).

    ``candidates`` is only present when ``detect`` just ran in this process;
    resuming from a stored 2D result has none, so a ``pictorial`` config falls
    back to ``reproject`` with a warning.
    """
    cameras = result.cameras

    kwargs = _run_kwargs(config)
    if kwargs["correct"] == "pictorial" and candidates is None:
        log.warning(
            "pictorial correction needs detector candidates, which a cached 2D "
            "result does not store; falling back to 'reproject'. Re-run with "
            "--overwrite (re-running detect from the recording) to use pictorial."
        )
        kwargs["correct"] = "reproject"

    carry = {"input": result.meta["input"]} if "input" in (result.meta or {}) else {}
    log.info(
        "pose3d: calibrate=%s merge_stripes=%s correct=%s smooth=%s fps=%s "
        "(%d frames, %d views)",
        kwargs["do_calibrate"],
        kwargs["merge_stripes"],
        kwargs["correct"],
        kwargs["smooth"],
        kwargs["fps"],
        result.n_frames,
        result.n_views,
    )
    return run_from_points2d(
        cameras,
        result.skeleton,
        result.pts2d,
        result.conf,
        candidates=candidates,
        meta=carry,
        **kwargs,
    )


def _overlay_frames(
    args: argparse.Namespace, config: dict, result: PoseResult, camera: int
) -> np.ndarray:
    """Load one camera's frames for a 2D overlay when resuming (no frames in hand).

    Source order: ``--recording`` -> the recording path stored in ``result.meta``
    (if it still exists) -> the run's own input recording -> error pointing at
    ``--recording``.
    """
    from . import video

    inp = getattr(args, "input", None)
    if args.recording:
        root = args.recording
    elif (stored := (result.meta or {}).get("input")) and Path(stored).exists():
        root = stored
        log.info("sourcing overlay frames from recorded input %s", root)
    elif inp and Path(inp).exists():
        root = inp
        log.info("sourcing overlay frames from -i input %s", root)
    else:
        raise SystemExit(
            "2D overlay needs the original frames, but none are in memory and the "
            "recording recorded in the result is unavailable. Pass --recording <dir>."
        )
    name = result.cameras.names[camera]
    inputs = config.get("inputs", {})
    backend = config.get("detector", {}).get("video_backend", "auto")
    src = _camera_source(root, inputs.get(name, name))
    return video.read_frames(src, backend=backend)


def _stage_visualize(
    args: argparse.Namespace,
    config: dict,
    result: PoseResult,
    frames: list | None,
    outdir: Path,
) -> None:
    """Render the 3D skeleton MP4 (and, with ``--overlay-camera``, a 2D overlay).

    Both land in ``outdir`` under fixed names; an existing MP4 is kept (skipped)
    unless ``--overwrite`` is set, and an overlay's frames are sourced only when
    the overlay actually needs rendering.
    """
    from . import video

    pipe = config.get("pipeline", {})
    fps = args.fps if args.fps is not None else pipe.get("fps", 30.0)
    video_path = outdir / "pose3d.mp4"
    if video_path.exists() and not args.overwrite:
        log.info("3D video already present, skipping: %s", video_path)
    else:
        log.info("rendering 3D pose video -> %s", video_path)
        video.render_pose3d_video(result, video_path, fps=fps, background=args.bg)
        log.info("wrote %s", video_path)

    if args.overlay_camera is not None:
        cam = args.overlay_camera
        overlay_path = outdir / f"pose3d_overlay_cam{cam}.mp4"
        if overlay_path.exists() and not args.overwrite:
            log.info("2D overlay already present, skipping: %s", overlay_path)
            return
        cam_frames = (
            frames[cam]
            if frames is not None
            else _overlay_frames(args, config, result, cam)
        )
        log.info("rendering 2D overlay (camera %d) -> %s", cam, overlay_path)
        video.render_overlay_video(
            result, cam_frames, overlay_path, camera=cam, fps=fps, background=args.bg
        )
        log.info("wrote %s", overlay_path)


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


def _save_config_snapshot(
    config_path: Path, outdir: Path, *, reused_cache: bool
) -> None:
    """Copy the run config into ``outdir`` for reproducibility.

    If a snapshot from a previous run is already there and differs while we are
    reusing that run's cached results, warn: those artifacts were produced with a
    different config (pass ``--overwrite`` to recompute from scratch). The new
    config then replaces the snapshot.
    """
    src = config_path.read_text()
    dst = outdir / "config.toml"
    if reused_cache and dst.exists() and dst.read_text() != src:
        log.warning(
            "config %s differs from the one that produced the cached results in "
            "%s; reusing the cache anyway. Pass --overwrite to recompute.",
            config_path,
            outdir,
        )
    dst.write_text(src)


def _cmd_run(args: argparse.Namespace) -> None:
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    config = _load_config(config_path)
    outdir = Path(args.output) if args.output else _default_outdir(args.input)
    outdir.mkdir(parents=True, exist_ok=True)
    log.info("output directory: %s", outdir)
    h5_path = outdir / "poses.h5"

    cached = h5_path.exists() and not args.overwrite
    result = PoseResult.load(h5_path) if cached else None
    have_2d = result is not None
    have_3d = result is not None and result.pts3d is not None

    start, stop = _resolve_stages(args, have_2d=have_2d, have_3d=have_3d)
    if stop < start:
        log.info(
            "all requested stages already cached in %s; pass --overwrite to recompute",
            outdir,
        )
        return
    log.info("running stages %s..%s", STAGES[start], STAGES[stop])
    _save_config_snapshot(config_path, outdir, reused_cache=cached)

    frames = candidates = None
    if _stage_in_range("detect", start, stop):
        result, candidates, frames, _ = _stage_detect(args, config)
    if _stage_in_range("pose3d", start, stop):
        result = _stage_pose3d(args, config, result, candidates)
    if _stage_in_range("detect", start, stop) or _stage_in_range("pose3d", start, stop):
        result.save(h5_path)
        log.info(
            "wrote %s  (%d frames, %d views)",
            h5_path,
            result.n_frames,
            result.n_views,
        )
    if _stage_in_range("visualize", start, stop):
        _stage_visualize(args, config, result, frames, outdir)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deeperfly",
        description=(
            "Markerless 3D pose estimation of tethered Drosophila from a "
            "multi-camera rig. 'deeperfly init' writes a config to edit; "
            "'deeperfly run' detects 2D pose, reconstructs 3D and renders a "
            "video; 'deeperfly inspect' summarizes a result file; "
            "'deeperfly doctor' reports the installation/runtime."
        ),
    )

    common = argparse.ArgumentParser(add_help=False)
    g = common.add_mutually_exclusive_group()
    g.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="more logging (-v info, -vv debug)",
    )
    g.add_argument(
        "-q", "--quiet", action="store_true", help="only errors; hide progress bars"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    pini = sub.add_parser(
        "init", parents=[common], help="write a default config.toml to edit"
    )
    pini.add_argument(
        "output",
        nargs="?",
        default="config.toml",
        help="destination (default config.toml)",
    )
    pini.add_argument("--force", action="store_true", help="overwrite an existing file")
    pini.set_defaults(func=_cmd_init)

    pr = sub.add_parser(
        "run",
        parents=[common],
        help="detect 2D -> 3D -> visualize, or a prefix of it (resumes from cache)",
    )
    pr.add_argument(
        "input",
        help="recording dir/glob (per-camera videos or image folders)",
    )
    pr.add_argument(
        "-c",
        "--config",
        default=None,
        help="merged config TOML (from 'deeperfly init'); "
        "defaults to the packaged default config",
    )
    pr.add_argument(
        "-o",
        "--output-dir",
        dest="output",
        help="output directory (default: <input>/deeperfly_outputs; created if missing)",
    )
    pr.add_argument(
        "--until",
        choices=STAGES,
        help="stop after this stage (default: run through visualize)",
    )
    pr.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute and re-render everything, ignoring cached outputs",
    )
    pr.add_argument(
        "--overlay-camera",
        dest="overlay_camera",
        type=int,
        default=None,
        help="also render a 2D overlay for this camera index",
    )
    pr.add_argument(
        "--recording", help="per-camera frames for the 2D overlay when resuming"
    )
    pr.add_argument(
        "--fps", type=float, default=None, help="video fps (default from config)"
    )
    pr.add_argument("--bg", choices=["white", "black"], default="white")
    pr.set_defaults(func=_cmd_run)

    pi = sub.add_parser(
        "inspect", parents=[common], help="print a summary of a result file"
    )
    pi.add_argument("input", help="path to a result .h5 file")
    pi.set_defaults(func=_cmd_inspect)

    pd = sub.add_parser(
        "doctor",
        parents=[common],
        help="report installation/runtime: accelerators, video backends, weights",
    )
    pd.set_defaults(func=_cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", 0), getattr(args, "quiet", False))
    args.func(args)
