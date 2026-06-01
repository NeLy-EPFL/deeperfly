"""Command-line interface: ``deeperfly <subcommand>``.

Subcommands are thin wrappers over :mod:`deeperfly.pipeline`, :mod:`deeperfly.io`,
:mod:`deeperfly.video` and :mod:`deeperfly.pose2d`, so everything is equally
usable as a library. Everything a run needs lives in one merged config TOML
(``deeperfly init`` writes a default to edit): the camera rig, the input
filename->camera map, the 2D detector, the pipeline options, bundle adjustment
and the skeleton. The commands:

- ``init`` -- write a default config.toml.
- ``run`` -- the whole pipeline (``detect`` 2D -> ``pose3d`` calibrate+triangulate
  -> ``visualize``) or any prefix of it. ``-i`` takes a recording; ``-o`` is an
  output *directory* (default ``<input>/deeperfly_outputs``) that collects the
  result ``poses.h5``, the videos and a copy of the config. Each run reuses
  whatever is already cached in that directory and computes only what is missing;
  ``--overwrite`` recomputes everything and ``--until`` stops early. Detector
  weights are downloaded and converted automatically on first use.
- ``info`` -- print a summary of a result file.

The 2D->3D pipeline is a linear sequence of three stages; ``run`` resumes from the
furthest-along artifact cached in the output directory, so partial work is never
recomputed.
"""

from __future__ import annotations

import argparse
import functools
import glob
import logging
import sys
import tomllib
from pathlib import Path

import numpy as np

from .cameras import CameraGroup
from .io import PoseResult
from .pipeline import run_from_points2d

#: Packaged template emitted by ``deeperfly init`` (also the run-config example).
DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "default_config.toml"

#: The linear pipeline stages, in order. ``run`` executes a contiguous range.
STAGES = ("detect", "pose3d", "visualize")

log = logging.getLogger("deeperfly")


def _load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _configure_logging(verbose: int, quiet: bool) -> None:
    """Map ``-v``/``-q`` onto a root log level (default ``WARNING``)."""
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stderr)
    log.setLevel(level)


def _load_detector(checkpoint: str | None, backend: str):
    """Load the JAX detector (native .eqx) or the PyTorch detector (.tar).

    With no explicit ``checkpoint`` the cached weights are used, provisioning them
    on demand: the PyTorch ``.tar`` is downloaded, and for the JAX backend it is
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


def _read_camera_frames(
    input_dir: str | Path, config: dict
) -> tuple[list, dict[str, tuple[int, int]]]:
    """Read per-camera frames mapped by ``[inputs]`` (default: prefix == name).

    Returns the per-camera frame stacks (in camera order) and a
    ``name -> (height, width)`` map used to infer each view's principal point.
    """
    from . import video

    root = Path(input_dir)
    backend = config.get("detector", {}).get("video_backend", "auto")
    inputs = config.get("inputs", {})
    frames, image_sizes = [], {}
    for name in config.get("cameras", {}):
        src = _camera_source(root, inputs.get(name, name))
        view = video.read_frames(src, backend=backend)
        frames.append(view)
        image_sizes[name] = tuple(view.shape[1:3])  # (height, width)
    return frames, image_sizes


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


def _require_viz() -> None:
    """Fail fast if the visualize stage can't import its rendering deps."""
    try:
        import imageio  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "visualization needs the 'viz' extra (matplotlib + imageio). Install it "
            "(e.g. pip install 'deeperfly[viz]') or stop earlier with --until pose3d."
        ) from e


def _make_progress(quiet: bool, desc: str):
    """A ``tqdm``-backed iterator wrapper for the detector loop (off when quiet)."""
    from tqdm import tqdm

    return functools.partial(tqdm, desc=desc, unit="frame", disable=quiet or None)


# -- pipeline stages ---------------------------------------------------------


def _stage_detect(
    args: argparse.Namespace, config: dict
) -> tuple[PoseResult, object | None, list, dict]:
    """Run 2D detection -> a 2D-only :class:`PoseResult` + in-memory artifacts.

    Returns ``(result, candidates, frames, image_sizes)``. ``candidates`` is the
    top-K peak set (only when ``correct = "pictorial"``); ``frames`` are reused by
    a later visualize stage. The recording path is recorded in ``result.meta`` so
    a future resume can recover the frames for visualization.
    """
    from .pose2d import inference
    from .skeleton import Skeleton

    frames, image_sizes = _read_camera_frames(args.input, config)
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

    pipe = config.get("pipeline", {})
    correct = pipe.get("correct", "reproject")
    sides, flips = inference.fly_camera_layout(cameras.names)
    n_passes = len(inference.expand_passes(sides, flips)[0])
    n_frames = len(frames[0]) if frames else 0
    log.info(
        "detecting 2D poses: %d frames x %d views, %d forward passes/frame, "
        "network input %dx%d",
        n_frames,
        len(frames),
        n_passes,
        inference.IMG_SIZE[0],
        inference.IMG_SIZE[1],
    )
    progress = _make_progress(args.quiet, "detect 2D")
    candidates = None
    if correct == "pictorial":
        k = pipe.get("pictorial", {}).get("k", 5)
        pts2d, conf, candidates = inference.detect_candidates_sequence(
            model, frames, sides, flips, k=k, progress=progress
        )
    else:
        pts2d, conf = inference.detect_sequence(
            model, frames, sides, flips, progress=progress
        )

    result = PoseResult(
        cameras=cameras,
        skeleton=skeleton,
        pts2d=pts2d,
        conf=conf,
        meta={"input": str(Path(args.input).resolve())},
    )
    return result, candidates, frames, image_sizes


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
    (if it still exists) -> the run's own ``-i`` input -> error pointing at
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
    print(f"wrote {dst}")
    print(
        "next: edit [inputs]/[cameras] to match your rig, then "
        f"'deeperfly run {dst} -i <recording>' "
        "(outputs land in <recording>/deeperfly_outputs/; override with -o <dir>)"
    )


def _save_config_snapshot(
    args: argparse.Namespace, outdir: Path, *, reused_cache: bool
) -> None:
    """Copy the run config into ``outdir`` for reproducibility.

    If a snapshot from a previous run is already there and differs while we are
    reusing that run's cached results, warn: those artifacts were produced with a
    different config (pass ``--overwrite`` to recompute from scratch). The new
    config then replaces the snapshot.
    """
    src = Path(args.config).read_text()
    dst = outdir / "config.toml"
    if reused_cache and dst.exists() and dst.read_text() != src:
        log.warning(
            "config %s differs from the one that produced the cached results in "
            "%s; reusing the cache anyway. Pass --overwrite to recompute.",
            args.config,
            outdir,
        )
    dst.write_text(src)


def _cmd_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
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

    # Fail fast before the expensive detect if we can't render at the end.
    if _stage_in_range("visualize", start, stop):
        _require_viz()
    _save_config_snapshot(args, outdir, reused_cache=cached)

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


def _cmd_info(args: argparse.Namespace) -> None:
    result = PoseResult.load(args.input)
    print(f"file:     {args.input}")
    print(f"views:    {result.n_views}  {result.cameras.names}")
    print(f"frames:   {result.n_frames}")
    print(f"skeleton: {result.skeleton.name}  ({result.skeleton.n_points} points)")
    print(f"has 3D:   {result.pts3d is not None}")
    if result.reproj_error is not None:
        print(
            f"reproj:   median {np.nanmedian(result.reproj_error):.3f} px"
            f"  max {np.nanmax(result.reproj_error):.3f} px"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deeperfly", description=__doc__)

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
        help="detect 2D -> 3D -> visualize, or a prefix of it (resumes from -i)",
    )
    pr.add_argument("config", help="merged config TOML (from 'deeperfly init')")
    pr.add_argument(
        "-i",
        "--input-dir",
        dest="input",
        required=True,
        help="recording dir/glob",
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
        "info", parents=[common], help="print a summary of a result file"
    )
    pi.add_argument("--in", dest="input", required=True)
    pi.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _configure_logging(getattr(args, "verbose", 0), getattr(args, "quiet", False))
    args.func(args)
