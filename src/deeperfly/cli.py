"""Command-line interface: ``deeperfly <subcommand>``.

Subcommands are thin wrappers over :mod:`deeperfly.pipeline`, :mod:`deeperfly.io`,
:mod:`deeperfly.video` and :mod:`deeperfly.pose2d`, so everything is equally
usable as a library. Everything a run needs lives in one merged config TOML
(``deeperfly init`` writes a default to edit): the camera rig, the input
filename->camera map, the 2D detector, the pipeline options, bundle adjustment
and the skeleton. The commands:

- ``init`` -- write a default config.toml.
- ``run`` -- detect 2D -> calibrate -> 3D from images/video
  (``run cfg.toml -i <recording> -o <out.h5>``).
- ``pose3d`` -- triangulate + correct an existing 2D result.
- ``visualize`` / ``info`` -- render or summarize a result.
- ``download-weights`` / ``convert-weights`` -- fetch / convert detector weights.
"""

from __future__ import annotations

import argparse
import glob
import tomllib
from pathlib import Path

import numpy as np

from .cameras import CameraGroup
from .io import PoseResult
from .pipeline import run_from_points2d

#: Packaged template emitted by ``deeperfly init`` (also the run-config example).
DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "default_config.toml"


def _load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_detector(checkpoint: str | None, backend: str):
    """Load the JAX detector (native .eqx) or the PyTorch detector (.tar)."""
    from .pose2d import backends

    if backend == "torch":
        from .pose2d.download import download_torch_weights

        path = checkpoint or download_torch_weights()
        return backends.load_detector("torch", path)

    from .pose2d.download import jax_weights_path

    path = checkpoint or jax_weights_path()
    if not Path(path).exists():
        raise SystemExit(
            f"no JAX checkpoint at {path}. Run 'deeperfly download-weights' and "
            "'deeperfly convert-weights' first, set [detector].checkpoint, or use "
            "[detector].backend = 'torch'."
        )
    return backends.load_detector("jax", path)


# -- config-driven option resolution -----------------------------------------


def _calibrate_kwargs(config: dict) -> dict:
    """Bundle-adjustment options for :func:`deeperfly.pipeline.calibrate`.

    Reads ``[bundle_adjustment]``: ``keypoints`` (-> ``ba_keypoints``), ``fixed``,
    ``shared`` and the solver sub-table (e.g.
    ``[bundle_adjustment.scipy.least_squares]``, forwarded as solver kwargs like
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
    for part in ba.get("solver", "scipy.least_squares").split("."):
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


# -- subcommands -------------------------------------------------------------


def _cmd_init(args: argparse.Namespace) -> None:
    dst = Path(args.output)
    if dst.exists() and not args.force:
        raise SystemExit(f"{dst} already exists (pass --force to overwrite)")
    dst.write_text(DEFAULT_CONFIG_PATH.read_text())
    print(f"wrote {dst}")
    print(
        "next: edit [inputs]/[cameras] to match your rig, then "
        f"'deeperfly run {dst} -i <recording> -o <out.h5>'"
    )


def _cmd_run(args: argparse.Namespace) -> None:
    from .pose2d import inference
    from .skeleton import Skeleton

    config = _load_config(args.config)

    # Read frames first so any camera without an explicit principal point can
    # fall back to its view's image center.
    frames, image_sizes = _read_camera_frames(args.input, config)
    cameras = CameraGroup.from_config(config, image_sizes=image_sizes)
    skeleton = Skeleton.from_config(config) if "skeleton" in config else Skeleton.fly()

    det = config.get("detector", {})
    model = _load_detector(det.get("checkpoint"), det.get("backend", "jax"))

    pipe = config.get("pipeline", {})
    correct = pipe.get("correct", "reproject")
    sides, flips = inference.fly_camera_layout(cameras.names)
    candidates = None
    if correct == "pictorial":
        k = pipe.get("pictorial", {}).get("k", 5)
        pts2d, conf, candidates = inference.detect_candidates_sequence(
            model, frames, sides, flips, k=k
        )
    else:
        pts2d, conf = inference.detect_sequence(model, frames, sides, flips)

    result = run_from_points2d(
        cameras, skeleton, pts2d, conf, candidates=candidates, **_run_kwargs(config)
    )
    result.save(args.output)
    print(f"wrote {args.output}  ({result.n_frames} frames, {result.n_views} views)")


def _cmd_download_weights(args: argparse.Namespace) -> None:
    from .pose2d.download import download_torch_weights

    path = download_torch_weights(force=args.force)
    print(f"downloaded PyTorch weights to {path}")
    print("next: deeperfly convert-weights")


def _cmd_convert_weights(args: argparse.Namespace) -> None:
    import jax

    from .pose2d.backends import infer_num_stacks
    from .pose2d.backends.jax import (
        HourglassNet,
        convert_state_dict,
        save_checkpoint,
    )
    from .pose2d.backends.torch import state_dict_from_torch_checkpoint
    from .pose2d.download import download_torch_weights, jax_weights_path

    src = args.pth or download_torch_weights()
    sd = state_dict_from_torch_checkpoint(src)
    num_stacks = infer_num_stacks(sd)
    skeleton = HourglassNet.deepfly2d(key=jax.random.PRNGKey(0), num_stacks=num_stacks)
    model = convert_state_dict(sd, skeleton)
    out = args.output or jax_weights_path()
    save_checkpoint(model, out)
    print(f"converted {src} -> {out}  ({num_stacks} stacks)")


def _cmd_pose3d(args: argparse.Namespace) -> None:
    result = PoseResult.load(args.input)
    cameras = result.cameras
    config = _load_config(args.config) if args.config else {}
    if "cameras" in config:  # a config with cameras overrides the stored rig
        cameras = CameraGroup.from_config(config)
    kwargs = _run_kwargs(config)
    kwargs["correct"] = "reproject"  # no stored candidates -> reproject only
    out = run_from_points2d(
        cameras, result.skeleton, result.pts2d, result.conf, **kwargs
    )
    out.save(args.output)
    print(f"wrote {args.output}  ({out.n_frames} frames, {out.n_views} views)")


def _cmd_visualize(args: argparse.Namespace) -> None:
    from . import video  # imports matplotlib/imageio lazily

    result = PoseResult.load(args.input)
    if args.mode == "3d":
        video.render_pose3d_video(result, args.output, fps=args.fps, background=args.bg)
    else:
        if args.images is None:
            raise SystemExit("2d overlay needs --images <video|dir>")
        frames = video.read_frames(args.images)
        video.render_overlay_video(
            result,
            frames,
            args.output,
            camera=args.camera,
            fps=args.fps,
            background=args.bg,
        )
    print(f"wrote {args.output}")


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
    sub = parser.add_subparsers(dest="command", required=True)

    pini = sub.add_parser("init", help="write a default config.toml to edit")
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
        help="full pipeline: detect 2D -> calibrate -> 3D (options in the config)",
    )
    pr.add_argument("config", help="merged config TOML (from 'deeperfly init')")
    pr.add_argument(
        "-i",
        "--in",
        dest="input",
        required=True,
        help="dir of per-camera videos / image sequences (see the config's [inputs])",
    )
    pr.add_argument("-o", "--out", dest="output", required=True, help="output .h5")
    pr.set_defaults(func=_cmd_run)

    pdl = sub.add_parser("download-weights", help="fetch the original PyTorch weights")
    pdl.add_argument("--force", action="store_true")
    pdl.set_defaults(func=_cmd_download_weights)

    pcw = sub.add_parser(
        "convert-weights", help="convert PyTorch weights to a JAX checkpoint"
    )
    pcw.add_argument("--pth", help="source .tar (downloads if omitted)")
    pcw.add_argument("--out", dest="output", help="destination .eqx (cache default)")
    pcw.set_defaults(func=_cmd_convert_weights)

    p3 = sub.add_parser("pose3d", help="triangulate + correct an existing 2D result")
    p3.add_argument("-i", "--in", dest="input", required=True)
    p3.add_argument("-o", "--out", dest="output", required=True)
    p3.add_argument(
        "config",
        nargs="?",
        help="optional config TOML (overrides stored cameras + sets pipeline options)",
    )
    p3.set_defaults(func=_cmd_pose3d)

    pv = sub.add_parser("visualize", help="render a 2D overlay or 3D skeleton MP4")
    pv.add_argument("--in", dest="input", required=True)
    pv.add_argument("--out", dest="output", required=True)
    pv.add_argument("--mode", choices=["2d", "3d"], default="3d")
    pv.add_argument("--images", help="video/dir of frames (2d mode)")
    pv.add_argument("--camera", type=int, default=0)
    pv.add_argument("--fps", type=float, default=30.0)
    pv.add_argument("--bg", choices=["white", "black"], default="white")
    pv.set_defaults(func=_cmd_visualize)

    pi = sub.add_parser("info", help="print a summary of a result file")
    pi.add_argument("--in", dest="input", required=True)
    pi.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
