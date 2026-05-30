"""Command-line interface: ``deeperfly <subcommand>``.

Subcommands are thin wrappers over :mod:`deeperfly.pipeline`, :mod:`deeperfly.io`,
:mod:`deeperfly.video` and :mod:`deeperfly.pose2d`, so everything is equally
usable as a library: ``run`` (detect 2D -> calibrate -> 3D from images/video),
``pose3d`` (triangulate + correct an existing 2D result), ``visualize``,
``info``, and the weight helpers ``download-weights`` / ``convert-weights``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .cameras import CameraGroup
from .io import PoseResult
from .pipeline import run_from_points2d


def _load_detector(checkpoint: str | None, backend: str):
    """Load the JAX detector (native .eqx) or the PyTorch detector (.tar)."""
    if backend == "torch":
        from .pose2d import torch_backend
        from .pose2d.download import download_torch_weights

        path = checkpoint or download_torch_weights()
        return torch_backend.load_model(path)

    import jax

    from .pose2d.download import jax_weights_path
    from .pose2d.weights import load_checkpoint

    path = checkpoint or jax_weights_path()
    if not Path(path).exists():
        raise SystemExit(
            f"no JAX checkpoint at {path}. Run 'deeperfly download-weights' and "
            "'deeperfly convert-weights' first, pass --checkpoint, or use --backend torch."
        )
    return load_checkpoint(path, key=jax.random.PRNGKey(0))


def _cmd_run(args: argparse.Namespace) -> None:
    from . import video
    from .pose2d import inference
    from .skeleton import Skeleton

    cameras = CameraGroup.from_config(args.config)
    skeleton = Skeleton.fly()
    model = _load_detector(args.checkpoint, args.backend)

    # Read per-camera frames named after the camera (e.g. camera_rh.mp4 or a
    # subdirectory of images per camera name).
    root = Path(args.input)
    frames = []
    for name in cameras.names:
        hits = list(root.glob(f"*{name}.mp4")) + [root / name]
        src = next((h for h in hits if h.exists()), None)
        if src is None:
            raise SystemExit(f"no video/dir for camera {name!r} under {root}")
        frames.append(
            video.read_video(src, backend=args.video_backend)
            if src.is_file()
            else video.read_images(src)
        )

    sides, flips = inference.fly_camera_layout(cameras.names)
    pts2d, conf = inference.detect_sequence(model, frames, sides, flips)
    result = run_from_points2d(
        cameras,
        skeleton,
        pts2d,
        conf,
        do_calibrate=not args.no_calibrate,
        smooth=args.smooth,
        fps=args.fps,
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

    from .pose2d.download import download_torch_weights, jax_weights_path
    from .pose2d.model import HourglassNet
    from .pose2d.weights import (
        convert_state_dict,
        save_checkpoint,
        state_dict_from_torch_checkpoint,
    )

    src = args.pth or download_torch_weights()
    sd = state_dict_from_torch_checkpoint(src)
    model = convert_state_dict(sd, HourglassNet.deepfly2d(key=jax.random.PRNGKey(0)))
    out = args.output or jax_weights_path()
    save_checkpoint(model, out)
    print(f"converted {src} -> {out}")


def _cmd_pose3d(args: argparse.Namespace) -> None:
    result = PoseResult.load(args.input)
    cameras = result.cameras
    if args.config is not None:
        cameras = CameraGroup.from_config(args.config)
    out = run_from_points2d(
        cameras,
        result.skeleton,
        result.pts2d,
        result.conf,
        do_calibrate=not args.no_calibrate,
        smooth=args.smooth,
        fps=args.fps,
    )
    out.save(args.output)
    print(f"wrote {args.output}  ({out.n_frames} frames, {out.n_views} views)")


def _cmd_visualize(args: argparse.Namespace) -> None:
    from . import video  # imports matplotlib/imageio lazily

    result = PoseResult.load(args.input)
    if args.mode == "3d":
        video.render_pose3d_video(result, args.output, fps=args.fps)
    else:
        if args.images is None:
            raise SystemExit("2d overlay needs --images <video|dir>")
        frames = (
            video.read_video(args.images)
            if Path(args.images).is_file()
            else video.read_images(args.images)
        )
        video.render_overlay_video(
            result, frames, args.output, camera=args.camera, fps=args.fps
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

    pr = sub.add_parser("run", help="full pipeline: detect 2D -> calibrate -> 3D")
    pr.add_argument(
        "--in",
        dest="input",
        required=True,
        help="dir of per-camera videos/image folders",
    )
    pr.add_argument("--config", required=True, help="camera rig TOML")
    pr.add_argument("--out", dest="output", required=True)
    pr.add_argument("--backend", choices=["jax", "torch"], default="jax")
    pr.add_argument(
        "--video-backend",
        default="auto",
        help="frame reader: auto|imageio|opencv|pyav|decord|video_reader_rs|"
        "torchcodec|pynvvideocodec|dali",
    )
    pr.add_argument(
        "--checkpoint", help="detector weights (.eqx for jax, .tar for torch)"
    )
    pr.add_argument("--no-calibrate", action="store_true")
    pr.add_argument("--smooth", choices=["gaussian", "one_euro"], default=None)
    pr.add_argument("--fps", type=float, default=100.0)
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
    p3.add_argument("--in", dest="input", required=True)
    p3.add_argument("--out", dest="output", required=True)
    p3.add_argument("--config", help="camera TOML overriding the stored cameras")
    p3.add_argument("--no-calibrate", action="store_true")
    p3.add_argument("--smooth", choices=["gaussian", "one_euro"], default=None)
    p3.add_argument("--fps", type=float, default=100.0)
    p3.set_defaults(func=_cmd_pose3d)

    pv = sub.add_parser("visualize", help="render a 2D overlay or 3D skeleton MP4")
    pv.add_argument("--in", dest="input", required=True)
    pv.add_argument("--out", dest="output", required=True)
    pv.add_argument("--mode", choices=["2d", "3d"], default="3d")
    pv.add_argument("--images", help="video/dir of frames (2d mode)")
    pv.add_argument("--camera", type=int, default=0)
    pv.add_argument("--fps", type=float, default=30.0)
    pv.set_defaults(func=_cmd_visualize)

    pi = sub.add_parser("info", help="print a summary of a result file")
    pi.add_argument("--in", dest="input", required=True)
    pi.set_defaults(func=_cmd_info)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)
