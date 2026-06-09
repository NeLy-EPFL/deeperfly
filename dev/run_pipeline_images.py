"""Run the full deeperfly pipeline on a flat directory of synchronized frames.

The ``data/images`` recording stores one JPEG per (camera, frame) as
``camera_<cam>_img_<frame>.jpg``; frames with the same number are synchronized
across the seven cameras. This driver maps camera indices 0..6 onto the
canonical fly-rig names (``rh rm rf f lf lm lh``), streams the aligned frames
through the 2D detector (loading 7 images at a time so memory stays flat),
calibrates the cameras with fly-as-target bundle adjustment, triangulates with
outlier rejection, optionally smooths, and writes a ``PoseResult`` HDF5.

    uv run python dev/run_pipeline_images.py --frames 200 --out results/fly.h5

The example rig's principal point assumes a 1024x512 sensor; these frames are
960x480, so the principal point is recentered to the real image center (the
intrinsics are otherwise the idealized example values and stay fixed in BA).

Throughput. The 8-stack hourglass forward (PyTorch) is the floor; batching past one
synchronized frame barely helps -- it already saturates the GPU. Image decode is
prefetched on worker threads (``--workers``) so disk + JPEG decode overlap the GPU
compute instead of serializing in front of it. Inference stays float32 (bf16/f16
shift even high-confidence peaks by several pixels -- not worth it for a sub-3px
pipeline).
"""

from __future__ import annotations

import argparse
import time
import tomllib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from deeperfly import video
from deeperfly.cameras import CameraGroup
from deeperfly.pipeline import run_from_points2d
from deeperfly.pose2d import backends, inference
from deeperfly.skeleton import Skeleton

CAMERA_NAMES = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]  # camera 0..6


def load_detector(checkpoint: str | None):
    from deeperfly.pose2d.download import torch_weights_path

    return backends.load_detector(checkpoint or torch_weights_path())


def build_detector(model, sides, flips, *, method="weighted", radius=2):
    """Return ``detect(batch) -> (pts (K,V,38,2) px, conf (K,V,38))``.

    ``batch`` is ``K`` synchronized frames as ``(K, V, H, W)`` uint8 (grayscale,
    as in this rig). Each frame runs through :func:`inference.detect`, which stacks
    its passes (the front camera twice, for both body sides) into one forward and
    decodes the peaks (``method`` / ``radius``, see
    :func:`inference.heatmap_to_points`).
    """
    n_views = len(sides)

    def detect_torch(batch):
        k = batch.shape[0]
        pts = np.empty((k, n_views, 38, 2))
        conf = np.empty((k, n_views, 38))
        for i in range(k):
            pts[i], conf[i] = inference.detect(
                model,
                [batch[i, v] for v in range(n_views)],
                sides,
                flips,
                method=method,
                radius=radius,
            )
        return pts, conf

    return detect_torch


def detect_pictorial(
    model,
    sides,
    flips,
    frame_ids,
    load_stack,
    *,
    k,
    workers,
    method="weighted",
    radius=2,
):
    """Stream detection keeping the top-K candidate peaks per joint (PS path).

    Pictorial structures needs the full heatmaps (to read off secondary peaks), so
    this deliberately uses the un-fused detect path -- ``backends.predict_heatmaps``
    per frame -- instead of the on-GPU fast path. The single-peak ``(pts2d, conf)``
    (for calibration) and the top-K candidates are both decoded with ``method`` /
    ``radius`` (see :func:`inference.heatmap_to_points`); in this path ``radius``
    also sets the candidate NMS neighbourhood. Returns ``(pts2d, conf)`` and a
    :class:`deeperfly.pictorial.Candidates`.
    """
    from deeperfly import pictorial
    from deeperfly.pose2d import backends

    n_views, n_t = len(sides), len(frame_ids)
    n_pts = 2 * inference.N_SIDE_JOINTS
    # Expand to passes: the front camera runs twice (un-flipped -> right legs,
    # flipped -> left legs), so it observes both body sides.
    pass_views, pass_sides, pass_flips = inference.expand_passes(sides, flips)
    pts2d = np.full((n_views, n_t, n_pts, 2), np.nan)
    conf = np.zeros((n_views, n_t, n_pts))
    cand_xy = np.full((n_views, n_t, n_pts, k, 2), np.nan)
    cand_score = np.zeros((n_views, n_t, n_pts, k))

    def detect_frame(stack):
        images = [stack[v] for v in range(n_views)]
        inputs = np.stack(
            [
                np.asarray(inference.preprocess(images[vw], flip=fl))
                for vw, fl in zip(pass_views, pass_flips)
            ]
        )
        heatmaps = np.asarray(backends.predict_heatmaps(model, inputs))
        image_size = [(im.shape[1], im.shape[0]) for im in images]
        pnorm, c = inference.heatmap_to_points(heatmaps, method=method, radius=radius)
        p2, cf = inference.assemble_skeleton(
            np.asarray(pnorm),
            np.asarray(c),
            pass_sides,
            pass_flips,
            image_size,
            views=pass_views,
            n_views=n_views,
        )
        cxy, csc = pictorial.extract_candidates(
            heatmaps,
            pass_sides,
            pass_flips,
            image_size,
            k=k,
            views=pass_views,
            n_views=n_views,
            method=method,
            radius=radius,
        )
        return p2, cf, cxy, csc

    depth = workers + 1
    report = 100
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        inflight = deque(pool.submit(load_stack, fid) for fid in frame_ids[:depth])
        for t in range(n_t):
            stack = inflight.popleft().result()
            nxt = t + depth
            if nxt < n_t:
                inflight.append(pool.submit(load_stack, frame_ids[nxt]))
            pts2d[:, t], conf[:, t], cand_xy[:, t], cand_score[:, t] = detect_frame(
                stack
            )
            if t + 1 >= report or t == n_t - 1:
                dt = time.perf_counter() - t0
                print(f"  detected {t + 1}/{n_t} frames  ({(t + 1) / dt:.1f} frame/s)")
                report += 100
    return pts2d, conf, pictorial.Candidates(xy=cand_xy, score=cand_score)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--config", default="examples/cameras.toml")
    ap.add_argument("--out", default="results/fly_pose.h5")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--frames", type=int, default=None, help="num frames (default all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--workers", type=int, default=4, help="prefetch threads for image decode"
    )
    ap.add_argument(
        "--batch",
        default="auto",
        help="synchronized frames per GPU forward ('auto' sizes it to VRAM)",
    )
    ap.add_argument("--smooth", choices=["gaussian", "one_euro"], default="one_euro")
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument(
        "--triangulation",
        choices=["ransac", "greedy", "dlt"],
        default="ransac",
        help="2D->3D triangulation: ransac consensus (default), greedy "
        "reprojection-outlier rejection, or plain dlt",
    )
    ap.add_argument(
        "--pictorial",
        action="store_true",
        help="run DeepFly3D-style pictorial structures before triangulation "
        "(slower; uses the full-heatmap detect path)",
    )
    ap.add_argument("--ps-k", type=int, default=5, help="candidate peaks/joint (PS)")
    ap.add_argument("--ps-temporal", action="store_true", help="PS temporal term")
    ap.add_argument("--ps-lambda", type=float, default=1.0, help="PS bone-prior weight")
    ap.add_argument(
        "--decode",
        choices=["argmax", "weighted", "taylor"],
        default="weighted",
        help="heatmap->point decode: raw arg-max cell, weighted-centroid sub-pixel "
        "(default), or DARK Taylor sub-pixel",
    )
    ap.add_argument(
        "--decode-radius",
        type=int,
        default=2,
        help="sub-pixel window half-width in heatmap px (taylor needs >=2)",
    )
    args = ap.parse_args()
    if args.decode == "taylor" and args.decode_radius < 2:
        ap.error("--decode taylor needs --decode-radius >= 2")

    root = Path(args.images)
    # Frame numbers present for every camera, in order.
    per_cam = [
        {int(p.stem.split("_img_")[1]) for p in root.glob(f"camera_{c}_img_*.jpg")}
        for c in range(len(CAMERA_NAMES))
    ]
    frame_ids = sorted(set.intersection(*per_cam))[:: args.stride]
    if args.frames is not None:
        frame_ids = frame_ids[: args.frames]
    n_t = len(frame_ids)
    print(f"{n_t} synchronized frames x {len(CAMERA_NAMES)} cameras")

    # Cameras: recenter the principal point onto the real 960x480 image center.
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    probe = video.read_frames([root / f"camera_0_img_{frame_ids[0]:06d}.jpg"])[0]
    h, w = probe.shape[:2]
    cfg.setdefault("camera_defaults", {})["principal_point_px"] = [
        (w - 1) / 2,
        (h - 1) / 2,
    ]
    cameras = CameraGroup.from_config(cfg)
    ba = cfg.get("bundle_adjustment", {})
    calibrate_kwargs = {"fixed": ba.get("fixed", []), "shared": ba.get("shared", [])}

    n_views = len(CAMERA_NAMES)
    skeleton = Skeleton.fly()
    sides, flips = inference.fly_camera_layout(CAMERA_NAMES)
    model = load_detector(args.checkpoint)
    n_pts = skeleton.n_points

    def load_stack(fid: int) -> np.ndarray:
        """The synchronized cameras for one frame as a single (V, H, W, 3) array."""
        return video.read_frames(
            [root / f"camera_{c}_img_{fid:06d}.jpg" for c in range(n_views)]
        )

    candidates = None
    t0 = time.perf_counter()
    print(f"decode: {args.decode} (radius {args.decode_radius})")
    if args.pictorial:
        # Accuracy mode: stream the full-heatmap detector and keep top-K candidates.
        print(f"pictorial structures: keeping {args.ps_k} candidates/joint")
        pts2d, conf, candidates = detect_pictorial(
            model,
            sides,
            flips,
            frame_ids,
            load_stack,
            k=args.ps_k,
            workers=args.workers,
            method=args.decode,
            radius=args.decode_radius,
        )
    else:
        detect_batch = build_detector(
            model,
            sides,
            flips,
            method=args.decode,
            radius=args.decode_radius,
        )

        # GPU batch (synchronized frames per forward). 'auto' uses the configured
        # forward batch; for this 8-stack net throughput plateaus at a small
        # batch on a fast GPU, so a modest batch saturates speed -- see
        # [pipeline.pose2d] batch_size.
        if args.batch == "auto":
            from deeperfly.config import Config

            vram = backends.gpu_memory_bytes()
            batch = max(1, Config.from_dict({}).pose2d.batch_size // n_views)
            where = f"VRAM {vram / 1e9:.1f} GB" if vram else "no GPU"
            print(
                f"batch: {batch} frame(s)/forward ({batch * n_views} imgs, auto, {where})"
            )
        else:
            batch = max(1, int(args.batch))
            print(f"batch: {batch} frame(s)/forward ({batch * n_views} imgs)")
        batch = min(batch, n_t)  # never batch more frames than we have

        # Stream detection: per-frame decode is prefetched on worker threads (so it
        # overlaps the GPU forward), frames are grouped into batches of `batch`, and
        # each group's 38-point 2D poses come back.
        pts2d = np.full((n_views, n_t, n_pts, 2), np.nan)
        conf = np.zeros((n_views, n_t, n_pts))
        warm = np.stack([load_stack(fid) for fid in frame_ids[:batch]])
        detect_batch(warm)  # warm up the JIT at the steady batch size before timing
        depth = args.workers + batch
        report = 100  # print progress roughly every 100 frames
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            inflight = deque(pool.submit(load_stack, fid) for fid in frame_ids[:depth])
            buf: list[np.ndarray] = []
            for t in range(n_t):
                buf.append(inflight.popleft().result())
                nxt = t + depth
                if nxt < n_t:
                    inflight.append(pool.submit(load_stack, frame_ids[nxt]))
                if len(buf) == batch or t == n_t - 1:
                    k = len(buf)
                    base = t + 1 - k
                    # Pad a short final group up to `batch` so the JIT shape never
                    # changes (one compile total); keep only the real `k` results.
                    group = buf + [buf[-1]] * (batch - k)
                    pf, cf = detect_batch(
                        np.stack(group)
                    )  # (batch,V,38,2),(batch,V,38)
                    pts2d[:, base : base + k] = pf[:k].transpose(1, 0, 2, 3)
                    conf[:, base : base + k] = cf[:k].transpose(1, 0, 2)
                    buf = []
                    if t + 1 >= report or t == n_t - 1:
                        dt = time.perf_counter() - t0
                        print(
                            f"  detected {t + 1}/{n_t} frames  ({(t + 1) / dt:.1f} frame/s)"
                        )
                        report += 100

    print(f"2D detection done in {time.perf_counter() - t0:.1f}s; calibrating + 3D ...")
    result = run_from_points2d(
        cameras,
        skeleton,
        pts2d,
        conf,
        do_calibrate=True,
        calibrate_kwargs=calibrate_kwargs,
        triangulation=args.triangulation,
        do_pictorial=args.pictorial,
        candidates=candidates,
        ps_kwargs={"temporal": args.ps_temporal, "lam": args.ps_lambda},
        smooth=args.smooth,
        fps=args.fps,
        meta={
            "source": str(root),
            "triangulation": args.triangulation,
            "pictorial": args.pictorial,
            "n_frames_input": n_t,
        },
    )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.out)

    re = result.reproj_error
    finite = np.isfinite(re)
    print(f"\nwrote {args.out}")
    print(f"  views x frames x points : {result.pts2d.shape}")
    print(f"  3D points               : {result.pts3d.shape}")
    print(f"  triangulated points     : {np.isfinite(result.pts3d).all(-1).sum()}")
    print(
        "  reprojection error (px) : "
        f"median {np.median(re[finite]):.2f}  mean {np.mean(re[finite]):.2f}  "
        f"p95 {np.percentile(re[finite], 95):.2f}"
    )


if __name__ == "__main__":
    main()
