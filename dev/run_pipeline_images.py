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
960x480, so the principal point is recentred to the real image centre (the
intrinsics are otherwise the idealised example values and stay fixed in BA).

Throughput. The 8-stack hourglass forward is the floor (~230 img/s in float32 on
an RTX 4090, and batching past one synchronized frame does not help -- it already
saturates the GPU). Two float32-exact tricks close the gap to that floor:

* The JAX path fuses preprocessing (flip + resize + mean-subtract), the forward
  pass and the arg-max peak decode into a single jitted, vmapped call, so each
  frame uploads only raw uint8 and downloads only the 19x2 peaks -- no per-image
  resize round-trips and no multi-MB heatmap transfer.
* Image decode is prefetched on worker threads (``--workers``) so disk + JPEG
  decode overlap the GPU compute instead of serialising in front of it.

Together these take this recording from ~19 to ~31 frame/s. (Half precision would
roughly double the forward, but bf16/f16 shift even high-confidence peaks by
several pixels -- not worth it for a sub-3px pipeline -- so inference stays
float32.)
"""

from __future__ import annotations

import argparse
import time
import tomllib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from deeperfly.cameras import CameraGroup
from deeperfly.pipeline import run_from_points2d
from deeperfly.pose2d import backends, inference
from deeperfly.skeleton import Skeleton

CAMERA_NAMES = ["rh", "rm", "rf", "f", "lf", "lm", "lh"]  # camera 0..6


def load_detector(backend: str, checkpoint: str | None):
    from deeperfly.pose2d.download import jax_weights_path, torch_weights_path

    default = torch_weights_path() if backend == "torch" else jax_weights_path()
    return backends.load_detector(backend, checkpoint or default)


def build_frame_detector(model, backend: str, sides, flips):
    """Return ``detect(stack) -> (pts (V,38,2) px, conf (V,38))`` for one frame.

    ``stack`` is the synchronized cameras as a single ``(V, H, W)`` uint8 array
    (grayscale, as in this rig). The JAX backend fuses preprocess + forward +
    arg-max into one jitted vmap so only uint8 goes up and only peaks come down;
    it is numerically identical to :func:`inference.detect` (same ops, same
    float32). The torch backend reuses :func:`inference.detect` per frame.
    """
    if backend != "jax":
        return lambda stack: inference.detect(
            model, [stack[v] for v in range(stack.shape[0])], sides, flips
        )

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    flips_arr = jnp.asarray(flips)
    out_h, out_w = inference.IMG_SIZE
    mean = inference.MEAN

    @eqx.filter_jit
    def _peaks(m, stack_u8, fl):
        def one(img, flip):  # img: (H, W) uint8 grayscale, flip: bool
            x = img.astype(jnp.float32) / 255.0
            x = jnp.stack([x, x, x], axis=-1)  # grayscale -> 3 channels
            x = jnp.where(flip, x[:, ::-1, :], x)  # mirror far-side cameras
            x = jax.image.resize(x, (out_h, out_w, 3), method="linear", antialias=True)
            x = jnp.transpose(x, (2, 0, 1)) - mean
            hm = m.heatmaps(x)
            j, h, w = hm.shape
            flat = hm.reshape(j, h * w)
            idx = jnp.argmax(flat, axis=-1)
            pts = jnp.stack([(idx % w) / w, (idx // w) / h], axis=-1)
            return pts, jnp.max(flat, axis=-1)

        return jax.vmap(one)(stack_u8, fl)

    def detect(stack):
        pts_norm, conf = _peaks(model, jnp.asarray(stack), flips_arr)
        image_size = [(stack.shape[2], stack.shape[1])] * stack.shape[0]
        return inference.assemble_skeleton(
            np.asarray(pts_norm), np.asarray(conf), sides, flips, image_size
        )

    return detect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--config", default="examples/cameras.toml")
    ap.add_argument("--out", default="results/fly_pose.h5")
    ap.add_argument("--backend", choices=["jax", "torch"], default="jax")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--frames", type=int, default=None, help="num frames (default all)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument(
        "--workers", type=int, default=4, help="prefetch threads for image decode"
    )
    ap.add_argument("--smooth", choices=["gaussian", "one_euro"], default="one_euro")
    ap.add_argument("--fps", type=float, default=100.0)
    args = ap.parse_args()

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

    # Cameras: recentre the principal point onto the real 960x480 image centre.
    with open(args.config, "rb") as f:
        cfg = tomllib.load(f)
    probe = iio.imread(root / f"camera_0_img_{frame_ids[0]:06d}.jpg")
    h, w = probe.shape[:2]
    cfg.setdefault("camera_defaults", {})["principal_point_px"] = [
        (w - 1) / 2,
        (h - 1) / 2,
    ]
    cameras = CameraGroup.from_config(cfg)
    ba = cfg.get("bundle_adjustment", {})
    calibrate_kwargs = {"fixed": ba.get("fixed", []), "shared": ba.get("shared", [])}

    skeleton = Skeleton.fly()
    sides, flips = inference.fly_camera_layout(CAMERA_NAMES)
    model = load_detector(args.backend, args.checkpoint)
    detect_frame = build_frame_detector(model, args.backend, sides, flips)

    def load_stack(fid: int) -> np.ndarray:
        """The synchronized cameras for one frame as a single (V, H, W[, C]) array."""
        return np.stack(
            [
                iio.imread(root / f"camera_{c}_img_{fid:06d}.jpg")
                for c in range(len(CAMERA_NAMES))
            ]
        )

    # Stream detection: decode is prefetched on worker threads (overlapping the
    # GPU forward), one synchronized frame in, its 38-point 2D pose out.
    n_pts = skeleton.n_points
    pts2d = np.full((len(CAMERA_NAMES), n_t, n_pts, 2), np.nan)
    conf = np.zeros((len(CAMERA_NAMES), n_t, n_pts))
    detect_frame(load_stack(frame_ids[0]))  # warm up the JIT before timing
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        inflight = deque(
            pool.submit(load_stack, fid) for fid in frame_ids[: args.workers + 1]
        )
        for t in range(n_t):
            stack = inflight.popleft().result()
            nxt = t + args.workers + 1
            if nxt < n_t:
                inflight.append(pool.submit(load_stack, frame_ids[nxt]))
            pts2d[:, t], conf[:, t] = detect_frame(stack)
            if (t + 1) % 100 == 0 or t + 1 == n_t:
                dt = time.perf_counter() - t0
                print(f"  detected {t + 1}/{n_t} frames  ({(t + 1) / dt:.1f} frame/s)")

    print(f"2D detection done in {time.perf_counter() - t0:.1f}s; calibrating + 3D ...")
    result = run_from_points2d(
        cameras,
        skeleton,
        pts2d,
        conf,
        do_calibrate=True,
        calibrate_kwargs=calibrate_kwargs,
        smooth=args.smooth,
        fps=args.fps,
        meta={"source": str(root), "backend": args.backend, "n_frames_input": n_t},
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
