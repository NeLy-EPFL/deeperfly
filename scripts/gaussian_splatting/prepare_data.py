"""Export a single-frame, multi-view dataset for Gaussian-splatting experiments.

Reads a deeperfly ``results.h5`` (bundle-adjusted cameras, 2D keypoints,
triangulated 3D points) plus the source videos, and writes a small, portable
dataset that the ``fit_gsplat.py`` script (which runs in an isolated gsplat
venv, without importing deeperfly) can consume.

deeperfly's camera model is OpenCV-convention world->camera ``R(rvec) @ X +
tvec`` with the rows of ``R`` being image-right (+x), image-down (+y),
camera-forward (+z), and intrinsics packed ``[fx, fy, cx, cy]`` with no
distortion here -- this maps 1:1 onto gsplat's ``viewmats`` (world->camera) and
``Ks`` (3x3), so the conversion is trivial.

Output layout (under ``--out``):
    images/<i>_<name>.png     RGB frame per camera
    masks/<i>_<name>.png      uint8 0/255 fly mask (keypoint hull & brightness)
    overlays/<i>_<name>.png   mask-on-image preview for eyeballing
    cameras.npz               names, viewmats (V,4,4), Ks (V,3,3), width, height
    init.npz                  points (M,3), colors (M,3 in 0..1), center, radius
    keypoints.npz             pts2d (V,P,2), points3d (P,3), bones (B,2)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def rvec_to_rmat(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return R


def project(
    points: np.ndarray, R: np.ndarray, t: np.ndarray, intr: np.ndarray
) -> np.ndarray:
    """Project world points (...,3) to pixels (...,2) under OpenCV pinhole."""
    fx, fy, cx, cy = intr
    cam = points @ R.T + t  # (...,3)
    z = np.clip(cam[..., 2], 1e-6, None)
    u = fx * cam[..., 0] / z + cx
    v = fy * cam[..., 1] / z + cy
    return np.stack([u, v], axis=-1)


def read_frame(video: Path, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame} from {video}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_mask(
    img_rgb: np.ndarray,
    pts2d: np.ndarray,
    pad_px: int,
    bright_gate: bool,
) -> np.ndarray:
    """Fly mask = dilated convex hull of 2D keypoints, gated by brightness.

    The support ball and the specular band are also bright, so brightness alone
    cannot isolate the fly; the keypoint hull is the discriminator. We keep the
    connected bright region that overlaps the keypoints and fill small holes.
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    finite = np.isfinite(pts2d).all(1)
    pts = pts2d[finite].astype(np.int32)
    hull = cv2.convexHull(pts)
    hull_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 255)
    k = 2 * pad_px + 1
    hull_dil = cv2.dilate(
        hull_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )

    if not bright_gate:
        return hull_dil

    # Otsu threshold over pixels inside the dilated hull -> separates fly/ball
    # from the black background; the hull already excludes the top specular band.
    inside = gray[hull_dil > 0]
    thr, _ = cv2.threshold(inside, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (gray.astype(np.float32) >= thr).astype(np.uint8) * 255
    m = cv2.bitwise_and(hull_dil, bright)
    # close small holes (dark eyes/joints on the fly)
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    # keep the connected component covering the most keypoints
    n, labels = cv2.connectedComponents(m)
    if n > 1:
        kp_px = pts2d[finite].astype(np.int32)
        kp_px[:, 0] = np.clip(kp_px[:, 0], 0, w - 1)
        kp_px[:, 1] = np.clip(kp_px[:, 1], 0, h - 1)
        votes = np.bincount(labels[kp_px[:, 1], kp_px[:, 0]], minlength=n)
        votes[0] = 0  # background label
        keep = int(votes.argmax())
        m = np.where(labels == keep, 255, 0).astype(np.uint8)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return m


def sample_colors(
    points: np.ndarray,
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    intrs: np.ndarray,
    images: list[np.ndarray],
    masks: list[np.ndarray],
) -> np.ndarray:
    """Per-point colour = mean of masked reprojected pixels over all views."""
    V = len(images)
    h, w = images[0].shape[:2]
    acc = np.zeros((len(points), 3), np.float64)
    cnt = np.zeros((len(points), 1), np.float64)
    for i in range(V):
        uv = project(points, Rs[i], ts[i], intrs[i])
        u = np.round(uv[:, 0]).astype(int)
        v = np.round(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        uu, vv = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
        inmask = ok & (masks[i][vv, uu] > 0)
        acc[inmask] += images[i][vv[inmask], uu[inmask]].astype(np.float64)
        cnt[inmask, 0] += 1
    col = np.full((len(points), 3), 0.5)
    hit = cnt[:, 0] > 0
    col[hit] = (acc[hit] / cnt[hit]) / 255.0
    return col.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="examples/data/deeperfly_outputs/results.h5")
    ap.add_argument("--videos-dir", default="examples/data")
    ap.add_argument("--frame", type=int, default=32)
    ap.add_argument("--out", default=None, help="default: scratchpad gs_data/frame_<N>")
    ap.add_argument(
        "--cameras",
        choices=["bundle_adjustment", "pose2d"],
        default="bundle_adjustment",
    )
    ap.add_argument(
        "--mask-pad", type=int, default=35, help="dilation (px) of keypoint hull"
    )
    ap.add_argument("--no-bright-gate", action="store_true", help="hull-only mask")
    ap.add_argument(
        "--n-random", type=int, default=12000, help="random init points in bbox"
    )
    ap.add_argument(
        "--bone-samples", type=int, default=6, help="interior points per bone"
    )
    args = ap.parse_args()

    out = Path(args.out or f"scratchpad/gs_data/frame_{args.frame}")
    if args.out is None:
        # resolve scratchpad relative to this repo's session scratchpad if present
        sp = Path(
            "/tmp/claude-1000/-home-tlam-deeperfly/570081dd-0897-484e-ae32-039178556c10/scratchpad"
        )
        if sp.exists():
            out = sp / "gs_data" / f"frame_{args.frame}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    with h5py.File(args.results, "r") as f:
        cam = f[f"{args.cameras}/cameras"]
        rvecs = cam["rvecs"][:]
        tvecs = cam["tvecs"][:]
        intrs = cam["intrs"][:]
        names = [
            n.decode() if isinstance(n, bytes) else str(n) for n in cam["names"][:]
        ]
        footage = json.loads(f["pose2d"].attrs["footage"])
        pts2d = f["pose2d/points"][:, args.frame]  # (V,P,2)
        p3d = f["triangulation/points3d"][args.frame]  # (P,3)
        bones = f["skeleton/bones"][:]

    V = len(names)
    Rs = [rvec_to_rmat(rvecs[i]) for i in range(V)]
    ts = [np.asarray(tvecs[i], dtype=np.float64) for i in range(V)]

    # world->camera 4x4 (OpenCV) and 3x3 K
    viewmats = np.zeros((V, 4, 4), np.float64)
    Ks = np.zeros((V, 3, 3), np.float64)
    for i in range(V):
        viewmats[i, :3, :3] = Rs[i]
        viewmats[i, :3, 3] = ts[i]
        viewmats[i, 3, 3] = 1.0
        fx, fy, cx, cy = intrs[i]
        Ks[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    images, masks = [], []
    H = W = None
    for i, name in enumerate(names):
        video = Path(footage[name]["abs"][0])
        if not video.exists():
            video = Path(args.videos_dir) / Path(footage[name]["rel"][0]).name
        img = read_frame(video, args.frame)
        H, W = img.shape[:2]
        mask = build_mask(img, pts2d[i], args.mask_pad, not args.no_bright_gate)
        images.append(img)
        masks.append(mask)
        cv2.imwrite(
            str(out / "images" / f"{i}_{name}.png"),
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(out / "masks" / f"{i}_{name}.png"), mask)
        ov = img.copy()
        ov[mask > 0] = (0.5 * ov[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(
            np.uint8
        )
        # draw reprojected keypoints for a calibration sanity check
        for uv in pts2d[i]:
            if np.isfinite(uv).all():
                cv2.circle(ov, (int(uv[0]), int(uv[1])), 3, (255, 60, 60), -1)
        cv2.imwrite(
            str(out / "overlays" / f"{i}_{name}.png"),
            cv2.cvtColor(ov, cv2.COLOR_RGB2BGR),
        )
        cover = 100.0 * (mask > 0).mean()
        print(f"  {name:>3} ({video.name}): mask covers {cover:5.1f}% of frame")

    # ---- init point cloud -------------------------------------------------
    finite = np.isfinite(p3d).all(1)
    kp = p3d[finite]
    center = kp.mean(0)
    radius = float(np.linalg.norm(kp - center, axis=1).max())

    pts = [kp]
    # densify along skeleton bones
    for a, b in bones:
        if finite[a] and finite[b]:
            fr = np.linspace(0, 1, args.bone_samples + 2)[1:-1][:, None]
            pts.append(p3d[a] * (1 - fr) + p3d[b] * fr)
    # random fill inside padded bbox of the fly
    lo, hi = kp.min(0), kp.max(0)
    pad = 0.15 * (hi - lo)
    rng = np.random.default_rng(0)
    rand = rng.uniform(lo - pad, hi + pad, size=(args.n_random, 3))
    pts.append(rand)
    points = np.concatenate(pts, 0).astype(np.float32)
    colors = sample_colors(points.astype(np.float64), Rs, ts, intrs, images, masks)

    np.savez(
        out / "cameras.npz",
        names=np.array(names),
        viewmats=viewmats.astype(np.float32),
        Ks=Ks.astype(np.float32),
        width=W,
        height=H,
    )
    np.savez(
        out / "init.npz",
        points=points,
        colors=colors,
        center=center.astype(np.float32),
        radius=np.float32(radius),
    )
    np.savez(
        out / "keypoints.npz",
        pts2d=pts2d.astype(np.float32),
        points3d=p3d.astype(np.float32),
        bones=bones,
    )

    cam_centers = np.stack([-Rs[i].T @ ts[i] for i in range(V)])
    print(f"\nframe {args.frame}: {V} views {W}x{H}")
    print(f"  fly center {np.round(center, 3)}  radius {radius:.3f}")
    print(
        f"  camera dist to center: {np.round(np.linalg.norm(cam_centers - center, axis=1), 1)}"
    )
    print(f"  init points: {len(points)} ({len(kp)} keypoints seeded)")
    print(f"  wrote dataset -> {out}")


if __name__ == "__main__":
    main()
