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
import struct
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


# --------------------------------------------------------------------------- #
# NeuroMechFly leg-mesh seeding
#
# deeperfly's 5 keypoints/leg (thorax_coxa, coxa_trochanter, femur_tibia,
# tibia_tarsus, claw) map 1:1 onto NeuroMechFly's leg-segment meshes
# (coxa, trochanterfemur, tibia, tarsus1..5). We place each segment mesh onto
# its keypoint bone with a similarity fit (align the mesh's principal axis to
# the bone, scale so its length matches, translate) -- no MuJoCo/IK needed --
# then sample its surface. This seeds the thin legs with anatomically-shaped,
# correctly-proportioned volume, which a random/hull init cannot resolve.
# --------------------------------------------------------------------------- #
LEG_JOINTS = ["thorax_coxa", "coxa_trochanter", "femur_tibia", "tibia_tarsus", "claw"]


def load_stl(path: Path) -> np.ndarray:
    """Load a binary (or ASCII) STL as a (T,3,3) array of triangle vertices."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + n * 50 == len(data):
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(data, dtype=dt, count=n, offset=84)["v"].astype(
                np.float64
            )
    verts = [
        [float(x) for x in ln.split()[1:4]]
        for ln in data.decode("ascii", "ignore").splitlines()
        if ln.strip().startswith("vertex")
    ]
    return np.asarray(verts, np.float64).reshape(-1, 3, 3)


def sample_surface(tris: np.ndarray, n: int, rng) -> np.ndarray:
    """Area-weighted uniform surface sampling of a triangle soup (T,3,3)."""
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    if area.sum() <= 0:
        return v0
    idx = rng.choice(len(tris), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1.0
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    a, b, c = v0[idx], v1[idx], v2[idx]
    return a + u[:, None] * (b - a) + w[:, None] * (c - a)


def _rot_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation mapping unit vector ``a`` onto unit vector ``b`` (Rodrigues)."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c > 1 - 1e-8:
        return np.eye(3)
    if c < -1 + 1e-8:  # antiparallel: 180 deg about any perpendicular axis
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, perp)
        ax /= np.linalg.norm(ax)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1.0 / (1.0 + c))


def place_segment(verts: np.ndarray, kp_a: np.ndarray, kp_b: np.ndarray) -> np.ndarray:
    """Similarity-fit mesh ``verts`` so its long axis spans bone kp_a -> kp_b."""
    c = verts.mean(0)
    Vc = verts - c
    _, _, Vt = np.linalg.svd(Vc, full_matrices=False)
    axis = Vt[0]
    proj = Vc @ axis
    tmin, tmax = float(proj.min()), float(proj.max())
    p0 = c + axis * tmin  # proximal end in mesh coords
    length_mesh = tmax - tmin
    d = kp_b - kp_a
    length_bone = float(np.linalg.norm(d))
    if length_mesh < 1e-9 or length_bone < 1e-9:
        return np.empty((0, 3))
    R = _rot_align(axis, d / length_bone)
    s = length_bone / length_mesh
    return kp_a + s * ((verts - p0) @ R.T)


def nmf_leg_seed(
    point_names: list[str],
    p3d: np.ndarray,
    finite: np.ndarray,
    mesh_dir: Path,
    per_seg: int,
    rng,
) -> np.ndarray:
    """Seed all six legs from NeuroMechFly segment meshes. Right-side legs reuse
    the (mirrored) left meshes; similarity placement makes the mirror moot."""
    name2idx = {n: i for i, n in enumerate(point_names)}
    out = []
    n_ok = 0
    for leg in ["lf", "lm", "lh", "rf", "rm", "rh"]:
        try:
            idx = [name2idx[f"{leg}_{j}"] for j in LEG_JOINTS]
        except KeyError:
            continue
        ml = "l" + leg[1]  # only left meshes exist -> lf/lm/lh
        segs = [
            (f"{ml}_coxa", idx[0], idx[1]),
            (f"{ml}_trochanterfemur", idx[1], idx[2]),
            (f"{ml}_tibia", idx[2], idx[3]),
        ]
        for mesh, ia, ib in segs:
            fp = mesh_dir / f"{mesh}.stl"
            if fp.exists() and finite[ia] and finite[ib]:
                surf = sample_surface(load_stl(fp), per_seg, rng)
                w = place_segment(surf, p3d[ia], p3d[ib])
                if len(w):
                    out.append(w)
                    n_ok += 1
        # tarsus chain: split (tibia_tarsus -> claw) into the 5 tarsomeres
        ta, tb = idx[3], idx[4]
        if finite[ta] and finite[tb]:
            A, B = p3d[ta], p3d[tb]
            for k in range(5):
                fp = mesh_dir / f"{ml}_tarsus{k + 1}.stl"
                if not fp.exists():
                    continue
                pa = A + (k / 5.0) * (B - A)
                pb = A + ((k + 1) / 5.0) * (B - A)
                surf = sample_surface(load_stl(fp), max(per_seg // 3, 120), rng)
                w = place_segment(surf, pa, pb)
                if len(w):
                    out.append(w)
                    n_ok += 1
    if not out:
        return np.empty((0, 3), np.float32)
    print(f"  NeuroMechFly seed: {n_ok} leg segments placed")
    return np.concatenate(out, 0).astype(np.float32)


# NMF body hierarchy (child -> parent), joints at neutral; used to assemble the
# rigid anterior body (thorax/head/antennae) by forward kinematics. The abdomen
# is deliberately excluded -- it flexes and this fly's is distended, so the
# neutral pose mismatches; the visual hull covers that large smooth mass well.
NMF_PARENT = {
    "c_thorax": None,
    "c_head": "c_thorax",
    "l_eye": "c_head",
    "c_rostrum": "c_head",
    "c_haustellum": "c_rostrum",
    "l_pedicel": "c_head",
    "l_funiculus": "l_pedicel",
    "lf_coxa": "c_thorax",
    "lm_coxa": "c_thorax",
    "lh_coxa": "c_thorax",
    "rf_coxa": "c_thorax",
    "rm_coxa": "c_thorax",
    "rh_coxa": "c_thorax",
}
NMF_BODY_MESHES = [
    "c_thorax",
    "c_head",
    "c_rostrum",
    "c_haustellum",
    "l_eye",
    "l_pedicel",
    "l_funiculus",
]
NMF_POS_SCALE = 0.001  # rigging pos are mm; STL meshes are metres


def _quat2R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def _nmf_fk(rig: dict) -> dict:
    """World (R,p) per body in the assembled NMF frame (neutral joints=0)."""
    T = {}

    def solve(name):
        if name in T:
            return T[name]
        pos = np.asarray(rig[name]["pos"], float) * NMF_POS_SCALE
        Rl = _quat2R(rig[name]["quat"])
        par = NMF_PARENT[name]
        if par is None:
            T[name] = (Rl, pos)
        else:
            Rp, pp = solve(par)
            T[name] = (Rp @ Rl, pp + Rp @ pos)
        return T[name]

    for n in NMF_PARENT:
        solve(n)
    return T


def _umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) mapping src -> dst (Umeyama 1991)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = (D.T @ S) / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    dsign = np.sign(np.linalg.det(U @ Vt))
    W = np.diag([1.0, 1.0, dsign])
    R = U @ W @ Vt
    s = float(np.trace(np.diag(sig) @ W) / ((S**2).sum() / len(src)))
    t = mu_d - s * R @ mu_s
    return s, R, t


def nmf_body_seed(point_names, p3d, finite, mesh_dir, rig_path, per_seg, rng):
    """Seed the rigid anterior body from NMF meshes: assemble via FK, fit to the
    world with a similarity on the 6 thorax-coxa joints, surface-sample."""
    import yaml

    rig = yaml.safe_load(Path(rig_path).read_text())
    T = _nmf_fk(rig)
    n2i = {n: i for i, n in enumerate(point_names)}
    legs = ["lf", "lm", "lh", "rf", "rm", "rh"]
    have = [lg for lg in legs if finite[n2i[f"{lg}_thorax_coxa"]]]
    if len(have) < 3:
        return np.empty((0, 3), np.float32)
    src = np.array([T[f"{lg}_coxa"][1] for lg in have])
    dst = np.array([p3d[n2i[f"{lg}_thorax_coxa"]] for lg in have])
    s, R, t = _umeyama(src, dst)
    resid = np.linalg.norm((s * (src @ R.T) + t) - dst, axis=1).mean()

    out = []
    for m in NMF_BODY_MESHES:
        fp = mesh_dir / f"{m}.stl"
        if not fp.exists() or m not in T:
            continue
        Rw, pw = T[m]
        surf = sample_surface(load_stl(fp), per_seg, rng)
        out.append((s * ((surf @ Rw.T + pw) @ R.T) + t).astype(np.float32))
    if not out:
        return np.empty((0, 3), np.float32)
    print(f"  NeuroMechFly body: {len(out)} segs, coxa-fit residual {resid:.3f}")
    return np.concatenate(out, 0)


def carve_visual_hull(
    masks: list[np.ndarray],
    Rs: list[np.ndarray],
    ts: list[np.ndarray],
    intrs: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    res: int,
    min_views: int,
) -> np.ndarray:
    """Space-carve a voxel grid: keep voxel centres that reproject inside the
    fly mask in at least ``min_views`` of the views (silhouette intersection).

    This is a strong, cheap geometric prior: the true fly+ball volume is
    silhouette-consistent in every view, so it survives; empty background does
    not. Thin legs leave phantom volume (silhouettes of thin structures
    intersect loosely) but that only over-seeds -- unsupported init is pruned.
    """
    gx = np.linspace(lo[0], hi[0], res)
    gy = np.linspace(lo[1], hi[1], res)
    gz = np.linspace(lo[2], hi[2], res)
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)  # (res^3, 3)
    h, w = masks[0].shape
    count = np.zeros(P.shape[0], np.int16)
    for i in range(len(masks)):
        uv = project(P, Rs[i], ts[i], intrs[i])
        u = np.round(uv[:, 0]).astype(np.int64)
        v = np.round(uv[:, 1]).astype(np.int64)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        uu, vv = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
        count += (ok & (masks[i][vv, uu] > 0)).astype(np.int16)
    return P[count >= min_views]


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
        "--n-random",
        type=int,
        default=3000,
        help="light uniform bbox fill (robustness)",
    )
    ap.add_argument(
        "--bone-samples", type=int, default=8, help="minimum interior points per bone"
    )
    ap.add_argument(
        "--bone-spacing",
        type=float,
        default=0.02,
        help="target spacing between bone samples, in units of fly radius",
    )
    ap.add_argument(
        "--bone-jitter",
        type=float,
        default=0.012,
        help="perpendicular jitter on bone samples (units of fly radius)",
    )
    ap.add_argument("--no-hull", action="store_true", help="disable visual-hull carve")
    ap.add_argument("--hull-res", type=int, default=144, help="voxel grid resolution")
    ap.add_argument(
        "--hull-min-views",
        type=int,
        default=0,
        help="min views in-mask to keep a voxel (0 => V-1)",
    )
    ap.add_argument(
        "--n-hull", type=int, default=30000, help="points sampled from hull"
    )
    ap.add_argument(
        "--hull-pad", type=float, default=0.18, help="carve bbox pad (units of radius)"
    )
    ap.add_argument(
        "--no-nmf-legs",
        action="store_true",
        help="disable NeuroMechFly leg-mesh seeding",
    )
    ap.add_argument(
        "--nmf-mesh-dir",
        default="~/flygym/src/flygym/assets/model/neuromechfly/meshes/simplified_max2000faces",
        help="dir of NeuroMechFly simplified segment STLs",
    )
    ap.add_argument(
        "--nmf-per-seg", type=int, default=500, help="surface pts per leg segment mesh"
    )
    ap.add_argument(
        "--no-nmf-body",
        action="store_true",
        help="disable NeuroMechFly anterior-body (thorax/head) seeding",
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
        point_names = [
            n.decode() if isinstance(n, bytes) else str(n)
            for n in f["skeleton/point_names"][:]
        ]

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
    rng = np.random.default_rng(0)

    # Each source carries a position-prior weight in [0,1]: how strongly a fit
    # should anchor that gaussian to its seeded location. The NMF anatomical mesh
    # is trusted most (mesh-as-geometry), the visual hull loosely, random fill
    # not at all. Saved as ``prior_w`` for a surface-anchored (fixed) fit.
    pts, wts = [], []

    def add(arr, w):
        if arr is not None and len(arr):
            pts.append(np.asarray(arr, np.float32))
            wts.append(np.full(len(arr), w, np.float32))

    add(kp, 0.8)

    # (1) dense seeding ALONG skeleton bones, sample count proportional to bone
    #     length, with a small perpendicular jitter so each bone seeds a thin
    #     tube not a bare line -- this is what lets the thin legs form.
    n_bone = 0
    jit = args.bone_jitter * radius
    for a, b in bones:
        if finite[a] and finite[b]:
            length = float(np.linalg.norm(p3d[b] - p3d[a]))
            m = max(
                args.bone_samples, int(round(length / (args.bone_spacing * radius)))
            )
            fr = np.linspace(0, 1, m)[:, None]
            seg = p3d[a] * (1 - fr) + p3d[b] * fr
            seg = seg + rng.normal(size=seg.shape) * jit
            add(seg, 0.5)
            n_bone += m

    # (1b) NeuroMechFly leg-mesh seeding -- anatomically-shaped thin legs.
    mesh_dir = Path(args.nmf_mesh_dir).expanduser()
    if not args.no_nmf_legs and mesh_dir.exists():
        add(
            nmf_leg_seed(point_names, p3d, finite, mesh_dir, args.nmf_per_seg, rng), 1.0
        )
    elif not args.no_nmf_legs:
        print(f"  NeuroMechFly meshes not found at {mesh_dir} -> skipping leg seed")

    # (1c) NeuroMechFly anterior-body seeding (thorax/head/antennae; abdomen ->
    #      hull). Assembled via FK from rigging.yaml, fit to the 6 coxa joints.
    rig_path = mesh_dir.parent.parent / "rigging.yaml"
    if not args.no_nmf_body and mesh_dir.exists() and rig_path.exists():
        add(
            nmf_body_seed(
                point_names, p3d, finite, mesh_dir, rig_path, args.nmf_per_seg, rng
            ),
            1.0,
        )

    # (2) visual-hull carve from the 7 masks -> volumetric init concentrated in
    #     the actual fly+ball, not the (mostly empty) bounding box.
    if not args.no_hull:
        ext = kp.max(0) - kp.min(0)
        half = 0.5 * ext + args.hull_pad * radius
        lo, hi = center - half, center + half
        min_views = args.hull_min_views if args.hull_min_views > 0 else V - 1
        hull = carve_visual_hull(masks, Rs, ts, intrs, lo, hi, args.hull_res, min_views)
        if len(hull):
            n = min(args.n_hull, len(hull))
            idx = rng.choice(len(hull), size=n, replace=len(hull) < args.n_hull)
            vox = (hi - lo) / (args.hull_res - 1)
            add(hull[idx] + rng.uniform(-0.5, 0.5, size=(n, 3)) * vox, 0.35)
            print(
                f"  visual hull: {len(hull)} voxels in >={min_views}/{V} masks "
                f"-> {n} init pts"
            )
        else:
            print("  visual hull empty -> relying on bone + bbox fill")

    # (3) light uniform bbox fill for robustness (no position prior)
    if args.n_random > 0:
        lo2, hi2 = kp.min(0), kp.max(0)
        pad = 0.15 * (hi2 - lo2)
        add(rng.uniform(lo2 - pad, hi2 + pad, size=(args.n_random, 3)), 0.0)

    points = np.concatenate(pts, 0).astype(np.float32)
    prior_w = np.concatenate(wts, 0).astype(np.float32)
    colors = sample_colors(points.astype(np.float64), Rs, ts, intrs, images, masks)
    print(
        f"  init: {len(kp)} keypoints + {n_bone} bone + hull/fill = {len(points)} pts "
        f"({int((prior_w >= 0.99).sum())} on NMF mesh)"
    )

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
        prior_w=prior_w,
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
