"""Solve single-T once, apply rigid-move-legs-only, RENDER far-leg overlay.

Per camera tile over the raw frame:
  RED  = BA far-leg bones (current)
  GREEN= corrected far-leg bones (rigid-move-legs-only + compensated viz rig)
  dim CYAN = near-leg bones of the corrected pose (to check leg-base attachment)
Also prints the body-shift-in-left-views (the one known imperfection).
"""

from __future__ import annotations

import sys

import cv2
import h5py
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import least_squares

sys.path.insert(0, "src")
from deeperfly import io
from deeperfly.cameras import CameraGroup
from deeperfly.config import Config
from deeperfly.photometric.maps import build_leg_maps
from deeperfly.photometric.objective import (
    compensate_left,
    far_leg_bones,
    left_point_indices,
    project_np,
    rigid,
)
from deeperfly.pipeline.core import _subsample

REC = "examples/JSP_SCAPE_260417_IN07B001_Fly3_004"
H5 = f"{REC}/deeperfly_outputs/results.h5"
SP = "/tmp/claude-1000/-home-tlam-deeperfly/b0dc9814-9dd3-41fb-b065-8ee87fb0fd9d/scratchpad"
LEFT, RIGHT, FRONT = ("lf", "lm", "lh"), ("rh", "rm", "rf"), ("f",)
CS, KS, REG = 6.0, 3.0, 0.3

cfg = Config.from_toml(f"{REC}/deeperfly_outputs/config.toml")
skel = cfg.skeleton()
pn = list(skel.point_names)
with h5py.File(H5, "r") as f:
    names = [x.decode() for x in f["bundle_adjustment/cameras/names"][:]]
    ba = CameraGroup.from_arrays(
        names,
        f["bundle_adjustment/cameras/rvecs"][:],
        f["bundle_adjustment/cameras/tvecs"][:],
        f["bundle_adjustment/cameras/intrs"][:],
        f["bundle_adjustment/cameras/dists"][:],
    )
    pts2d_all = f["pose2d/points"][:]
    conf_all = f["pose2d/conf"][:]
    pts3d_all = f["triangulation/points3d"][:]
V, T, P = conf_all.shape
ci = {n: i for i, n in enumerate(names)}
left_pts = left_point_indices(skel)
side = {n: ("left" if n in LEFT else "right" if n in RIGHT else "front") for n in names}
far_bones = {n: far_leg_bones(skel, side[n]) for n in names}
near_bones = {}
for n in names:  # all leg bones minus far = near, for context
    farset = {(a, b) for a, b, _ in far_bones[n]}
    nb = []
    for a, b in skel.bones:
        a, b = int(a), int(b)
        if skel.limb_names[skel.limb_id[a]].endswith("_leg") and (a, b) not in farset:
            nb.append((a, b))
    near_bones[n] = nb
cam0 = {
    n: (
        np.asarray(ba[n].rvec),
        np.asarray(ba[n].tvec),
        np.asarray(ba[n].kmat),
        np.asarray(ba[n].dist),
    )
    for n in names
}

NF = 40
sel = _subsample(T, NF, "diversity", pts2d=pts2d_all, conf=conf_all)
F = len(sel)
pts3d = pts3d_all[sel]
readers = {n: io.open_reader([f"{REC}/camera_{n.upper()}.mp4"]) for n in names}
print("building leg maps ...", flush=True)
dt_maps, scale = build_leg_maps(
    readers,
    sel,
    ba,
    np.nan_to_num(pts3d),
    skel,
    downscale=0.5,
    downscale_views={"f": 0.7},
    leg_width_px=5.0,
    left=LEFT,
    right=RIGHT,
    front=FRONT,
)
dt_maps = {n: np.asarray(v, np.float32) for n, v in dt_maps.items()}
SS = np.linspace(0.0, 1.0, 9)
TRUNC = 20.0
finite3d = np.isfinite(pts3d).all(-1)


def project_all(delta):
    moved = pts3d.copy()
    moved[:, left_pts] = rigid(pts3d[:, left_pts].reshape(-1, 3), delta).reshape(
        F, len(left_pts), 3
    )
    out = {}
    for n in names:
        r, t, k, d = cam0[n]
        if side[n] == "left":
            r, t = compensate_left(r, t, delta)
        out[n] = np.stack([project_np(moved[fi], r, t, k, d) for fi in range(F)])
    return out


def cham(proj):
    for n in names:
        dtv, s = dt_maps[n], scale[n]
        for fi in range(F):
            pr = proj[n][fi]
            for a, b, w in far_bones[n]:
                if not (finite3d[fi, a] and finite3d[fi, b]):
                    yield n, None, w
                    continue
                xy = (pr[a][None] * (1 - SS[:, None]) + pr[b][None] * SS[:, None]) * s
                yield (
                    n,
                    ndi.map_coordinates(
                        dtv[fi],
                        [xy[:, 1], xy[:, 0]],
                        order=1,
                        mode="constant",
                        cval=TRUNC,
                    ),
                    w,
                )


keep = np.array(
    [
        (d is not None) and (w > 0) and (np.median(d) <= 9.0)
        for _, d, w in cham(project_all(np.zeros(6)))
    ]
)
amask = {
    n: (np.isfinite(pts2d_all[ci[n]][sel]).all(-1) & (conf_all[ci[n]][sel] > 0))
    for n in names
}


def residual(delta):
    proj = project_all(delta)
    blocks = []
    ch = []
    for i, (n, d, w) in enumerate(cham(proj)):
        if keep[i] and d is not None:
            ch.append(w * (d / scale[n]) / CS)
    blocks.append(np.concatenate(ch) if ch else np.zeros(0))
    for n in names:
        m = amask[n] & np.isfinite(proj[n]).all(-1)
        if m.any():
            diff = (proj[n] - pts2d_all[ci[n]][sel])[m]
            blocks.append(
                (np.sqrt(conf_all[ci[n]][sel][m])[:, None] * diff / KS).ravel()
            )
    blocks.append(np.sqrt(REG) * delta)
    return np.concatenate(blocks)


res = least_squares(
    residual, np.zeros(6), method="trf", loss="linear", max_nfev=60, diff_step=1e-4
)
delta = res.x
print(
    f"delta rot={np.degrees(np.linalg.norm(delta[:3])):.2f}deg trans={np.linalg.norm(delta[3:]):.4f}"
)

# viz rig: left cams compensated; corrected pts3d: left legs moved
viz = {}
for n in names:
    r, t, k, d = cam0[n]
    if side[n] == "left":
        r, t = compensate_left(r, t, delta)
    viz[n] = (r, t, k, d)

# body-shift-in-left-views: reproject midline points on left cams, BA vs corrected
body_idx = [
    i
    for i in range(P)
    if skel.limb_id[i] < 0 or not skel.limb_names[skel.limb_id[i]].endswith("_leg")
]
shifts = []
for n in LEFT:
    for fi in range(F):
        p3 = pts3d[fi, body_idx]
        m = np.isfinite(p3).all(-1)
        if not m.any():
            continue
        pa = project_np(p3[m], *cam0[n])
        pb = project_np(p3[m], *viz[n])
        shifts.append(np.linalg.norm(pa - pb, axis=-1))
print(
    f"body-shift in LEFT views: median={np.median(np.concatenate(shifts)):.2f}px "
    f"p90={np.percentile(np.concatenate(shifts), 90):.2f}px"
)

# ---------------- render 3 frames x side cameras
RENDER_CAMS = ["rf", "rm", "lm", "lh", "f"]
frames_to_show = [0, 13, 26]  # indices into sel


def moved_pts(fi):
    m = pts3d[fi].copy()
    m[left_pts] = rigid(pts3d[fi, left_pts], delta)
    return m


def draw(img, proj, bones, color, thick=2):
    for a, b in bones:
        pa, pb = proj[a], proj[b]
        if np.isfinite(pa).all() and np.isfinite(pb).all():
            cv2.line(
                img,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                color,
                thick,
                cv2.LINE_AA,
            )


tiles = []
for fi in frames_to_show:
    row = []
    m3 = moved_pts(fi)
    for n in RENDER_CAMS:
        raw = np.asarray(readers[n][[sel[fi]]])[0]
        img = raw.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = (img * 0.55).astype(np.uint8)
        proj_ba = project_np(pts3d[fi], *cam0[n])
        proj_cr = project_np(m3, *viz[n])
        fb = [(a, b) for a, b, _ in far_bones[n]]
        draw(
            img, proj_cr, near_bones[n], (200, 120, 40), 1
        )  # near (context) dim cyan/blue
        draw(img, proj_ba, fb, (60, 60, 230), 2)  # BA far = RED
        draw(img, proj_cr, fb, (60, 220, 60), 2)  # corrected far = GREEN
        cv2.putText(img, n, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        row.append(cv2.resize(img, (480, 480)))
    tiles.append(np.hstack(row))
mont = np.vstack(tiles)
cv2.imwrite(f"{SP}/derisk2_overlay.png", mont)
print(f"wrote {SP}/derisk2_overlay.png  (RED=BA far, GREEN=corrected far, dim=near)")
