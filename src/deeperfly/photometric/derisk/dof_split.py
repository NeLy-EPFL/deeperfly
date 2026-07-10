"""Which DOF buys the far-leg gain, and what body-shift does each cost?

Solve the single-T objective restricted to (a) full 6-DOF, (b) translation-only,
(c) rotation-only. For each report: rotation, translation, mean far-leg chamfer
gain, and the body-shift it induces in the left views (the cost).
"""

from __future__ import annotations

import sys

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
LEFT, RIGHT, FRONT = ("lf", "lm", "lh"), ("rh", "rm", "rf"), ("f",)
CS, KS, REG = 6.0, 3.0, 0.3

cfg = Config.from_toml(f"{REC}/deeperfly_outputs/config.toml")
skel = cfg.skeleton()
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
cam0 = {
    n: (
        np.asarray(ba[n].rvec),
        np.asarray(ba[n].tvec),
        np.asarray(ba[n].kmat),
        np.asarray(ba[n].dist),
    )
    for n in names
}
body_idx = [
    i
    for i in range(P)
    if skel.limb_id[i] < 0 or not skel.limb_names[skel.limb_id[i]].endswith("_leg")
]

sel = _subsample(T, 40, "diversity", pts2d=pts2d_all, conf=conf_all)
F = len(sel)
pts3d = pts3d_all[sel]
readers = {n: io.open_reader([f"{REC}/camera_{n.upper()}.mp4"]) for n in names}
print("building maps ...", flush=True)
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


def mean_far_chamfer(delta):
    proj = project_all(delta)
    vals = []
    for i, (n, d, w) in enumerate(cham(proj)):
        if keep[i] and d is not None:
            vals.append((d / scale[n]).mean())
    return float(np.mean(vals))


def body_shift(delta):
    vizshift = []
    for n in LEFT:
        r, t, k, d = cam0[n]
        rc, tc = compensate_left(r, t, delta)
        for fi in range(F):
            p3 = pts3d[fi, body_idx]
            m = np.isfinite(p3).all(-1)
            if m.any():
                vizshift.append(
                    np.linalg.norm(
                        project_np(p3[m], r, t, k, d) - project_np(p3[m], rc, tc, k, d),
                        axis=-1,
                    )
                )
    return float(np.median(np.concatenate(vizshift)))


def make_resid(free):
    def resid(x):
        delta = np.zeros(6)
        delta[free] = x
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
        blocks.append(np.sqrt(REG) * delta[free])
        return np.concatenate(blocks)

    return resid


base = mean_far_chamfer(np.zeros(6))
print(f"\nINIT mean far-leg chamfer = {base:.2f} raw px\n")
for label, free in [
    ("full 6-DOF", [0, 1, 2, 3, 4, 5]),
    ("translation-only", [3, 4, 5]),
    ("rotation-only", [0, 1, 2]),
]:
    r = least_squares(
        make_resid(free),
        np.zeros(len(free)),
        method="trf",
        loss="linear",
        max_nfev=60,
        diff_step=1e-4,
    )
    delta = np.zeros(6)
    delta[free] = r.x
    print(
        f"{label:18s}: rot={np.degrees(np.linalg.norm(delta[:3])):5.2f}deg trans={np.linalg.norm(delta[3:]):.4f} "
        f"| far chamfer {base:.2f}->{mean_far_chamfer(delta):.2f}px | body-shift(left views)={body_shift(delta):5.2f}px"
    )
