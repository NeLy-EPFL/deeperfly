"""De-risk the REDESIGNED photometric fix on the feature branch.

New design vs the regressing prototype:
  * objective: HARD (non-saturating, plain least-squares) anchor on the trusted
    keypoints -- the front-camera front/mid distal legs are the real cross-side
    bridge -- plus a strong Tikhonov, so the chamfer term can only nudge the one
    weakly-observed cross-side depth DOF. Single 6-DOF T on the left cluster.
  * apply: RIGID-MOVE-LEGS-ONLY. Body + near legs copied straight from BA; only
    the far/left-leg 3D points move by T; left cameras compensated for viz. The
    body's 3D can never move -> no scramble, by construction.

Reproduces exactly what the rewired pipeline (photometric AFTER triangulation)
will do, so a good result here == a good end-to-end result.
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
SP = "/tmp/claude-1000/-home-tlam-deeperfly/b0dc9814-9dd3-41fb-b065-8ee87fb0fd9d/scratchpad"
LEFT, RIGHT, FRONT = ("lf", "lm", "lh"), ("rh", "rm", "rf"), ("f",)

# ---------------------------------------------------------------- load
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
    pts2d_all = f["pose2d/points"][:]  # (V,T,P,2)
    conf_all = f["pose2d/conf"][:]  # (V,T,P)
    pts3d_all = f["triangulation/points3d"][:]  # (T,P,3)  <- BA-rig triangulation
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

# ---------------------------------------------------------------- sample + maps
NF = 40
sel = _subsample(T, NF, "diversity", pts2d=pts2d_all, conf=conf_all)
F = len(sel)
pts3d = pts3d_all[sel]  # (F,P,3) frozen BA structure
readers = {n: io.open_reader([f"{REC}/camera_{n.upper()}.mp4"]) for n in names}
print(f"building leg maps on {F} frames ...", flush=True)
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

# ---------------------------------------------------------------- objective (single-T)
SS = np.linspace(0.0, 1.0, 9)
TRUNC = 20.0
finite3d = np.isfinite(pts3d).all(-1)  # (F,P)


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


def chamfer_samples(proj):
    """yield (n, dt_values[9]) for every kept far-bone sample."""
    for n in names:
        dtv, s = dt_maps[n], scale[n]
        for fi in range(F):
            pr = proj[n][fi]
            for a, b, w in far_bones[n]:
                if not (finite3d[fi, a] and finite3d[fi, b]):
                    yield n, None, w
                    continue
                xy = (pr[a][None] * (1 - SS[:, None]) + pr[b][None] * SS[:, None]) * s
                d = ndi.map_coordinates(
                    dtv[fi], [xy[:, 1], xy[:, 0]], order=1, mode="constant", cval=TRUNC
                )
                yield n, d, w


# gate: drop far bones with too much distance at init (occluded / body-projected)
GATE = 9.0
init_ch = list(chamfer_samples(project_all(np.zeros(6))))
keep = np.array(
    [(d is not None) and (w > 0) and (np.median(d) <= GATE) for _, d, w in init_ch]
)
print(f"kept {keep.sum()}/{len(keep)} far-bone samples after gate", flush=True)

# anchor mask per view
amask = {
    n: (np.isfinite(pts2d_all[ci[n]][sel]).all(-1) & (conf_all[ci[n]][sel] > 0))
    for n in names
}


def residual(delta, chamfer_scale, kpt_scale, reg):
    proj = project_all(delta)
    blocks = []
    ch = []
    for i, (n, d, w) in enumerate(chamfer_samples(proj)):
        if not keep[i] or d is None:
            continue
        ch.append(w * (d / scale[n]) / chamfer_scale)  # -> raw px / scale
    blocks.append(np.concatenate(ch) if ch else np.zeros(0))
    for n in names:  # HARD anchor, all views
        m = amask[n] & np.isfinite(proj[n]).all(-1)
        if not m.any():
            continue
        diff = (proj[n] - pts2d_all[ci[n]][sel])[m]
        sw = np.sqrt(conf_all[ci[n]][sel][m])[:, None]
        blocks.append((sw * diff / kpt_scale).ravel())
    blocks.append(np.sqrt(reg) * delta)
    return np.concatenate(blocks)


def percam_chamfer(delta):
    proj = project_all(delta)
    out = {}
    for i, (n, d, w) in enumerate(chamfer_samples(proj)):
        if keep[i] and d is not None:
            out.setdefault(n, []).append(d / scale[n])  # raw px
    return {n: float(np.concatenate(v).mean()) for n, v in out.items()}


def anchor_rms(delta):
    """per-view RMS keypoint reprojection error (raw px), for the no-regress guard."""
    proj = project_all(delta)
    out = {}
    for n in names:
        m = amask[n] & np.isfinite(proj[n]).all(-1)
        e = np.linalg.norm((proj[n] - pts2d_all[ci[n]][sel])[m], axis=-1)
        out[n] = float(np.sqrt((e**2).mean()))
    return out


# ---------------------------------------------------------------- solve + report a small sweep
def geodesic_deg(rvec):
    return np.degrees(np.linalg.norm(rvec))


ch0 = percam_chamfer(np.zeros(6))
an0 = anchor_rms(np.zeros(6))
print("\n=== INIT (BA) far-leg chamfer (raw px) ===")
for n in names:
    if n in ch0:
        print(f"  {n}: chamfer={ch0[n]:5.2f}  anchorRMS={an0[n]:5.2f}")

for cs, ks, rg in [(6.0, 3.0, 0.3), (6.0, 3.0, 1.0), (6.0, 2.0, 0.1), (8.0, 4.0, 0.3)]:
    res = least_squares(
        residual,
        np.zeros(6),
        args=(cs, ks, rg),
        method="trf",
        loss="linear",
        max_nfev=60,
        diff_step=1e-4,
    )
    d = res.x
    ch1 = percam_chamfer(d)
    an1 = anchor_rms(d)
    dch = np.mean([ch1[n] - ch0[n] for n in ch1])
    dan = np.mean([an1[n] - an0[n] for n in names])
    dan_left = np.mean([an1[n] - an0[n] for n in LEFT])
    print(
        f"\n--- cs={cs} ks={ks} reg={rg}: rot={geodesic_deg(d[:3]):.2f}deg "
        f"trans={np.linalg.norm(d[3:]):.4f}  Dchamfer={dch:+.2f}px  "
        f"Danchor(all)={dan:+.2f}px Danchor(left)={dan_left:+.2f}px  status={res.status}"
    )
    for n in names:
        tag = "far" if n in ch1 else "   "
        c = f"{ch0.get(n, 0):4.1f}->{ch1.get(n, 0):4.1f}" if n in ch1 else "  -       "
        print(f"    {n} {tag}: chamfer {c}   anchorRMS {an0[n]:4.1f}->{an1[n]:4.1f}")
