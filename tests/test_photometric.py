"""Tests for the photometric cross-side refinement.

Self-contained: builds a synthetic rig from the packaged config and renders far-leg
lines, so no footage or ``results.h5`` is needed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from deeperfly.config import Config
from deeperfly.photometric import (
    compensate_left,
    far_leg_bones,
    left_point_indices,
    leg_response,
    refine_extrinsics_photometric,
    rigid,
    truncated_dt,
)
from deeperfly.photometric.objective import project_np

IMG_SIZES = {  # name -> (H, W)
    "rh": (512, 960),
    "rm": (512, 960),
    "rf": (512, 960),
    "f": (1008, 1600),
    "lf": (512, 960),
    "lm": (512, 960),
    "lh": (512, 960),
}


# ---------- leg-response / distance transform ----------
def test_ridge_peaks_on_centerline_and_dt():
    img = np.zeros((80, 120, 3), np.uint8)
    cv2.line(img, (10, 40), (110, 40), (255, 255, 255), 3)  # bright horizontal line
    resp = leg_response(img[None], method="ridge", leg_width_px=3.0, polarity="bright")[
        0
    ]
    # response peaks on the line centerline (row 40), not 8px away
    assert resp[40, 60] > resp[48, 60] + 0.1
    dt = truncated_dt(resp, threshold=0.2, trunc_px=15.0)
    assert dt[40, 60] < 1.5  # ~0 on the line
    assert dt[40, 60] < dt[55, 60]  # grows away from the line
    assert dt.max() <= 15.0 + 1e-5  # truncated


def test_truncated_dt_empty_is_constant():
    dt = truncated_dt(np.zeros((30, 30), np.float32), threshold=0.2, trunc_px=12.0)
    assert np.allclose(dt, 12.0)


def test_auto_polarity_picks_bright():
    img = np.full((80, 120, 3), 20, np.uint8)  # dark background
    cv2.line(img, (10, 40), (110, 40), (240, 240, 240), 3)  # bright leg
    ridge_only = leg_response(
        img[None], method="ridge", leg_width_px=3.0, polarity="auto"
    )[0]
    assert ridge_only[40, 60] > 0.2


# ---------- SE(3) parameterization ----------
def test_intra_left_reprojection_invariant():
    """A left camera imaging a moved left point is unchanged for any delta."""
    cfg = Config.default()
    cg = cfg.camera_group(image_sizes=IMG_SIZES)
    p = np.array([0.3, -0.2, 0.5])
    cam = cg["lf"]
    delta = np.array([0.02, -0.03, 0.01, 0.4, -0.2, 0.3])
    rv, tv = compensate_left(cam.rvec, cam.tvec, delta)
    base = project_np(p[None], cam.rvec, cam.tvec, cam.kmat, cam.dist)
    moved = project_np(rigid(p[None], delta), rv, tv, cam.kmat, cam.dist)
    assert np.allclose(base, moved, atol=1e-6)


def test_far_bones_selected_by_side():
    skel = Config.default().skeleton()
    # a right camera's far bones are all left legs
    for a, b, w in far_leg_bones(skel, "right"):
        assert skel.limb_names[skel.limb_id[a]].startswith("l")
    # front camera's far set is the left legs (the T-dependent ones)
    for a, b, w in far_leg_bones(skel, "front"):
        assert skel.limb_names[skel.limb_id[a]].startswith("l")


# ---------- end-to-end synthetic recovery ----------
def _render_dt(cg, pts3d, skel, scale, trunc_px):
    dt, sc = {}, {}
    for n in cg.names:
        H, W = IMG_SIZES[n]
        Hv, Wv = int(H * scale), int(W * scale)
        cam = cg[n]
        maps = np.empty((len(pts3d), Hv, Wv), np.float32)
        for f in range(len(pts3d)):
            proj = project_np(pts3d[f], cam.rvec, cam.tvec, cam.kmat, cam.dist) * scale
            img = np.zeros((Hv, Wv), np.uint8)
            for a, b in skel.bones:
                a, b = int(a), int(b)
                if not skel.limb_names[skel.limb_id[a]].endswith("_leg"):
                    continue
                pa, pb = proj[a], proj[b]
                if np.isfinite(pa).all() and np.isfinite(pb).all():
                    cv2.line(
                        img,
                        tuple(pa.astype(int)),
                        tuple(pb.astype(int)),
                        255,
                        3,
                        cv2.LINE_AA,
                    )
            maps[f] = truncated_dt(
                (img > 40).astype(np.float32), threshold=0.5, trunc_px=trunc_px
            )
        dt[n], sc[n] = maps, scale
    return dt, sc


def test_recovers_perturbed_left_cluster():
    rng = np.random.default_rng(0)
    cfg = Config.default()
    cg = cfg.camera_group(image_sizes=IMG_SIZES)
    skel = cfg.skeleton()
    P = len(skel.point_names)
    F = 12
    # synthetic 3D joints in a small volume around the origin (project near frame center)
    pts3d = rng.uniform(-1.2, 1.2, size=(F, P, 3))
    scale, trunc = 0.5, 20.0
    dt_gt, sc = _render_dt(cg, pts3d, skel, scale, trunc)

    # perturb the left cluster (cameras + points) by a known rigid T_err
    T_err = np.array([0.008, 0.012, 0.005, 0.35, -0.25, 0.20])
    left_pts = left_point_indices(skel)
    pts_mis = pts3d.copy()
    pts_mis[:, left_pts] = rigid(pts3d[:, left_pts].reshape(-1, 3), T_err).reshape(
        F, len(left_pts), 3
    )
    rvecs, tvecs = cg.rvecs.copy(), cg.tvecs.copy()
    from deeperfly.cameras import CameraGroup

    names = list(cg.names)
    for i, n in enumerate(names):
        if n in ("lf", "lm", "lh"):
            rvecs[i], tvecs[i] = compensate_left(rvecs[i], tvecs[i], T_err)
    cg_mis = CameraGroup.from_arrays(names, rvecs, tvecs, cg.intrs, cg.dists)

    res = refine_extrinsics_photometric(
        cg_mis,
        pts_mis,
        dt_gt,
        sc,
        skel,
        gate_px=None,
        reg=1e-3,
        chamfer_scale=trunc,
        samples_per_bone=9,
        max_nfev=60,
    )
    before = np.mean(list(res.chamfer_before.values()))
    after = np.mean(list(res.chamfer_after.values()))
    assert after < 0.35 * before  # chamfer collapses

    # recovered left-camera pose matches GT
    for n in ("lf", "lm", "lh"):
        gt = cg[n]
        rec = res.cameras[n]
        dR = np.degrees(
            np.linalg.norm(
                cv2.Rodrigues(cv2.Rodrigues(rec.rvec)[0] @ cv2.Rodrigues(gt.rvec)[0].T)[
                    0
                ]
            )
        )
        assert dR < 1.0
        assert np.linalg.norm(rec.tvec - gt.tvec) < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
