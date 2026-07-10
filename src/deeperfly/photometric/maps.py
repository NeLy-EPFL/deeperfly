"""Build per-view truncated distance-transform maps from footage.

For each camera and sampled frame: read the raw frame, downscale to a working
resolution, compute the :func:`~deeperfly.photometric.leg_response.leg_response`
legness map, **paint out the near (same-side) legs** using their accurate reprojection
(so far bones can only match unexplained far-leg response), threshold, and take the
:func:`~deeperfly.photometric.distance.truncated_dt`. Only the compact DT maps are kept.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..cameras import CameraGroup
from .distance import truncated_dt
from .leg_response import leg_response, paint_out_bones
from .objective import far_leg_bones, project_np
from .refine import DEFAULT_FRONT, DEFAULT_LEFT, DEFAULT_RIGHT, _cam_side


def _near_leg_bones(skeleton, cam_side):
    """Leg bones NOT in the far set (the near/other legs to paint out)."""
    far = {(a, b) for a, b, _ in far_leg_bones(skeleton, cam_side)}
    out = []
    for a, b in skeleton.bones:
        a, b = int(a), int(b)
        if (
            skeleton.limb_names[skeleton.limb_id[a]].endswith("_leg")
            and (a, b) not in far
        ):
            out.append((a, b))
    return out


def build_leg_maps(
    readers: dict[str, object],
    frame_indices,
    cameras: CameraGroup,
    pts3d,
    skeleton,
    *,
    downscale: float = 0.5,
    downscale_views: dict[str, float] | None = None,
    method: str = "ridge_fg",
    leg_width_px: float = 5.0,
    polarity: str = "auto",
    response_threshold: float = 0.15,
    trunc_px: float = 20.0,
    paint_out: bool = True,
    left=DEFAULT_LEFT,
    right=DEFAULT_RIGHT,
    front=DEFAULT_FRONT,
    dtype=np.float16,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Return ``(dt_maps, scale)``: per-view ``(F, Hv, Wv)`` truncated DT and raw->working scale.

    ``pts3d`` are the 3D joints on the *same* ``frame_indices`` (used for near-leg
    paint-out at the input rig). ``downscale_views`` overrides ``downscale`` per camera
    (e.g. keep the front camera less downscaled so far legs stay a few working pixels).
    """
    dv = downscale_views or {}
    dt_maps, scale = {}, {}
    idx = list(frame_indices)
    for n, reader in readers.items():
        s = float(dv.get(n, downscale))
        frames = np.asarray(reader[idx])  # (F, H, W, 3) uint8
        F, H, W = frames.shape[:3]
        Hv, Wv = int(round(H * s)), int(round(W * s))
        small = np.stack(
            [cv2.resize(f, (Wv, Hv), interpolation=cv2.INTER_AREA) for f in frames]
        )
        resp = leg_response(
            small, method=method, leg_width_px=leg_width_px, polarity=polarity
        )
        side = _cam_side(n, left, right, front)
        near = _near_leg_bones(skeleton, side) if paint_out else []
        cam = cameras[n]
        maps = np.empty((F, Hv, Wv), dtype)
        for f in range(F):
            rr = resp[f]
            if near:
                proj = project_np(pts3d[f], cam.rvec, cam.tvec, cam.kmat, cam.dist) * s
                segs = np.stack([[proj[a], proj[b]] for a, b in near])  # (B,2,2)
                rr = paint_out_bones(rr, segs, width_px=2.0 * leg_width_px)
            maps[f] = truncated_dt(
                rr, threshold=response_threshold, trunc_px=trunc_px
            ).astype(dtype)
        dt_maps[n] = maps
        scale[n] = s
    return dt_maps, scale
