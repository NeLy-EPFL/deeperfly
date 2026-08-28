"""A synthetic dorsal plan-view camera, fitted to the animal rather than to the rig.

The rig has no camera above the animal, and on this preparation it never will: the
tether comes down from there. But a plan view is the one viewpoint that shows all six
legs at once without a body in the way, which makes it the panel a person actually reads
a gait from. So it is *derived* -- from the reconstructed 3D, not from the calibration.

**Why not a `[cameras.bird]` orbit entry.** Two reasons, and the second is the real one.
An orbit spec cannot express raw ``rvec``/``tvec``, so the viewpoint would have to be
written as an azimuth/elevation against the world axes -- but the world frame is whatever
bundle adjustment converged to, and "above the animal" is a fact about the *animal*. The
body's dorsal direction moves between recordings (a different tether angle) while the
world frame does not, so a fixed elevation would point at the animal's back in one
recording and at its flank in the next. Deriving the axes from the animal makes the panel
mean the same thing everywhere.

**The chirality is measured, never assumed.** Looking down at a back and looking up at a
belly differ by a mirror, and a mirrored plan view is not obviously wrong on screen -- it
just quietly swaps the animal's left and right legs. The sign of the dorsal axis is
therefore settled by the pretarsi: they are the feet, so the body must lie on the positive
side of the plane through them. Nothing here trusts a cross-product's sign by itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..cameras import Camera
    from ..skeleton import Skeleton

__all__ = ["BIRD_VIEW", "dorsal_camera"]

#: The reserved panel ``view`` name that resolves to this synthetic camera. A rig camera
#: of the same name would win, which is the right precedence: a real view beats a
#: derived one.
BIRD_VIEW: str = "bird"

#: The synthetic frame's ``(height, width)``. Only the aspect matters -- a panel's
#: ``width``/``height`` rescales it -- and 2:1 matches the montage cells.
_IMAGE_HW: tuple[int, int] = (240, 480)

#: How much bigger than the fitted extent the frame is, so the animal does not touch
#: the edges.
_MARGIN: float = 1.06

#: Percentile trimmed off each end when fitting that extent. Cells clip cleanly, so it
#: is worth letting the rare fully-extended leg leave the panel to keep the animal large.
_PCT: float = 0.4

#: Distance, in multiples of the animal's own radius, to put the camera back. The real
#: cameras sit at focal ~22000 over distance ~107, i.e. essentially orthographic; this
#: matches that so the panel reads as a plan view and not as a wide-angle one.
_STANDOFF: float = 100.0


def _abdomen_tip(names: list[str]) -> list[int]:
    """The most posterior abdomen point(s), for either skeleton generation.

    ``fly38`` has one midline chain (``abdomen0..4``) whose tip is the highest index. A
        two-side-chain abdomen (``l_abdomen0..2`` / ``r_abdomen0..2``, as the DeepFly3D set
        had) has its tip at the highest index on *both* sides, whose centroid is back on the
        midline. Keying on the trailing number rather than on position in the list covers
        either without the caller having to say which it has, which is why the generality is
        kept now that only one skeleton ships: it costs a dict and it is what lets a project
        bring its own abdomen.
    """
    ranked: dict[int, list[int]] = {}
    for i, n in enumerate(names):
        if "abdomen" not in n:
            continue
        digits = "".join(c for c in n if c.isdigit())
        ranked.setdefault(int(digits) if digits else -1, []).append(i)
    return ranked[max(ranked)] if ranked else []


def _unit(v: np.ndarray, what: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError(
            f"cannot place the dorsal view: its {what} axis is degenerate. The 3D pose "
            "has to have a resolved body, so check the triangulation before this panel."
        )
    return np.asarray(v, dtype=float) / n


def _orientation_groups(skeleton: "Skeleton") -> dict[str, list[int]]:
    """The anatomical groups the body frame is built from, resolved BY NAME.

    Renamed from ``_landmarks`` in 0.3.0: calibration landmarks were removed, and the word
    now means only one thing (a chain's base marker in the IK stage), so a local helper
    that meant a third thing had to stop colliding with both.

    Resolved by name rather than by index so a skeleton change cannot silently
    re-point them -- this is exactly the region the DeepFly3D set and ``fly38`` differ in
    (the abdomen went from two side chains to one midline chain, and gained a neck).
    """
    names = list(skeleton.point_names)
    groups = {
        "anterior": [
            i for i, n in enumerate(names) if n in ("neck", "l_antenna", "r_antenna")
        ],
        "posterior": _abdomen_tip(names),
        "left_coxa": [
            i
            for i, n in enumerate(names)
            if n.startswith("l") and n.endswith("_thorax_coxa")
        ],
        "right_coxa": [
            i
            for i, n in enumerate(names)
            if n.startswith("r") and n.endswith("_thorax_coxa")
        ],
        "pretarsi": [i for i, n in enumerate(names) if n.endswith("_pretarsus")],
    }
    missing = [k for k, v in groups.items() if not v]
    if missing:
        raise ValueError(
            f"this skeleton has no points for {missing}, so the dorsal plan view cannot "
            "be oriented. It needs an anterior group (neck/antenna), an abdomen "
            "point, both sides' thorax-coxa joints, and pretarsi (which is what fixes "
            f"dorsal from ventral). Points are: {names}"
        )
    return groups


def dorsal_camera(
    pts3d: np.ndarray,
    skeleton: "Skeleton",
    *,
    like: "Camera | None" = None,
    image_hw: tuple[int, int] = _IMAGE_HW,
    margin: float = _MARGIN,
    pct: float = _PCT,
) -> "Camera":
    """A camera looking straight down the animal's dorsal axis, framed on the clip.

    A *median* pose over the whole clip gives three anatomical axes -- ``ap`` anterior
    (abdomen tip to neck/antennae), ``lat`` the animal's left (right coxae to left
    coxae), and ``up`` dorsal (``ap x lat``, its sign settled by the pretarsi). The camera
    is placed far back along ``up`` and its focal solved so the animal's own extent fills
    the frame. Anterior points to the top of the panel, which puts the animal's left on
    the left -- what you see looking down at its back.

    The framing is fitted ONCE over every frame, not per frame: a per-frame fit would
    make the panel breathe, and a leg sweep would read as the body moving.

    Parameters
    ----------
    pts3d
        ``(T, P, 3)`` world-coordinate 3D pose.
    skeleton
        The skeleton naming those points (see :func:`_orientation_groups`).
    like
        A rig camera to copy the distortion *shape* from; the synthetic camera is
        distortion-free, and this only keeps the coefficient vector the same length so
        it sits in a :class:`~deeperfly.cameras.CameraGroup` beside the real ones.
    image_hw, margin, pct
        The synthetic frame size, the air left around the animal, and the percentile
        trimmed off each end of its extent.

    Returns
    -------
    Camera
        Named :data:`BIRD_VIEW`.
    """
    import cv2

    from ..cameras import Camera

    pts3d = np.asarray(pts3d, dtype=float)
    if pts3d.ndim != 3 or pts3d.shape[-1] != 3:
        raise ValueError(f"pts3d must be (T, P, 3), got {pts3d.shape}")
    g = _orientation_groups(skeleton)

    ref = np.nanmedian(pts3d, axis=0)  # (P, 3): the clip's median pose
    if not np.isfinite(ref).any():
        raise ValueError("the 3D pose is entirely NaN; nothing to orient a view on")

    def centroid(idx):
        sel = ref[idx]
        sel = sel[np.isfinite(sel).all(-1)]
        if not len(sel):
            raise ValueError(
                "an orientation group is NaN in the median pose, so the dorsal view cannot "
                "be oriented; the animal is not resolved in 3D"
            )
        return sel.mean(0)

    ap = _unit(centroid(g["anterior"]) - centroid(g["posterior"]), "anterior")
    lat = centroid(g["left_coxa"]) - centroid(g["right_coxa"])
    lat = _unit(lat - float(lat @ ap) * ap, "lateral")  # orthogonalize against ap
    up = _unit(np.cross(ap, lat), "dorsal")
    thorax = np.concatenate([g["left_coxa"], g["right_coxa"]])
    if float(up @ (centroid(thorax) - centroid(g["pretarsi"]))) < 0:
        up = -up  # the pretarsi are the feet, so the body is dorsal of them
    center = centroid(np.concatenate([thorax, g["posterior"]]))

    # OpenCV cameras look along +z_cam. Anterior at the TOP of the panel means the
    # image's DOWN (+y_cam) is -ap; x_cam = y x z then keeps the frame right-handed.
    z, y = -up, -ap
    x = _unit(np.cross(y, z), "image-x")
    rmat = np.stack([x, y, z])

    finite = pts3d[np.isfinite(pts3d).all(axis=-1)]
    span = float(np.percentile(np.linalg.norm(finite - center, axis=-1), 99))
    distance = _STANDOFF * span
    position = center + up * distance
    local = (finite - position) @ rmat.T

    # Solve the focal AND the principal point from the content's own bounds in
    # normalized (x/z, y/z) coordinates, so the animal comes out both centred and as
    # large as the frame allows.
    h, w = image_hw
    nx, ny = local[:, 0] / local[:, 2], local[:, 1] / local[:, 2]
    lo_x, hi_x = np.percentile(nx, [pct, 100 - pct])
    lo_y, hi_y = np.percentile(ny, [pct, 100 - pct])
    mid_x, mid_y = (lo_x + hi_x) / 2.0, (lo_y + hi_y) / 2.0
    half_x = max((hi_x - lo_x) / 2.0, 1e-9)
    half_y = max((hi_y - lo_y) / 2.0, 1e-9)
    focal = min(w / (2 * half_x * margin), h / (2 * half_y * margin))

    n_dist = 0 if like is None else len(np.asarray(like.dist, dtype=float))
    return Camera(
        rvec=cv2.Rodrigues(rmat)[0].ravel().astype(float),
        tvec=(-rmat @ position).astype(float),
        intr=np.array(
            [
                focal,
                focal,
                (w - 1) / 2.0 - focal * mid_x,
                (h - 1) / 2.0 - focal * mid_y,
            ],
            dtype=float,
        ),
        dist=np.zeros(n_dist, dtype=float),
        name=BIRD_VIEW,
    )
