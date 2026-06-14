"""Pose the bundled NeuroMechFly mesh to a fitted pose, for the 2D overlay.

The mesh overlay reuses the simplified NeuroMechFly model the docs keypoint viewer
ships, baked into a runtime-only asset (``data/nmf_mesh.npz`` -- see
``scripts/build_nmf_mesh_asset.py``). There is no MuJoCo/flygym at runtime: the
mesh is posed straight from the inverse-kinematics result's fitted joint positions
(``PoseResult.nmf_pts3d``, skeleton point order) by skinning each segment between
its two endpoint keypoints.

- Each **leg** mesh (coxa / trochanter+femur / tibia / tarsus) is a bone between
  two tracked keypoints; it is mapped from its neutral endpoints onto the live
  endpoints by a per-bone similarity transform, with the roll about the bone fixed
  by the body's dorsal axis so it stays stable as the leg swings.
- The rigid **body** (thorax / head / abdomen / wings / antennae) is placed by one
  global similarity transform fit (Umeyama) from the six thorax-coxa keypoints,
  which are body-fixed.

:func:`NmfMesh.pose` returns world-space vertices for one frame (NaN for segments
whose endpoints are missing); the rasterizer in :mod:`deeperfly.visualization.mesh`
projects and shades them per view.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from jaxtyping import Float, Int

__all__ = ["NmfMesh", "load_nmf_mesh"]

#: The packaged baked mesh asset (built by ``scripts/build_nmf_mesh_asset.py``).
DEFAULT_MESH_PATH = Path(__file__).parent.parent / "data" / "nmf_mesh.npz"

_EPS = 1e-9


@dataclass(frozen=True)
class NmfMesh:
    """The baked NeuroMechFly overlay mesh and the data to pose it from a fit.

    Attributes
    ----------
    vertices
        ``(Nv, 3)`` neutral-pose vertices in the model's world frame.
    faces
        ``(Nf, 3)`` triangle vertex indices into ``vertices``.
    face_rgb
        ``(Nf, 3)`` uint8 per-face color.
    vert_slot
        ``(Nv,)`` the render slot of each vertex: ``0`` is the rigid body, ``>= 1``
        are leg bones.
    slot_prox, slot_dist
        ``(n_slots,)`` the proximal / distal keypoint index of each leg bone
        (``-1`` for the body slot), into the skeleton's point order.
    kp_neutral
        ``(P, 3)`` the model's neutral keypoint positions (skeleton order).
    coxa_idx
        ``(6,)`` the thorax-coxa keypoint indices that anchor the body transform.
    """

    vertices: Float[np.ndarray, "Nv 3"]
    faces: Int[np.ndarray, "Nf 3"]
    face_rgb: np.ndarray
    vert_slot: Int[np.ndarray, "Nv"]
    slot_prox: Int[np.ndarray, "S"]
    slot_dist: Int[np.ndarray, "S"]
    kp_neutral: Float[np.ndarray, "P 3"]
    coxa_idx: Int[np.ndarray, "6"]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MESH_PATH) -> "NmfMesh":
        z = np.load(Path(path), allow_pickle=True)
        return cls(
            vertices=z["vertices"].astype(float),
            faces=z["faces"].astype(np.int64),
            face_rgb=z["face_rgb"],
            vert_slot=z["vert_slot"].astype(np.int64),
            slot_prox=z["slot_prox"].astype(np.int64),
            slot_dist=z["slot_dist"].astype(np.int64),
            kp_neutral=z["kp_neutral"].astype(float),
            coxa_idx=z["coxa_idx"].astype(np.int64),
        )

    @functools.cached_property
    def _slot_verts(self) -> list[np.ndarray]:
        """Vertex-row indices for each slot (so posing is a few vectorized blocks)."""
        n = int(self.slot_prox.shape[0])
        return [np.flatnonzero(self.vert_slot == s) for s in range(n)]

    def pose(
        self, pts3d: Float[np.ndarray, "P 3"]
    ) -> tuple[Float[np.ndarray, "Nv 3"], np.ndarray]:
        """Pose the mesh to one frame's fitted joints.

        Parameters
        ----------
        pts3d
            The fitted model joints for one frame ``(P, 3)`` in world coordinates
            (skeleton point order; NaN where a joint was not solved).

        Returns
        -------
        verts : np.ndarray
            ``(Nv, 3)`` posed world vertices (NaN for un-poseable segments).
        valid_faces : np.ndarray
            ``(Nf,)`` bool mask of faces whose three vertices are all finite.
        """
        pts3d = np.asarray(pts3d, dtype=float)
        out = np.full_like(self.vertices, np.nan)

        rot, scale, trans = self._body_transform(pts3d)
        up = rot @ np.array([0.0, 0.0, 1.0])  # live dorsal axis fixes the bone roll

        for slot, rows in enumerate(self._slot_verts):
            if rows.size == 0:
                continue
            v = self.vertices[rows]
            if slot == 0:  # rigid body
                out[rows] = scale * (v @ rot.T) + trans
                continue
            a0 = self.kp_neutral[self.slot_prox[slot]]
            b0 = self.kp_neutral[self.slot_dist[slot]]
            a1 = pts3d[self.slot_prox[slot]]
            b1 = pts3d[self.slot_dist[slot]]
            if not (np.isfinite(a1).all() and np.isfinite(b1).all()):
                continue  # an occluded leg segment stays NaN -> dropped from the draw
            out[rows] = _skin_bone(v, a0, b0, a1, b1, up)

        valid_faces = np.isfinite(out[self.faces]).all(axis=(1, 2))
        return out, valid_faces

    def _body_transform(
        self, pts3d: Float[np.ndarray, "P 3"]
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Global similarity (R, s, t) from the neutral coxae to the live coxae."""
        src = self.kp_neutral[self.coxa_idx]
        dst = pts3d[self.coxa_idx]
        good = np.isfinite(dst).all(axis=1)
        if int(good.sum()) < 3:
            return np.eye(3), 1.0, np.zeros(3)  # too few anchors to place the body
        return _umeyama(src[good], dst[good])


@functools.lru_cache(maxsize=2)
def load_nmf_mesh(path: str | Path = DEFAULT_MESH_PATH) -> NmfMesh:
    """Load (and cache) the packaged NeuroMechFly overlay mesh."""
    return NmfMesh.load(path)


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > _EPS else v


def _bone_basis(axis: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Orthonormal frame (columns) whose z is ``axis`` and whose roll follows ``up``."""
    z = _normalize(axis)
    x = np.cross(up, z)
    if np.linalg.norm(x) < _EPS:  # up parallel to the bone: any perpendicular will do
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        if np.linalg.norm(x) < _EPS:
            x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x = _normalize(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _skin_bone(
    v: np.ndarray, a0: np.ndarray, b0: np.ndarray, a1: np.ndarray, b1: np.ndarray, up
) -> np.ndarray:
    """Map vertices ``v`` from the neutral bone (a0->b0) onto the live bone (a1->b1)."""
    d0, d1 = b0 - a0, b1 - a1
    len0, len1 = np.linalg.norm(d0), np.linalg.norm(d1)
    if len0 < _EPS:
        return np.full_like(v, np.nan)
    rot = _bone_basis(d1, up) @ _bone_basis(d0, np.array([0.0, 0.0, 1.0])).T
    return a1 + (len1 / len0) * ((v - a0) @ rot.T)


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-squares similarity transform (rotation, scale, translation) ``src -> dst``."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (d0.T @ s0) / len(src)
    u, sigma, vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[2, 2] = -1.0
    rot = u @ correction @ vt
    var = (s0**2).sum() / len(src)
    scale = float(np.trace(np.diag(sigma) @ correction) / var) if var > _EPS else 1.0
    trans = mu_d - scale * (rot @ mu_s)
    return rot, scale, trans
