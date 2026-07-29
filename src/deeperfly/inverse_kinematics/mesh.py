"""Pose the bundled NeuroMechFly mesh to a fitted pose, for the 2D overlay.

The mesh overlay reuses the simplified NeuroMechFly model the docs keypoint viewer
ships, baked into a runtime-only asset (``data/nmf_mesh.npz`` -- see
``scripts/build_nmf_mesh_asset.py``). There is no MuJoCo/flygym at runtime: the
mesh is posed straight from the inverse-kinematics result's fitted joint positions
(``PoseResult.nmf_pts3d``) and chain angles (``PoseResult.nmf_angles``), each mesh
in its own "slot":

- Each **leg** mesh (coxa / trochanter+femur / tibia / tarsus) is a bone between
  two tracked keypoints; it is mapped from its neutral endpoints onto the live
  endpoints by a per-bone similarity transform, with the roll about the bone fixed
  by the body's dorsal axis so it stays stable as the leg swings.
- The **head** and each **abdomen** segment are articulated nodes: a node mesh is
  carried by its head/abdomen chain forward kinematics at the node's depth (from the
  fitted angles), so the overlay head turns and abdomen curls with the fit. Without
  angles a node stays at its neutral pose on the rigid body.
- The rigid **body** (thorax / wings / halteres) is placed by one global similarity
  transform fit (Umeyama) from the six thorax-coxa keypoints, which are body-fixed.
  That same transform also registers the articulated nodes.

:func:`NmfMesh.pose` returns world-space vertices for one frame (NaN for segments
whose endpoints are missing); the rasterizer in :mod:`deeperfly.visualization.mesh`
projects and shades them per view.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
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
        (``-1`` for the body / articulated-node slots), into the skeleton's order.
    slot_chain, slot_depth
        ``(n_slots,)`` the articulation chain index and chain depth of each node
        slot (head / abdomen-segment meshes), or ``-1`` for body / leg-bone slots.
    kp_neutral
        ``(P, 3)`` the model's neutral keypoint positions (skeleton order).
    coxa_idx
        ``(6,)`` the thorax-coxa keypoint indices that anchor the body transform.
    vert_part
        ``(Nv,)`` each vertex's body-part index into :attr:`part_names` (so the
        overlay can hide whole parts -- e.g. the wings -- per the render config).
    part_names
        The body-part names, indexed by :attr:`vert_part` (``"wings"``, ``"legs"``,
        ``"head"``, ``"thorax"``, ``"abdomen"``, ``"eyes"``, ``"antennae"``,
        ``"halteres"``).
    """

    vertices: Float[np.ndarray, "Nv 3"]
    faces: Int[np.ndarray, "Nf 3"]
    face_rgb: np.ndarray
    vert_slot: Int[np.ndarray, "Nv"]
    slot_prox: Int[np.ndarray, "S"]
    slot_dist: Int[np.ndarray, "S"]
    slot_chain: Int[np.ndarray, "S"]
    slot_depth: Int[np.ndarray, "S"]
    kp_neutral: Float[np.ndarray, "P 3"]
    coxa_idx: Int[np.ndarray, "6"]
    vert_part: Int[np.ndarray, "Nv"]
    part_names: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path = DEFAULT_MESH_PATH) -> "NmfMesh":
        z = np.load(Path(path), allow_pickle=True)
        n_slots = z["slot_prox"].shape[0]
        nofill = np.full(n_slots, -1, dtype=np.int64)
        n_verts = z["vertices"].shape[0]
        return cls(
            vertices=z["vertices"].astype(float),
            faces=z["faces"].astype(np.int64),
            face_rgb=z["face_rgb"],
            vert_slot=z["vert_slot"].astype(np.int64),
            slot_prox=z["slot_prox"].astype(np.int64),
            slot_dist=z["slot_dist"].astype(np.int64),
            slot_chain=(
                z["slot_chain"].astype(np.int64) if "slot_chain" in z else nofill
            ),
            slot_depth=(
                z["slot_depth"].astype(np.int64) if "slot_depth" in z else nofill
            ),
            kp_neutral=z["kp_neutral"].astype(float),
            coxa_idx=z["coxa_idx"].astype(np.int64),
            vert_part=(
                z["vert_part"].astype(np.int64)
                if "vert_part" in z
                else np.zeros(n_verts, dtype=np.int64)
            ),
            part_names=(
                tuple(str(p) for p in z["part_names"]) if "part_names" in z else ()
            ),
        )

    def hidden_face_mask(self, hide_parts: "Sequence[str]") -> np.ndarray:
        """``(Nf,)`` bool: faces belonging to a body part named in ``hide_parts``.

        Each face takes the part of its (single-geom) vertices, so a config list like
        ``["wings"]`` drops every wing face from the overlay. Names not in
        :attr:`part_names` are ignored (so an unknown/empty list hides nothing).
        """
        n_faces = self.faces.shape[0]
        if not hide_parts or not self.part_names:
            return np.zeros(n_faces, dtype=bool)
        index = {name: i for i, name in enumerate(self.part_names)}
        hide_ids = [index[p] for p in hide_parts if p in index]
        if not hide_ids:
            return np.zeros(n_faces, dtype=bool)
        face_part = self.vert_part[self.faces[:, 0]]
        return np.isin(face_part, hide_ids)

    @functools.cached_property
    def _slot_verts(self) -> list[np.ndarray]:
        """Vertex-row indices for each slot (so posing is a few vectorized blocks)."""
        n = int(self.slot_prox.shape[0])
        return [np.flatnonzero(self.vert_slot == s) for s in range(n)]

    @functools.cached_property
    def _dist_anchor(self) -> Float[np.ndarray, "S 3"]:
        """``(n_slots, 3)`` neutral distal endpoint each leg bone is skinned to.

        Normally a bone's distal *keypoint*. But the **terminal** leg segment -- the
        tarsus, whose distal keypoint (the claw) is no other segment's proximal joint
        -- has a baked mesh that stops short of that keypoint, so the model skeleton's
        claw juts past the mesh tip. For such a segment the distal anchor is instead
        the segment mesh's own farthest point along the bone, so skinning stretches it
        out to land its tip on the live claw keypoint (where the skeleton draws it),
        removing the gap. Non-terminal / non-leg slots keep their distal keypoint (or
        ``NaN`` when there is none).
        """
        n = int(self.slot_prox.shape[0])
        anchor = np.full((n, 3), np.nan)
        prox = {int(p) for p in self.slot_prox.tolist() if p >= 0}
        for s in range(n):
            pi, di = int(self.slot_prox[s]), int(self.slot_dist[s])
            if pi < 0 or di < 0:
                continue
            a0, b0 = self.kp_neutral[pi], self.kp_neutral[di]
            anchor[s] = b0
            if di in prox:  # a mid-chain joint shared with the next segment: keep it
                continue
            axis = b0 - a0
            length = float(np.linalg.norm(axis))
            v = self.vertices[self._slot_verts[s]]
            if length < _EPS or v.shape[0] == 0:
                continue
            reach = float(
                (((v - a0) @ axis) / length**2).max()
            )  # farthest vertex / |axis|
            if reach > _EPS:
                anchor[s] = a0 + reach * axis
        return anchor

    def pose(
        self,
        pts3d: Float[np.ndarray, "P 3"],
        angles: Float[np.ndarray, "D"] | None = None,
        angle_names: list[str] | None = None,
        *,
        head_scale: float = 1.0,
        abdomen_scale: float = 1.0,
        body_scale: float | None = None,
    ) -> tuple[Float[np.ndarray, "Nv 3"], np.ndarray]:
        """Pose the mesh to one frame's fitted joints (and optional chain angles).

        Parameters
        ----------
        pts3d
            The fitted model joints for one frame ``(P, 3)`` in world coordinates
            (skeleton point order; NaN where a joint was not solved). Drives the
            rigid-body placement and the leg-segment skinning.
        angles, angle_names
            One frame's fitted joint angles and their names. When given, the head
            and abdomen mesh nodes articulate through the baked chain kinematics so
            the overlay head turns / abdomen curls with the fit; otherwise those
            nodes ride the rigid body at their neutral pose.
        head_scale, abdomen_scale
            Extra size multipliers for the head / abdomen meshes, about each chain's
            base, *on top of* the coxa-derived body scale. NeuroMechFly's head and
            abdomen are fixed model geometry (unlike the legs, which skin to the real
            keypoints), so these let the overlay match a fly whose head/abdomen differ
            in size (e.g. a fuller abdomen). ``1.0`` leaves them at the model size.
        body_scale
            The recording's fixed body scale (see
            :attr:`~deeperfly.inverse_kinematics.IKResult.body_scale`). When given,
            the rigid body + head + abdomen are placed at this constant size and the
            per-frame fit recovers only rotation + translation, so the body does not
            breathe with per-frame coxa noise. ``None`` re-fits the scale per frame.

        Returns
        -------
        verts : np.ndarray
            ``(Nv, 3)`` posed world vertices (NaN for un-poseable segments).
        valid_faces : np.ndarray
            ``(Nf,)`` bool mask of faces whose three vertices are all finite.
        """
        pts3d = np.asarray(pts3d, dtype=float)
        out = np.full_like(self.vertices, np.nan)

        rot, scale, trans = self._body_transform(pts3d, fixed_scale=body_scale)
        up = rot @ np.array([0.0, 0.0, 1.0])  # live dorsal axis fixes the bone roll
        node_xform = self._node_transforms(
            angles, angle_names, (head_scale, abdomen_scale)
        )

        for slot, rows in enumerate(self._slot_verts):
            if rows.size == 0:
                continue
            v = self.vertices[rows]
            if self.slot_chain[slot] >= 0:  # articulated head / abdomen node
                a, b = node_xform[
                    int(self.slot_chain[slot]), int(self.slot_depth[slot])
                ]
                out[rows] = scale * ((v @ a.T + b) @ rot.T) + trans
            elif self.slot_prox[slot] >= 0:  # leg bone -> skin between its endpoints
                a0 = self.kp_neutral[self.slot_prox[slot]]
                b0 = self._dist_anchor[slot]  # tarsus tip is stretched out to the claw
                a1 = pts3d[self.slot_prox[slot]]
                b1 = pts3d[self.slot_dist[slot]]
                if not (np.isfinite(a1).all() and np.isfinite(b1).all()):
                    continue  # an occluded leg segment stays NaN -> dropped
                out[rows] = _skin_bone(v, a0, b0, a1, b1, up)
            else:  # slot 0: rigid body (thorax / wings / halteres)
                out[rows] = scale * (v @ rot.T) + trans

        valid_faces = np.isfinite(out[self.faces]).all(axis=(1, 2))
        return out, valid_faces

    def _node_transforms(
        self,
        angles: np.ndarray | None,
        angle_names: list[str] | None,
        scales: tuple[float, float] = (1.0, 1.0),
    ) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]:
        """Model-frame affine ``(A, b)`` per ``(chain, depth)`` node from the angles.

        Missing angles (no IK chain fit, or NaN) give the articulation identity, so
        the node rides the rigid body at its neutral pose. ``scales`` (head, abdomen)
        additionally grows each chain's mesh about its base anchor -- folded into the
        affine as ``A' = f A`` and ``b' = f b + (1 - f) base``. The solved body plan
        bakes that same growth into its chain offsets, so the fitted angles and the
        nodes drawn from them describe one pose (see
        :mod:`deeperfly.inverse_kinematics.bodyplan`).

        Deliberately driven by the *unfiltered* packaged articulation and the angle
        *names*, not by the recording's body plan: chain index 0 is always the head and
        1 the abdomen here (that is what ``nmf_mesh.npz`` bakes into its node slots),
        and this way the overlay articulates for a result file written by any version --
        including one whose run fit only the abdomen, and one produced before body plans
        existed.
        """
        from .articulation import load_articulation
        from .forward import chain_affine

        wanted = {
            (int(c), int(d)) for c, d in zip(self.slot_chain, self.slot_depth) if c >= 0
        }
        out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        art = load_articulation()
        have_angles = angles is not None and angle_names is not None
        col = {name: i for i, name in enumerate(angle_names or [])}
        for chain_idx, depth in wanted:
            a, b = np.eye(3), np.zeros(3)
            if chain_idx < len(art.chains):
                chain = art.chains[chain_idx]
                if have_angles:
                    theta = np.array(
                        [
                            float(angles[col[n]]) if n in col else np.nan
                            for n in chain.dof_names
                        ]  # type: ignore[index]
                    )
                    if np.isfinite(theta).all():
                        a, b = chain_affine(chain, depth, theta)
                f = float(scales[chain_idx]) if chain_idx < len(scales) else 1.0
                if f != 1.0:  # grow the node about its chain base anchor
                    base = chain.anchors[0]
                    a, b = f * a, f * b + (1.0 - f) * base
            out[(chain_idx, depth)] = (a, b)
        return out

    def _body_transform(
        self,
        pts3d: Float[np.ndarray, "P 3"],
        fixed_scale: float | None = None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Body placement (R, s, t) from the neutral coxae to the live coxae.

        ``fixed_scale`` holds the body size at a constant (per-recording) value: the
        rotation is the same orthogonal fit, but the scale is forced and the
        translation re-derived from it, so the fit is rigid + a known scale instead of
        a full per-frame similarity (which let the body breathe with coxa noise).
        ``None`` fits the scale per frame (the legacy behaviour).
        """
        src = self.kp_neutral[self.coxa_idx]
        dst = pts3d[self.coxa_idx]
        good = np.isfinite(dst).all(axis=1)
        if int(good.sum()) < 3:  # too few anchors to place the body
            fallback = 1.0 if fixed_scale is None else float(fixed_scale)
            return np.eye(3), fallback, np.zeros(3)
        rot, scale, trans = _umeyama(src[good], dst[good])
        if fixed_scale is not None:
            scale = float(fixed_scale)
            trans = dst[good].mean(axis=0) - scale * (rot @ src[good].mean(axis=0))
        return rot, scale, trans


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
