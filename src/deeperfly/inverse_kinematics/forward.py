"""Forward kinematics in numpy: body plans, baked chains, and legs.

The IK *solve* is QuickIK's (see :mod:`deeperfly.inverse_kinematics`), but QuickIK's
Python bindings return only joint angles and a root pose -- its Rust
``evaluate_fwdkin`` is not exposed. Every consumer of the fit needs joint *positions*:
``IKResult.model_pts3d`` reprojects onto the raw views (the ``skeleton_model`` panel, the
GUI's model overlay) and the mesh overlay skins each leg bone between two fitted joints.
So deeperfly evaluates forward kinematics itself, here.

:class:`PlanKinematics` walks a QuickIK body plan (:mod:`deeperfly.inverse_kinematics.bodyplan`)
and is the counterpart to the solve. It reproduces QuickIK's own convention exactly
(``src/forward.rs``): a joint's ``offset_pos`` is applied in its parent's frame *after*
the parent's DOFs, a joint's own DOFs compose intrinsically in listed order, and a
joint's keypoint sits at its origin *before* its own DOFs -- a joint's DOFs move its
descendants, never itself.

:func:`chain_affine` and :func:`chain_fk` are the baked head/abdomen chains' kinematics
(a serial product of rotations about *neutral world* anchors, the convention
``the pack's ``articulation.json```` was baked in), and :func:`leg_fk` is a leg's. Those two
are algebraically the same transform as the corresponding subtree of a body plan --
which is what lets the mesh overlay keep posing its head/abdomen nodes from
:func:`chain_affine` while the angles come from a whole-body plan solve.

Everything here is pure numpy: no JAX, no QuickIK. That keeps the mesh overlay and the
GUI's static-fit path working on an install without the optional solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Float

__all__ = [
    "PlanKinematics",
    "axis_rmat",
    "chain_affine",
    "chain_fk",
    "leg_fk",
    "quat_to_rmat",
    "rmat_to_quat",
]

#: The rest direction a leg segment extends along in its parent joint's frame, for a
#: model that does not say. NeuroMechFly's is straight down in the leg-local frame,
#: whose z is dorsal; a model declares its own as ``rest_axis``
#: (:attr:`~deeperfly.inverse_kinematics.template.KinematicTemplate.rest_axis`).
REST_AXIS = np.array([0.0, 0.0, -1.0])

_EPS = 1e-12


def axis_rmat(
    axis: Float[np.ndarray, "3"], angle: float | Float[np.ndarray, "*batch"]
) -> Float[np.ndarray, "*batch 3 3"]:
    """Rotation(s) by ``angle`` about the fixed unit ``axis`` (Rodrigues, sin/cos form).

    ``R = I + sin(t) K + (1 - cos(t)) K^2`` with ``K = skew(axis)``. Smooth at
    ``t = 0`` (a joint angle is routinely exactly zero at the rest pose), unlike the
    general axis-angle form which divides by the rotation-vector norm.

    Parameters
    ----------
    axis
        ``(3,)`` unit rotation axis.
    angle
        Rotation angle(s) in radians; any shape, including a scalar.

    Returns
    -------
    np.ndarray
        ``angle.shape + (3, 3)`` rotation matrices (``(3, 3)`` for a scalar angle).
    """
    ax, ay, az = (float(v) for v in np.asarray(axis, dtype=float))
    k = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    angle = np.asarray(angle, dtype=float)
    s = np.sin(angle)[..., None, None]
    c = np.cos(angle)[..., None, None]
    return np.eye(3) + s * k + (1.0 - c) * (k @ k)


def quat_to_rmat(quat: Float[np.ndarray, "4"]) -> Float[np.ndarray, "3 3"]:
    """Rotation matrix of a quaternion given as ``(w, x, y, z)`` (the body-plan order)."""
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=float))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < _EPS:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rmat_to_quat(rmat: Float[np.ndarray, "3 3"]) -> Float[np.ndarray, "4"]:
    """Quaternion ``(w, x, y, z)`` of a rotation matrix (Shepperd's method).

    Picks the branch off the largest diagonal term rather than always solving for
    ``w``, so it stays conditioned at a 180-degree rotation (where ``w -> 0``).
    """
    m = np.asarray(rmat, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        )
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
        q = np.zeros(4)
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q /= max(float(np.linalg.norm(q)), _EPS)
    return q if q[0] >= 0.0 else -q  # canonical sign, so equal rotations compare equal


def leg_fk(
    angles: Float[np.ndarray, "D"],
    axes: Float[np.ndarray, "D 3"],
    seglens: Float[np.ndarray, "J"],
    dof_counts: tuple[int, ...],
    *,
    rest_axis: Float[np.ndarray, "3"] | None = None,
    quats: Float[np.ndarray, "J 4"] | None = None,
) -> Float[np.ndarray, "J 3"]:
    """One leg's joint positions in the leg-local frame, from its joint angles.

    Each joint enters its parent's post-DOF frame through a constant rotation
    (``quats``) and then rotates about its DOF axes in order; the *next* joint's
    segment offsets the position along the parent's frame in the ``rest_axis``
    direction. With every angle zero this is the rest pose -- for NeuroMechFly, a
    straight leg pointing down. The convention is the body plan's own
    (:meth:`PlanKinematics.joint_positions`), which is QuickIK's.

    The numpy counterpart of the (now-removed) JAX kernel, and identical to the
    corresponding subtree of a body plan whose segment offsets are
    ``[0, 0, -seglen]``.

    Parameters
    ----------
    angles
        ``(D,)`` joint angles in radians, in chain order.
    axes
        ``(D, 3)`` unit rotation axes, one per DOF in chain order.
    seglens
        ``(J,)`` segment length leading *into* each joint; ``seglens[0]`` is unused
        (the root sits at the origin).
    dof_counts
        DOFs at each joint, in chain order (e.g. ``(3, 2, 1, 1, 0)``).
    rest_axis
        ``(3,)`` the direction a segment extends along. ``None`` = :data:`REST_AXIS`.
    quats
        ``(J, 4)`` each joint's constant ``(w, x, y, z)`` rotation out of its parent's
        post-DOF frame. ``None`` = identity at every joint, which is NeuroMechFly.

    Returns
    -------
    np.ndarray
        ``(J, 3)`` joint positions in the leg-local frame.
    """
    angles = np.asarray(angles, dtype=float)
    axes = np.asarray(axes, dtype=float)
    seglens = np.asarray(seglens, dtype=float)
    axis = REST_AXIS if rest_axis is None else np.asarray(rest_axis, dtype=float)
    quats = None if quats is None else np.asarray(quats, dtype=float)
    rot = np.eye(3)
    pos = np.zeros(3)
    out = []
    d = 0
    for j, n_dofs in enumerate(dof_counts):
        pos = pos + rot @ (axis * seglens[j])
        out.append(pos)
        if quats is not None:
            rot = rot @ quat_to_rmat(quats[j])
        for _ in range(n_dofs):
            rot = rot @ axis_rmat(axes[d], angles[d])
            d += 1
    return np.stack(out) if out else np.zeros((0, 3))


def chain_affine(
    chain, depth: int, theta: Float[np.ndarray, "D"]
) -> tuple[Float[np.ndarray, "3 3"], Float[np.ndarray, "3"]]:
    """The model-frame affine ``(A, b)`` carrying a depth-``d`` point: ``A p + b``.

    The baked head/abdomen chains rotate about their *neutral world* anchors (the
    fixed/spatial-frame convention ``the pack's ``articulation.json```` was baked in, which
    reproduces the MJCF frames exactly): with ``A_d = A_{d-1} R_d`` and
    ``b_d = A_{d-1}(c_d - R_d c_d) + b_{d-1}``, a point rigidly attached at depth
    ``d`` maps to ``A_d p + b_d``. At all-zero angles every ``R_d = I``, so points
    stay at their neutral positions.

    Used by the mesh overlay to pose the head / abdomen node meshes
    (:meth:`deeperfly.inverse_kinematics.mesh.ModelMesh._node_transforms`).

    Parameters
    ----------
    chain
        The baked chain (:class:`~deeperfly.inverse_kinematics.articulation.Chain`);
        its ``anchors`` and ``axes`` supply the geometry.
    depth
        How many proximal joints move the point.
    theta
        The chain's joint angles in radians.
    """
    a_cum = np.eye(3)
    b_cum = np.zeros(3)
    for i in range(int(depth)):
        r = axis_rmat(chain.axes[i], float(theta[i]))
        c = np.asarray(chain.anchors[i], dtype=float)
        b_cum = a_cum @ (c - r @ c) + b_cum
        a_cum = a_cum @ r
    return a_cum, b_cum


def chain_fk(
    anchors: Float[np.ndarray, "D 3"],
    axes: Float[np.ndarray, "D 3"],
    depths: tuple[int, ...],
    angles: Float[np.ndarray, "D"],
    markers: Float[np.ndarray, "M 3"],
) -> Float[np.ndarray, "M 3"]:
    """Marker positions of a baked chain, in the model frame.

    The vectorized-over-markers form of :func:`chain_affine`: builds the cumulative
    affine once per depth and applies each marker's at its own depth.

    Parameters
    ----------
    anchors, axes
        ``(D, 3)`` neutral world anchor and unit axis of each joint, in chain order.
    depths
        ``(M,)`` chain depth of each marker.
    angles
        ``(D,)`` joint angles in radians.
    markers
        ``(M, 3)`` neutral marker positions.

    Returns
    -------
    np.ndarray
        ``(M, 3)`` posed marker positions in the model frame.
    """
    anchors = np.asarray(anchors, dtype=float)
    axes = np.asarray(axes, dtype=float)
    angles = np.asarray(angles, dtype=float)
    markers = np.asarray(markers, dtype=float)
    a_at = [np.eye(3)]
    b_at = [np.zeros(3)]
    for i in range(anchors.shape[0]):
        r = axis_rmat(axes[i], float(angles[i]))
        c = anchors[i]
        b_at.append(a_at[-1] @ (c - r @ c) + b_at[-1])
        a_at.append(a_at[-1] @ r)
    if markers.shape[0] == 0:
        return np.zeros((0, 3))
    return np.stack(
        [a_at[int(d)] @ markers[k] + b_at[int(d)] for k, d in enumerate(depths)]
    )


@dataclass(frozen=True)
class PlanKinematics:
    """A QuickIK body plan compiled for batched numpy forward kinematics.

    Built once per plan (:meth:`from_plan`) and then evaluated over a whole recording
    (:meth:`joint_positions`). The plan is small (tens of joints) but a recording can
    be thousands of frames, so the traversal loops over *joints* in Python and
    vectorizes over *frames* in numpy.

    Attributes
    ----------
    names
        ``(N,)`` joint names, in body-plan array order (which is also the observation
        order QuickIK expects and the DOF state layout).
    parent
        ``(N,)`` parent joint index, ``-1`` for the root.
    order
        ``(N,)`` a traversal order in which every joint follows its parent.
    offset_pos, offset_quat
        ``(N, 3)`` / ``(N, 4)`` each joint's constant offset from its parent's
        post-DOF frame (quaternion in ``(w, x, y, z)``).
    dof_start, dof_count
        ``(N,)`` each joint's slice of the flat DOF vector.
    dof_axis
        ``(D, 3)`` each DOF's axis in its own joint's post-``offset_quat`` frame.
    dof_hinge
        ``(D,)`` True for a hinge (rotation), False for a slide (translation).
    """

    names: tuple[str, ...]
    parent: np.ndarray
    order: np.ndarray
    offset_pos: np.ndarray
    offset_quat: np.ndarray
    dof_start: np.ndarray
    dof_count: np.ndarray
    dof_axis: np.ndarray
    dof_hinge: np.ndarray

    @classmethod
    def from_plan(cls, plan: dict) -> "PlanKinematics":
        """Compile a body-plan mapping (the parsed JSON) for evaluation.

        Raises
        ------
        ValueError
            If the plan has no single root, names a parent that is not a joint, or
            its joints do not form a tree.
        """
        joints = list(plan["joints"])
        names = tuple(str(j["name"]) for j in joints)
        index = {name: i for i, name in enumerate(names)}
        if len(index) != len(names):
            raise ValueError("body plan has duplicate joint names")
        parent = np.full(len(joints), -1, dtype=np.int64)
        for i, j in enumerate(joints):
            p = j["parent"]
            if p is None:
                continue
            if str(p) not in index:
                raise ValueError(
                    f"body-plan joint {names[i]!r} names unknown parent {p!r}"
                )
            parent[i] = index[str(p)]
        if int((parent < 0).sum()) != 1:
            raise ValueError(
                f"body plan needs exactly one root joint (parent=null), "
                f"found {int((parent < 0).sum())}"
            )

        # A traversal order with every parent before its child. QuickIK's own parser
        # requires the array to already be in such an order, but we do not depend on
        # that -- a plan read back from an older result file should still evaluate.
        order: list[int] = []
        pending = {i: int(parent[i]) for i in range(len(joints))}
        placed: set[int] = set()
        while pending:
            ready = [i for i, p in pending.items() if p < 0 or p in placed]
            if not ready:
                raise ValueError("body-plan joints do not form a tree (cycle)")
            for i in ready:
                order.append(i)
                placed.add(i)
                del pending[i]

        axes: list[list[float]] = []
        hinge: list[bool] = []
        starts = np.zeros(len(joints), dtype=np.int64)
        counts = np.zeros(len(joints), dtype=np.int64)
        for i, j in enumerate(joints):
            dofs: list[dict] = list(j.get("dofs") or ())
            starts[i] = len(axes)
            counts[i] = len(dofs)
            for d in dofs:
                axes.append([float(v) for v in d["axis"]])
                hinge.append(str(d["type"]) == "hinge")
        return cls(
            names=names,
            parent=parent,
            order=np.asarray(order, dtype=np.int64),
            offset_pos=np.asarray(
                [[float(v) for v in j["offset_pos"]] for j in joints], dtype=float
            ),
            offset_quat=np.asarray(
                [[float(v) for v in j["offset_quat"]] for j in joints], dtype=float
            ),
            dof_start=starts,
            dof_count=counts,
            dof_axis=np.asarray(axes, dtype=float).reshape(-1, 3),
            dof_hinge=np.asarray(hinge, dtype=bool),
        )

    @property
    def n_joints(self) -> int:
        return len(self.names)

    @property
    def n_dofs(self) -> int:
        return int(self.dof_axis.shape[0])

    def dof_names(self, plan: dict) -> list[str]:
        """The ``x-deeperfly-angle`` label of every DOF, in flat DOF order.

        The body-plan schema's own per-DOF ``name`` is documented as parser-ignored,
        so :mod:`deeperfly.inverse_kinematics.bodyplan` carries the flygym angle name
        in an ``x-`` key instead; this reads it back out.
        """
        out: list[str] = []
        for j in plan["joints"]:
            for d in j.get("dofs") or ():
                out.append(str(d["x-deeperfly-angle"]))
        return out

    def joint_positions(
        self,
        dof_angles: Float[np.ndarray, "T D"],
        root_pos: Float[np.ndarray, "T 3"] | None = None,
        root_rot: Float[np.ndarray, "T 4"] | None = None,
    ) -> Float[np.ndarray, "T N 3"]:
        """Every joint's position, for a batch of poses.

        Mirrors QuickIK's ``evaluate_fwdkin``: the root frame is
        ``(root_pos, root_rot)`` (identity for a fixed-base plan, where the solver
        leaves them untouched), a child's ``offset_pos`` is applied in its parent's
        *post-DOF* frame, and a joint's own keypoint is placed *before* its own DOFs.

        NaN angles propagate to NaN positions for that joint and everything distal to
        it, which is how an unfittable limb stays visibly absent rather than being
        drawn at some invented pose.

        Parameters
        ----------
        dof_angles
            ``(T, D)`` joint angles (radians) / slide values, in plan DOF order. A
            single ``(D,)`` pose is accepted and returns ``(1, N, 3)``.
        root_pos, root_rot
            ``(T, 3)`` root position and ``(T, 4)`` root rotation ``(w, x, y, z)``.
            ``None`` means the origin / identity.

        Returns
        -------
        np.ndarray
            ``(T, N, 3)`` joint positions, in body-plan joint order.
        """
        angles = np.atleast_2d(np.asarray(dof_angles, dtype=float))
        if angles.shape[1] != self.n_dofs:
            raise ValueError(
                f"expected {self.n_dofs} DOF angles per frame, got {angles.shape[1]}"
            )
        n_frames = angles.shape[0]

        if root_rot is None:
            root_mat = np.broadcast_to(np.eye(3), (n_frames, 3, 3))
        else:
            rr = np.atleast_2d(np.asarray(root_rot, dtype=float))
            root_mat = np.stack([quat_to_rmat(q) for q in rr])
        if root_pos is None:
            root_org = np.zeros((n_frames, 3))
        else:
            root_org = np.atleast_2d(np.asarray(root_pos, dtype=float))

        n = self.n_joints
        frame_rot = np.empty((n, n_frames, 3, 3))
        frame_org = np.empty((n, n_frames, 3))
        out = np.empty((n, n_frames, 3))

        for j in self.order:
            p = int(self.parent[j])
            prot = root_mat if p < 0 else frame_rot[p]
            porg = root_org if p < 0 else frame_org[p]
            # The joint's own keypoint: parent frame + this joint's constant offset,
            # unmoved by its own DOFs.
            own = porg + prot @ self.offset_pos[j]
            rot = prot @ quat_to_rmat(self.offset_quat[j])
            org = own
            start, count = int(self.dof_start[j]), int(self.dof_count[j])
            for d in range(start, start + count):
                value = angles[:, d]
                if self.dof_hinge[d]:
                    rot = rot @ axis_rmat(self.dof_axis[d], value)
                else:
                    org = org + (rot @ self.dof_axis[d]) * value[:, None]
            out[j] = own
            frame_rot[j] = rot
            frame_org[j] = org
        return np.moveaxis(out, 0, 1)
