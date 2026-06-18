"""Forward kinematics for a serial leg chain, in JAX (grad-friendly).

A leg is a chain of joints from the body-fixed thorax-coxa (ThC) outward to the
claw. Each joint rotates the running frame about its revolute DOF axes (applied in
order) and the *next* joint's segment then offsets the position along the rest
direction ``-z`` of that running frame. With every angle zero the chain reproduces
the rest pose: a straight leg pointing down (``-z``) in the leg-local frame whose
``x`` is anterior, ``y`` is left and ``z`` is dorsal (up).

The model is intentionally tiny (a leg is 5 joints / up to 7 DOFs), so the
per-joint Python loop is fully unrolled inside the jitted kernel. Reverse-mode
autodiff (:func:`jax.jacrev`) over the flat angle vector gives the analytic
Jacobian the bounded least-squares solver (:mod:`deeperfly.inverse_kinematics.core`)
feeds to scipy -- the same JAX-Jacobian-into-scipy pattern as bundle adjustment.
"""

from __future__ import annotations

import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

__all__ = ["make_leg_fk", "make_leg_kernels", "make_chain_fk", "make_chain_kernels"]

#: The rest direction every segment extends along in its parent joint's frame
#: (straight down in the leg-local frame).
_REST = jnp.array([0.0, 0.0, -1.0])


def _axis_rmat(axis: Float[Array, "3"], angle: Float[Array, ""]) -> Float[Array, "3 3"]:
    """Rotation by ``angle`` about the fixed unit ``axis`` (Rodrigues, sin/cos form).

    ``R = I + sin(t) K + (1 - cos(t)) K^2`` with ``K = skew(axis)``. Unlike the
    general axis-angle :func:`deeperfly.geometry.rvec_to_rmat_one` (which divides by
    the rotation-vector norm and has a ``jnp.where`` whose unused branch poisons the
    gradient at ``t = 0``), this is smooth in ``angle`` everywhere -- essential
    because a joint angle is routinely exactly zero (the rest pose) and the solver
    differentiates through it. ``axis`` must be a unit vector.
    """
    ax, ay, az = axis[0], axis[1], axis[2]
    k = jnp.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
    s, c = jnp.sin(angle), jnp.cos(angle)
    return jnp.eye(3) + s * k + (1.0 - c) * (k @ k)


def make_leg_fk(
    dof_counts: tuple[int, ...],
) -> Callable[
    [Float[Array, "D"], Float[Array, "D 3"], Float[Array, "J"]], Float[Array, "J 3"]
]:
    """Build the forward-kinematics function for a chain with a fixed DOF layout.

    ``dof_counts`` is static (it comes from the template, not the data), so the
    returned ``fk`` unrolls the chain and is safe to :func:`jax.jit` and
    :func:`jax.jacrev` over the angle vector.

    Parameters
    ----------
    dof_counts
        Number of revolute DOFs at each joint, in chain order (e.g.
        ``(3, 2, 1, 1, 0)`` for ThC, CTr, FTi, TiTa, Claw). Its length is the
        joint count ``J`` and its sum is the total DOF count ``D``.

    Returns
    -------
    fk : callable
        ``fk(angles, axes, seglens) -> (J, 3)`` joint positions in the leg-local
        frame. ``angles`` is ``(D,)``; ``axes`` is ``(D, 3)`` unit rotation axes
        (one per DOF, in chain order); ``seglens`` is ``(J,)`` the segment length
        leading *into* each joint (``seglens[0]`` is unused -- the root sits at the
        origin).
    """
    n_joints = len(dof_counts)

    def fk(
        angles: Float[Array, "D"],
        axes: Float[Array, "D 3"],
        seglens: Float[Array, "J"],
    ) -> Float[Array, "J 3"]:
        rot = jnp.eye(3)
        pos = jnp.zeros(3)
        positions = []
        d = 0
        for j in range(n_joints):
            # Offset from the previous joint along the parent frame's rest axis.
            pos = pos + rot @ (_REST * seglens[j])
            positions.append(pos)
            # This joint's DOFs rotate the running frame for the segments below it.
            for _ in range(dof_counts[j]):
                rot = rot @ _axis_rmat(axes[d], angles[d])
                d += 1
        return jnp.stack(positions)

    return fk


def make_chain_fk(
    anchors: Float[Array, "D 3"],
    axes: Float[Array, "D 3"],
    depths: tuple[int, ...],
) -> Callable[[Float[Array, "D"], Float[Array, "M 3"]], Float[Array, "M 3"]]:
    """Build forward kinematics for a baked revolute chain (the head / abdomen).

    Unlike a leg (whose joints *are* tracked keypoints with measured segment
    lengths), the head and abdomen articulate joints that are not keypoints, with
    fixed model geometry baked from the MJCF: joint ``i`` rotates about the world
    anchor ``anchors[i]`` and axis ``axes[i]`` of the neutral model. A *marker* (a
    tracked keypoint rigidly attached at chain ``depth`` -- the number of proximal
    joints that move it) is carried by the running transform at that depth.

    The chain is a serial product of rotations about the *neutral* anchors (the
    fixed/spatial-frame convention), which reproduces the MJCF frames exactly: with
    cumulative ``A_d = A_{d-1} R_d`` and ``b_d = A_{d-1}(c_d - R_d c_d) + b_{d-1}``
    (``A_0 = I``, ``b_0 = 0``), a marker at depth ``d`` maps to ``A_d m + b_d``. At
    all-zero angles every ``R_d = I`` so markers stay at their neutral positions.

    Parameters
    ----------
    anchors, axes
        ``(D, 3)`` neutral world anchor and unit axis of each joint, in chain order.
    depths
        ``(M,)`` static chain depth of each marker (e.g. ``(3, 3)`` for the two
        antennae of the 3-DOF head, ``(2, 2, 4, 4, 5, 5)`` for the abdomen markers).

    Returns
    -------
    fk : callable
        ``fk(angles, markers) -> (M, 3)`` marker positions in the model frame.
        ``angles`` is ``(D,)`` and ``markers`` is ``(M, 3)`` neutral marker points.
    """
    anchors = jnp.asarray(anchors, dtype=float)
    axes = jnp.asarray(axes, dtype=float)
    depths = tuple(int(d) for d in depths)
    n = anchors.shape[0]

    def fk(
        angles: Float[Array, "D"], markers: Float[Array, "M 3"]
    ) -> Float[Array, "M 3"]:
        a_cum = jnp.eye(3)
        b_cum = jnp.zeros(3)
        a_at = [a_cum]
        b_at = [b_cum]
        for i in range(n):
            r = _axis_rmat(axes[i], angles[i])
            c = anchors[i]
            b_cum = a_cum @ (c - r @ c) + b_cum
            a_cum = a_cum @ r
            a_at.append(a_cum)
            b_at.append(b_cum)
        return jnp.stack([a_at[d] @ markers[k] + b_at[d] for k, d in enumerate(depths)])

    return fk


def make_chain_kernels(
    anchors: Float[Array, "D 3"], axes: Float[Array, "D 3"], depths: tuple[int, ...]
):
    """Jitted forward-kinematics value and angle-Jacobian for a baked chain.

    Cached on the (anchor, axis, depth) layout, since rebuilding the closure forces
    a fresh JAX trace+compile -- the per-frame GUI refit calls this every solve.

    Returns
    -------
    fk, jac : callable
        ``fk(angles, markers) -> (M, 3)`` and its Jacobian w.r.t. ``angles``
        ``jac(angles, markers) -> (M, 3, D)``, both :func:`jax.jit`-wrapped.
    """
    anchors = np.asarray(anchors, dtype=float)
    axes = np.asarray(axes, dtype=float)
    return _chain_kernels(
        anchors.tobytes(), axes.tobytes(), tuple(int(d) for d in depths), anchors.shape
    )


@functools.lru_cache(maxsize=16)
def _chain_kernels(anchor_bytes: bytes, axis_bytes: bytes, depths: tuple, shape: tuple):
    anchors = np.frombuffer(anchor_bytes).reshape(shape)
    axes = np.frombuffer(axis_bytes).reshape(shape)
    fk = make_chain_fk(anchors, axes, depths)
    return jax.jit(fk), jax.jit(jax.jacrev(fk, argnums=0))


@functools.lru_cache(maxsize=16)
def make_leg_kernels(dof_counts: tuple[int, ...]):
    """Jitted forward-kinematics value and angle-Jacobian for a DOF layout.

    Cached on ``dof_counts`` so the six legs (which share a layout) compile once and
    the per-frame GUI refit reuses the compiled kernels instead of re-tracing.

    Returns
    -------
    fk, jac : callable
        ``fk(angles, axes, seglens) -> (J, 3)`` and its Jacobian w.r.t. ``angles``
        ``jac(angles, axes, seglens) -> (J, 3, D)``, both :func:`jax.jit`-wrapped.
    """
    fk = make_leg_fk(dof_counts)
    fk_jit = jax.jit(fk)
    jac_jit = jax.jit(jax.jacrev(fk, argnums=0))
    return fk_jit, jac_jit
